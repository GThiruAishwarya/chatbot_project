import os
import csv
import uvicorn
import httpx
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Step 1: Configuration and Setup ---

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not found.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    user_prompt: str

documents = []
embeddings = None
index = None
client = httpx.AsyncClient()

# --- Step 2: Functions for Embeddings and RAG ---

async def get_embedding(text: str):
    """
    Generates an embedding for a given text using the Gemini API's embedding model.
    """
    # Corrected URL for the embedding model
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={API_KEY}"
    
    # Corrected payload structure
    payload = {
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    try:
        response = await client.post(api_url, json=payload, timeout=60.0)
        response.raise_for_status()
        embedding_data = response.json()
        return np.array(embedding_data['embedding']['values'], dtype='float32')
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Embedding API request failed: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during embedding: {str(e)}")

async def get_llm_response(prompt_with_context: str):
    """
    Generates a text response from the Gemini API using the RAG prompt.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_with_context
            }]
        }]
    }
    try:
        response = await client.post(api_url, json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        if "candidates" in data and data["candidates"]:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return "Sorry, I couldn't generate a response."
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"LLM API request failed: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during LLM generation: {str(e)}")

# --- Step 3: Application Startup (Lifecycle events) ---

@app.on_event("startup")
async def startup_event():
    """
    This function runs once when the application starts.
    It's used to load data and build the FAISS index.
    """
    global documents, embeddings, index
    try:
        with open("api/data.csv", mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                documents.append(row)
        
        # Create embeddings for all answers in the CSV.
        answer_texts = [doc['answer'] for doc in documents]
        embeddings = np.array([await get_embedding(text) for text in answer_texts], dtype='float32')
        
        # Build a FAISS index for fast similarity search.
        dimension = embeddings.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        index.add_with_ids(embeddings, np.arange(len(documents)))

        print("Data loaded and FAISS index built successfully.")

    except Exception as e:
        print(f"Error during startup: {e}")
        raise

# --- Step 4: API Endpoints ---

@app.get("/")
def read_root():
    """
    A simple test endpoint.
    """
    return {"status": "ok", "message": "Backend is running!"}

@app.post("/generate")
async def generate_response(prompt: Prompt):
    """
    Endpoint to handle user queries using a RAG pipeline with a safety guardrail.
    """
    user_query = prompt.user_prompt.lower().strip()

    # Strategy 1: Direct Lookup for exact or near-exact matches.
    for doc in documents:
        if user_query in doc['question'].lower() or user_query in doc['answer'].lower():
            return {"response": doc['answer']}
    
    # Strategy 2: RAG Pipeline for semantic matches with a confidence threshold.
    try:
        query_embedding = await get_embedding(user_query)
        
        # Search the FAISS index for the top 3 most similar documents to get more context.
        D, I = index.search(np.array([query_embedding]), k=3)
        
        distance_threshold = 0.6
        if D[0][0] > distance_threshold:
            return {"response": "I am a FAQ based chatbot. I can only answer questions related to my knowledge base. Please ask a question related to the topics I was trained on."}

        # Retrieve relevant documents and combine their content.
        retrieved_contexts = [documents[i]['answer'] for i in I[0] if D[0][I[0].tolist().index(i)] <= distance_threshold]
        combined_context = " ".join(retrieved_contexts)
        
        # Construct the RAG prompt for the LLM.
        rag_prompt = (
            "You are a friendly and helpful assistant for a FAQ chatbot. "
            "Provide a comprehensive answer to the user's question using the provided context. "
            "If the question cannot be answered from the context, "
            "politely say that you cannot provide an answer. "
            f"\n\nContext: {combined_context}\n\nQuestion: {user_query}"
        )

        llm_response = await get_llm_response(rag_prompt)
        return {"response": llm_response}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"RAG process failed, falling back to a generic error message. Error: {e}")
        return {"response": "I'm sorry, but I'm currently unable to process your request. Please try again later."}
