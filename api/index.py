import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import asyncio
from dotenv import load_dotenv
import csv
import httpx # For making asynchronous API calls

# Load environment variables from .env file for local development
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Define Pydantic models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source: str

# Initialize FastAPI app
app = FastAPI()

# Global variables
model = None
faiss_index = None
df_faq = None
DIRECT_MATCH_THRESHOLD = 0.95
RELEVANCE_THRESHOLD = 0.2
TOP_K_FAQS = 3  # Retrieve top 3 FAQs for better context

# Get the Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def load_data():
    """
    Load the FAQ data from a CSV file using Python's csv module for robustness.
    """
    try:
        file_path = ""
        if os.path.exists("data.csv"):
            file_path = "data.csv"
        elif os.path.exists("../data.csv"):
            file_path = "../data.csv"
        else:
            raise FileNotFoundError("data.csv not found.")
        
        data = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        
        df = pd.DataFrame(data)
        logging.info("FAQ data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def initialize_models_and_index():
    """
    Load the Sentence-Transformer model and create the FAISS index.
    The LLM is now cloud-based and does not need to be loaded here.
    """
    global model
    try:
        # Load Sentence-Transformer model for semantic search
        if model is None:
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            logging.info("Sentence-Transformer model loaded successfully.")
    except Exception as e:
        logging.error(f"Error initializing models: {e}")
        model = None

def create_faiss_index(df):
    """
    Create a FAISS index from the FAQ questions.
    """
    global faiss_index
    try:
        questions = df['question'].tolist()
        question_embeddings = model.encode(questions)
        d = question_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(d)
        faiss_index.add(np.array(question_embeddings).astype('float32'))
        logging.info("FAISS index created successfully.")
    except Exception as e:
        logging.error(f"Error creating FAISS index: {e}")
        faiss_index = None

# Event handler for application startup
@app.on_event("startup")
async def startup_event():
    """
    This function runs when the FastAPI application starts.
    """
    global df_faq
    initialize_models_and_index()
    df_faq = load_data()
    if df_faq is not None and model is not None:
        create_faiss_index(df_faq)
    else:
        logging.error("Startup failed. Missing data or models.")

async def generate_gemini_response(prompt: str) -> str:
    """
    Generates a response using the Google Gemini API.
    """
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not set. Please add it to your .env file.")
        return "An error occurred: API key is missing."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts":[{"text": prompt}]}]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            
            if 'candidates' in data and data['candidates']:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    return candidate['content']['parts'][0]['text']
            
            logging.error(f"Gemini API response was not as expected: {data}")
            return "An error occurred while getting a response from the LLM."

        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return "An error occurred while communicating with the LLM API."
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return "An unexpected error occurred."

@app.post("/api/query", response_model=QueryResponse)
async def query_faq(request: QueryRequest):
    """
    Main API endpoint to handle user queries.
    """
    if faiss_index is None or df_faq is None or model is None:
        raise HTTPException(status_code=503, detail="Service not yet ready. Please try again in a moment.")

    try:
        query_embedding = model.encode([request.query])
        # Retrieve the top K most relevant FAQs
        D, I = faiss_index.search(np.array(query_embedding).astype('float32'), k=TOP_K_FAQS)
        
        # Best match is always at index 0
        best_match_idx = I[0][0]
        distance = D[0][0]
        similarity_score = 1 - (distance / (distance + 1))
        
        logging.info(f"Found best match with score: {similarity_score}")
        
        faq_answer = str(df_faq.iloc[best_match_idx]['answer'])
        faq_question = str(df_faq.iloc[best_match_idx]['question'])

        if similarity_score >= DIRECT_MATCH_THRESHOLD and faq_answer.strip():
            # If a very high relevance match is found, return the direct FAQ answer.
            return QueryResponse(
                answer=faq_answer,
                source=f"FAQ Match: '{faq_question}'"
            )
        else:
            # If not a direct match, use LLM with all relevant contexts.
            context_list = []
            for i, idx in enumerate(I[0]):
                context_list.append({
                    "question": str(df_faq.iloc[idx]['question']),
                    "answer": str(df_faq.iloc[idx]['answer'])
                })
            
            context_str = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in context_list])

            prompt = (
                f"You are a helpful assistant. Use the following context(s) to answer the user's question. "
                f"If the context(s) do not contain the answer, state that you cannot answer. "
                f"Your answer must be concise and to the point.\n\n"
                f"Contexts:\n{context_str}\n\n"
                f"User Question: {request.query}"
            )
            llm_answer = await generate_gemini_response(prompt)
            
            # Check if the LLM's response indicates it couldn't find an answer
            if "i cannot answer" in llm_answer.lower() or "not in the provided context" in llm_answer.lower():
                custom_message = "I cannot answer that question. I am a specialized FAQ model and can only provide information from the provided data."
                return QueryResponse(
                    answer=custom_message,
                    source="Generated by LLM (No relevant FAQ found)"
                )
            
            # If a useful answer was generated, return it
            source_question = str(df_faq.iloc[best_match_idx]['question'])
            return QueryResponse(
                answer=llm_answer,
                source=f"Generated by LLM with context from relevant FAQs (best match: '{source_question}')"
            )

    except Exception as e:
        logging.error(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

