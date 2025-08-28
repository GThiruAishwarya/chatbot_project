import os
import uvicorn
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# This is a critical step: the API key must be stored securely as a Render secret.
# NEVER hardcode your API key in the file.
API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is set at startup.
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your Render dashboard.")

# This is the FastAPI instance that the uvicorn server will run.
# The name 'app' is mandatory for the startup command to work.
app = FastAPI()

# Add CORS middleware to allow the frontend to communicate with this backend.
# The allow_origins should be updated with your frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development. Be more specific in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This is the data model for the incoming request from the frontend.
class Prompt(BaseModel):
    user_prompt: str

@app.get("/")
def read_root():
    """
    A simple test endpoint to verify the backend is running.
    """
    return {"status": "ok", "message": "Backend is running!"}

@app.post("/generate")
async def generate_response(prompt: Prompt):
    """
    Endpoint to generate an embedding for a given text using the Gemini API.
    This offloads the memory-intensive work to Google's servers.
    """
    # Define the API endpoint and headers.
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
    headers = {
        "Content-Type": "application/json",
    }
    
    # Construct the payload for the API call. We are asking for an embedding.
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt.user_prompt
            }]
        }],
        "generationConfig": {
            # This is crucial for getting embedding as an output
            "responseModalities": ["TEXT"]
        },
        "tools": [
            {"google_search": {}}
        ]
    }
    
    try:
        # Use httpx to make an asynchronous POST request to the API.
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
        
        # Parse the JSON response.
        data = response.json()
        
        # Check if the response contains the expected embedding.
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"] and len(candidate["content"]["parts"]) > 0:
                # The Gemini API returns a generated text response.
                # In this specific case, we'll just return the generated text.
                # To get an embedding, you would use a different endpoint:
                # https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent
                # For this demonstration, we are just showing API usage for a text response.
                generated_text = candidate["content"]["parts"][0]["text"]
                return {"response": generated_text}
            
        # If the expected data is not found, raise an error.
        raise HTTPException(status_code=500, detail="Gemini API response format is invalid.")

    except httpx.HTTPStatusError as e:
        # Handle API errors gracefully.
        raise HTTPException(status_code=e.response.status_code, detail=f"API request failed: {e.response.text}")
    except Exception as e:
        # Handle other unexpected errors.
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# This ensures uvicorn runs the app when the file is executed directly.
# The Render deployment service will handle this automatically.
if __name__ == "__main__":
    uvicorn.run("api.index:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
