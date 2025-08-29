import os
import streamlit as st
import requests
import re
import asyncio

# --- Step 1: Configuration and API Setup ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Step 2: Helper Functions for API Interaction and Query Splitting ---
async def get_response(user_prompt: str):
    """
    Sends a single user prompt to the backend and returns the response.
    """
    try:
        url = f"{BACKEND_URL}/generate"
        response = requests.post(url, json={"user_prompt": user_prompt}, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "No response from backend.")
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e.response.text}")
        return "An error occurred with the backend."
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {e}")
        return "Failed to connect to the backend."

def split_query(query: str):
    """
    Splits a multi-part user query into a list of individual questions.
    Uses regex to split by punctuation like commas, and question marks.
    """
    # A simple regex to split by common delimiters.
    # It also handles cases with multiple question marks or periods.
    parts = re.split(r'[.,;?]\s*', query.strip())
    # Filter out any empty strings that might result from splitting.
    return [part for part in parts if part]

# --- Step 3: Streamlit Application Layout and Logic ---
st.set_page_config(page_title="FAQ Chatbot", layout="wide")
st.title("FAQ Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input loop
if prompt := st.chat_input("Ask a question about our business..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the user's prompt
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Split the user's question into parts
            query_parts = split_query(prompt)
            
            # Use a list to store responses for each part
            all_responses = []

            if len(query_parts) > 1:
                # Handle multi-part questions
                for part in query_parts:
                    response_text = asyncio.run(get_response(part))
                    all_responses.append(response_text)
                
                # Combine all responses into a single string for display
                combined_response = " ".join(all_responses)
            else:
                # Handle single-part questions
                combined_response = asyncio.run(get_response(prompt))

            # Display the combined response
            st.markdown(combined_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": combined_response})
