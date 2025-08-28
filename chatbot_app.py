# frontend/app.py - Streamlit Frontend
import streamlit as st
import requests
import os

# --- Configuration ---
# Use an environment variable set in the Render dashboard
# Fallback to localhost for local development
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# The endpoint path is constant
API_ENDPOINT = "/api/query"

# --- Page Setup ---
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
st.title("FAQ Chatbot")
st.caption("powered by FastAPI, Streamlit, and an LLM")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            st.info(f"Source: {message['source']}")

# Handle user input
if prompt := st.chat_input("Ask a question about our FAQs..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display an empty placeholder for the bot's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send the user query to the FastAPI backend
                full_backend_url = f"{BACKEND_URL}{API_ENDPOINT}"
                response = requests.post(
                    full_backend_url,
                    json={"query": prompt},
                    timeout=120 # Set a generous timeout for the LLM
                )
                response.raise_for_status() # Raise an exception for bad status codes

                # Parse the JSON response
                result = response.json()
                answer = result.get("answer", "I'm sorry, I could not find an answer.")
                source = result.get("source", "Unknown")

                # Check if the answer is empty or null and provide a fallback
                if not answer or answer.lower() == 'nan':
                    answer_to_display = "I'm sorry, an issue occurred and I couldn't provide an answer."
                else:
                    answer_to_display = answer

                # Display the full answer
                st.markdown(answer_to_display)

                # Display the source information
                st.info(f"Source: {source}")

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_to_display,
                    "source": source
                })

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while connecting to the backend: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
