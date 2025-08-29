import streamlit as st
import requests
import os

# Set the title and icon for the Streamlit app.
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# Use st.secrets to get the backend URL.
# If you are running locally, use a .streamlit/secrets.toml file.
# If you are deploying, set an environment variable.
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

# Set up the Streamlit interface.
st.title("FAQ Chatbot")
st.caption("🚀 A RAG-based chatbot powered by LLM FAISS")

# Initialize chat history in the session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input.
if prompt := st.chat_input("Ask a question about the project..."):
    # Add user message to chat history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user's message.
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare data for the backend request.
    data = {"user_prompt": prompt}

    try:
        # Send the user's prompt to the backend API.
        with st.spinner('Thinking...'):
            response = requests.post(f"{backend_url}/generate", json=data)
            
            # Raise an error for bad status codes.
            response.raise_for_status()

            # Get the response text.
            llm_response = response.json().get("response", "An error occurred.")
            
            # Display the chatbot's response.
            with st.chat_message("assistant"):
                st.markdown(llm_response)
            
            # Add the chatbot's response to the chat history.
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
    
    except requests.exceptions.RequestException as e:
        error_message = f"Failed to connect to the backend. Please check the backend URL and service status. Error: {e}"
        with st.chat_message("assistant"):
            st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        with st.chat_message("assistant"):
            st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
