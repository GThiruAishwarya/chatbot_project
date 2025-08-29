import streamlit as st
import requests
import json
import base64

# This is where you need to change the URL.
# Replace this placeholder URL with the actual public URL of your backend service on Render.
BACKEND_URL = "https://chatbot-project-knx9.onrender.com"

# --- Update the title here ---
st.title("FAQ Chatbot")
st.write("Ask me anything from my knowledge base!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare data for the backend API call
    payload = json.dumps({"user_prompt": prompt})
    headers = {"Content-Type": "application/json"}

    try:
        # Make the API call to the backend.
        # This will now use the correct public URL.
        with st.spinner("Thinking..."):
            response = requests.post(f"{BACKEND_URL}/generate", data=payload, headers=headers)
            response.raise_for_status() # Raise an error for bad status codes
            
            # The backend returns a JSON object.
            backend_response = response.json()
            
            # Extract the response from the JSON payload
            chatbot_response = backend_response.get("response", "Sorry, I couldn't get a response from the server.")
            
    except requests.exceptions.RequestException as e:
        chatbot_response = f"An error occurred while connecting to the backend: {e}"
    except Exception as e:
        chatbot_response = f"An unexpected error occurred: {e}"

    # Display chatbot response in chat message container
    with st.chat_message("assistant"):
        st.markdown(chatbot_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})