import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Initialize the Groq client with the API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Streamlit UI
st.title("FinGPT - AI Chatbot")

# Initialize chat history in session state if not already present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to query the LLaMA model using Groq
def llama_langchain(input_text):
    # Construct the user message for LLaMA
    user_message = f"Questions: {input_text}"
    
    # Send the request to the LLaMA model via Groq
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_message}],
        model="llama3-8b-8192"
    )
    
    # Return the response content
    return chat_completion.choices[0].message.content if chat_completion else "No response received."

# Create a form for user input
with st.form(key='chat_form'):
    input_text = st.text_input("You:", placeholder="Ask a question...")
    submit_button = st.form_submit_button(label='Send')

# Process the input and update the conversation history when form is submitted
if submit_button and input_text:
    # Generate AI response
    response = llama_langchain(input_text)
    
    # Append the user query and AI response to the chat history
    st.session_state['chat_history'].append(("You", input_text))
    st.session_state['chat_history'].append(("AI Assistant", response))

# Display the conversation history
st.subheader("Conversation:")
for sender, message in st.session_state['chat_history']:
    st.markdown(f"**{sender}:** {message}")
