from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"

# Set up the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Please respond to the user query"),
        ("user", "Questions:{question}")
    ]
)

# Set up the chatbot model
llm = Ollama(model="llama3.1")
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

# Streamlit UI
st.title("FinGPT - AI Chatbot")

# Initialize chat history in session state if not already present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Create a form for user input
with st.form(key='chat_form'):
    input_text = st.text_input("You:", placeholder="Ask a question...")
    submit_button = st.form_submit_button(label='Send')

# Process the input and update the conversation history when form is submitted
if submit_button and input_text:
    # Generate AI response
    response = chain.invoke({'question': input_text})
    
    # Append the user query and AI response to the chat history
    st.session_state['chat_history'].append(("You", input_text))
    st.session_state['chat_history'].append(("AI Assistant", response))

# Display the conversation history
st.subheader("Conversation:")
for i, (sender, message) in enumerate(st.session_state['chat_history']):
    if sender == "You":
        st.markdown(f"**{sender}:** {message}")
    else:
        st.markdown(f"**{sender}:** {message}")
