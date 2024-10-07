import streamlit as st
from PyPDF2 import PdfReader
import docx
import easyocr
from groq import Groq
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Groq client with the API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# EasyOCR reader for image-to-text extraction
reader = easyocr.Reader(['en'])  # Specify the languages you want EasyOCR to support

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

# Function to extract text from images using EasyOCR
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    result = reader.readtext(image_file, detail=0)  # Extract text without box details
    return ' '.join(result)

# Define the llama_langchain function for querying LLaMA
def llama_langchain(input_text, document_text):
    # Combine document text and input text into a formatted question
    user_message = f"Documents: {document_text}\nQuestions: {input_text}"

    # Send the request to the LLaMA model via Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
        model="llama3-8b-8192",  # Use the desired LLaMA model
    )
    
    # Return the generated response
    return chat_completion.choices[0].message.content if chat_completion else "No response received."

# Page title and description
st.title("Document Processing and Querying")
st.subheader("Upload documents (PDF, DOCX, or images) and ask questions based on the content.")

# File uploader
uploaded_files = st.file_uploader("Upload PDF, DOCX, or image files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

# Initialize a variable to store combined text
combined_text = ""

# Process the uploaded files
if uploaded_files:
    st.write("Files uploaded successfully. Processing...")
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            combined_text += extract_text_from_pdf(uploaded_file) + "\n\n"
        elif uploaded_file.name.endswith(".docx"):
            combined_text += extract_text_from_docx(uploaded_file) + "\n\n"
        elif uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
            combined_text += extract_text_from_image(uploaded_file) + "\n\n"
    
    st.success("Documents processed successfully.")
else:
    st.info("Please upload at least one document to continue.")

# Create a form for the query input
with st.form("query_form"):
    query = st.text_input("Ask a question based on the document content")
    submit_query = st.form_submit_button("Submit")

# Handle query submission and response display
if submit_query and query and combined_text:
    result = llama_langchain(query, combined_text)
    
    # Display result in a ChatGPT-like interface
    st.write("### Response:")
    st.write(result)
elif submit_query and not combined_text:
    st.warning("Please upload documents before submitting a query.")
