import streamlit as st
from PyPDF2 import PdfReader
import docx
import easyocr
import os
from PIL import Image
from langchain_community.llms import Ollama  # For LLaMA model querying
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup the environment for API keys if necessary
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"

# Set up the LLaMA model
llm = Ollama(model="llama3.1")

# Initialize the EasyOCR reader for image-to-text extraction
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

# Page title and description
st.title("Document Processing and Querying")
#st.subheader("Upload multiple documents (PDF, DOCX, or images) and ask questions based on the content.")

# File uploader (allow multiple files)
uploaded_files = st.file_uploader("Upload PDF, DOCX, or image files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

# Initialize a variable to store combined text
combined_text = ""

# Check if any file is uploaded
if uploaded_files:
    st.write("Files uploaded successfully. Processing...")
    
    for uploaded_file in uploaded_files:
        # Determine file type and extract text accordingly
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

# Display response from LLaMA model if query is submitted
if submit_query and query and combined_text:
    # Construct a system message for LLaMA
    prompt = f"The following content has been uploaded: {combined_text}\n\nUser query: {query}"

    # Send query to LLaMA
    result = llm(prompt)
    
    # Display result in a ChatGPT-like interface
    st.write("### Response:")
    st.write(result)
elif submit_query and not combined_text:
    st.warning("Please upload documents before submitting a query.")
