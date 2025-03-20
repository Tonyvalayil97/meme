import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# Ensure "pdfs/" directory exists
pdfs_directory = "pdfs/"
if not os.path.exists(pdfs_directory):
    os.makedirs(pdfs_directory)

# Initialize Embeddings and Model
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
model = OllamaLLM(model="deepseek-r1:1.5b")

# Define Extraction Prompt
extraction_prompt = """
Extract the following details from the invoice:

- **Company Name**
- **Invoice Number**
- **Weight** (mention units if available)
- **Volume** (mention units if available)
- **Final Amount** (include currency if mentioned)

If any value is missing, return "N/A".

Invoice Text:
{invoice_text}

Extracted Details:
"""

# Function to Save Uploaded PDFs
def upload_pdf(file):
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

# Function to Extract Text from PDF
def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    
    # Combine all pages into a single text block
    invoice_text = "\n".join([doc.page_content for doc in chunked_docs])
    return invoice_text

# Function to Extract Invoice Details using AI
def extract_invoice_details(invoice_text):
    prompt = extraction_prompt.format(invoice_text=invoice_text)
    response = model.invoke(prompt)
    
    extracted_info = {"Company Name": "N/A", "Invoice Number": "N/A", "Weight": "N/A", "Volume": "N/A", "Final Amount": "N/A"}

    # Parse AI response
    for line in response.split("\n"):
        if "Company Name" in line:
            extracted_info["Company Name"] = line.split(":")[1].strip()
        elif "Invoice Number" in line:
            extracted_info["Invoice Number"] = line.split(":")[1].strip()
        elif "Weight" in line:
            extracted_info["Weight"] = line.split(":")[1].strip()
        elif "Volume" in line:
            extracted_info["Volume"] = line.split(":")[1].strip()
        elif "Final Amount" in line:
            extracted_info["Final Amount"] = line.split(":")[1].strip()

    return extracted_info

# Function to Process Uploaded PDFs and Generate Excel Report
def process_invoices(uploaded_files):
    invoice_data = []
    
    for uploaded_file in uploaded_files:
        file_path = upload_pdf(uploaded_file)  # Save file
        invoice_text = extract_text_from_pdf(file_path)  # Extract text
        invoice_details = extract_invoice_details(invoice_text)  # Extract key details
        
        # Add file name for reference
        invoice_details["File Name"] = uploaded_file.name
        
        invoice_data.append(invoice_details)

    # Convert to DataFrame and Save to Excel
    df = pd.DataFrame(invoice_data)
    excel_file = "extracted_invoices.xlsx"
    df.to_excel(excel_file, index=False)
    
    return excel_file, df  # Return the file path and DataFrame
