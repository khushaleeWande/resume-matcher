import pickle
import PyPDF2
import os
import re
import requests
import fitz

def download_pdf(url):
    """Download PDF from a given URL and save it locally."""
    response = requests.get(url)
    pdf_path = 'downloaded_resume.pdf'
    with open(pdf_path, 'wb') as pdf_file:
        pdf_file.write(response.content)
    return pdf_path

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ''
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + ' '
    return text

def extract_text_from_pdf2(pdf_path) :
    doc = fitz.open(pdf_path)  
    text = ""
    for page in doc:
        text = text + str(page.get_text())
    tx = " ".join(text.split('\n'))

    # print(15 * "---")
    # print(tx)
    return tx

def cleanResume(text):
    """Clean the extracted resume text."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Load the model and vectorizer
with open('random_forest_resume_classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    loaded_tfidf = pickle.load(file)