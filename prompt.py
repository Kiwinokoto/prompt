# About : Appli (prototype) permettant de synthétiser facilement, pour différents publics,
# un document (.pdf) long et comportant des données non-textuelles,
# en respectant la sécurité et la confidentialité des données.

# limites V1 : pdf with embedded text only (easy to handle text docs later, just skip conversion / OCR steps),
# takes a certain lvl of encryption into account (no password yet),
# discard graphs ? Pas forcément pertinent dans l'optique d'un résumé,
# les infos / analyses importantes sont pbblement reprises ds le texte. (?)
# sécurité / confidentialité à ajouter de manière incrémentale, l'objectif est d'abord de
# réaliser vite un prototype opérationnel (mvp).
# pour les mm raisons, le multilingue sera pris en compte + tard

# modifier adresse ?
# https://summarize-pdf.streamlit.app/

# env
# conda create -n env_prompt python pip requests numpy pandas pytest gdown streamlit PyPDF2 pycryptodome pdf2image
# openai tiktoken pytesseract tesseract pillow transformers langchain poppler
# au final not sure i need openai and tiktoken here
# best solution for price, speed and confidentiality would pbbly be local model, but
# my hardware is limited, and streamlit has a limited cache capacity -> free huggingface api better for now
# conda env export > environment.yml


# librairies
import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PIL import Image
import os
import openai
from typing import List
import re
from transformers import pipeline
import requests
import time
from huggingface_hub import InferenceApi

st.set_page_config(layout='wide')


# out of main (cache)

# Define a function to load the annual review
@st.cache_data
def load_pdf(filepath):
    """Loads a PDF from a file and extracts its text, handling encryption if necessary."""
    try:
        reader = PdfReader(filepath)

        # Check if the PDF is encrypted
        if reader.is_encrypted:
            # Attempt to decrypt (empty password is common for encryption without protection)
            try:
                reader.decrypt("")  # Try with an empty password
            except Exception as e:
                return f"Error: PDF is encrypted and requires a password. {e}"

        # Extract text from the PDF
        pdf_content = ""
        for page in reader.pages:
            pdf_content += page.extract_text()
        return pdf_content
    except Exception as e:
        return f"Error loading PDF: {e}"

# hf api token, name :
# Summarization-API (read)
# key : see .toml file in .streamlit folder (gitignored for security)
# Try fetching the API key
try:
    hf_api_key = st.secrets["huggingface"]["api_key"]
except KeyError:
    st.error("Hugging Face API key is missing from Streamlit secrets. Please set it in the Secrets management panel.")
    hf_api_key = None

if hf_api_key:
    # API_URL = "https://api-inference.huggingface.co/models/facebook/mbart-large-50"
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    def summarize(text):
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            return response.json()[0]['summary_text']
        else:
            return f"Error: {response.status_code}, {response.text}"

# cut long text in small, coherent chunks
def chunk_text(text, max_chars=4000):
    """Splits text into smaller chunks."""
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += paragraph + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# summarize 1 chunk
def summarize_chunk_hf_api(chunk, max_retries=3, retry_delay=10):
    """
    Summarizes a chunk of text using Hugging Face API with retry logic.

    :param chunk: The chunk of text to summarize
    :param max_retries: Maximum number of retries if the request fails
    :param retry_delay: Delay between retries in seconds
    :return: The summary text
    """
    # Create inference API client
    inference = InferenceApi(repo_id="facebook/bart-large-cnn")

    # Try sending request with retry logic
    for attempt in range(max_retries):
        try:
            # Send text to summarize
            summary = inference(inputs=chunk)
            return summary[0]['summary_text']

        except Exception as e:
            # Handle errors (e.g., model not available or network issues)
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                # If all attempts fail, raise the error
                print("Max retries reached. Could not complete the request.")
                raise e

# chunk + summarize all text
def summarize_text(text):
    """Chunks the text and summarizes each chunk, then combines the summaries."""
    # Chunk the text
    chunks = chunk_text(text)
    st.write(f"Total chunks: {len(chunks)}")

    # Create an empty placeholder for the current chunk progress
    progress_placeholder = st.empty()

    # Create a progress bar
    progress_bar = st.progress(0)

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        # Update the progress bar
        progress_bar.progress((i + 1) / len(chunks))

        # Update the text placeholder with current progress
        progress_placeholder.write(f"Summarizing chunk {i+1}/{len(chunks)}...")

        try:
            summary = summarize_chunk_hf_api(chunk)
            summaries.append(summary)
        except Exception as e:
            print("Error during summarization:", e)

        # Optionally, clear the placeholder when done
        progress_placeholder.empty()

    # Combine the summaries
    final_summary = "\n".join(summaries)
    return final_summary



# Main function
def main():

    # Ajouter page d'accueil / présentation ?

    # Preloaded PDF path
    # shorter doc for tests (modify for deployment)
    preloaded_pdf_path = "./truncated_document.pdf"
    # complete doc
    # preloaded_pdf_path = "./pwc-luxembourg-annual-review-2024.pdf"

    # Create first form (choix du doc)
    with st.form("document_form"):
        st.markdown("<h3 style='font-size: 1.3em; font-weight: bold;'>1) Veuillez choisir le document à synthétiser</h3>", unsafe_allow_html=True)

        # Option to use preloaded document
        use_preloaded = st.radio(
            "Options :",
            ("Utiliser le document présélectionné (annual-review)", "Uploader un fichier PDF")
        )

        # File uploader for custom PDF (only appears if user chooses to upload)
        uploaded_file = None
        if use_preloaded == "Uploader un fichier PDF":
            uploaded_file = st.file_uploader("Uploader un fichier PDF :", type=["pdf"])

        # Submit button
        submitted = st.form_submit_button("Valider")

        if submitted:
            if use_preloaded == "Utiliser le document présélectionné (annual-review)":
                st.success("Vous avez choisi d'utiliser le document présélectionné : *annual-review*.")
                # Process the preloaded PDF
                pdf_content = load_pdf(preloaded_pdf_path)  # Cached function call
                final_summary = summarize_text(pdf_content)
                st.subheader("Final Summary")
                st.write(final_summary)
            elif uploaded_file:
                st.success("Fichier PDF uploadé avec succès.")
                st.write(f"Nom du fichier : {uploaded_file.name}")
                # Process the uploaded PDF
                # Save uploaded file to disk temporarily to allow caching
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                    pdf_content = load_pdf(temp_path)  # Cached function call
                    final_summary = summarize_text(pdf_content)
                    st.subheader("Final Summary")
                    st.write(first_summary)
            else:
                st.error("Veuillez uploader un fichier PDF avant de continuer.")


# Run the app
if __name__ == "__main__":
    main()



