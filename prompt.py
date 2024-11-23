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
    hf_api_key = st.secrets["huggingface"]["summarize_key"]
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

try:
    openai_api_key = st.secrets["openai"]["rephrase_key"]
except KeyError:
    st.error("OpenAI API key is missing from Streamlit secrets. Please set it in the Secrets management panel.")
    openai_api_key = None

# Initialize the OpenAI client
client = openai.Client(api_key=openai_api_key)

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

# summarize 1 chunk using bart
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

# summarize a chunk using OpenAI's GPT-3.5 API
# not used, possible update needed
def summarize_chunk_gpt_api(chunk):
    """Summarizes a single chunk using OpenAI's GPT-3.5 API."""
    prompt = f"Please summarize the following text:\n\n{chunk}"

    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" depending on what you want to use
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
    )

    summary = response.choices[0].message['content'].strip()
    return summary

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

# Function to rewrite a summary for a specific audience using GPT-3.5 API
# not used, not free
def rewrite_for_audience(text, audience="kids"):
    """Rewrites a text for a specific audience using OpenAI's latest API structure."""

    # Construct the prompt for the model
    prompt = f"Rewrite the following text for a {audience} audience, making it engaging and suitable for their understanding:\n\n{text}"

    response = client.chat.completions.create(
        messages = [
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo",
    )

    rewritten_text = response.choices[0].message['content'].strip()
    return rewritten_text

# Initialize the API client
inference = InferenceApi(repo_id="facebook/bart-large-cnn")

def rewrite_for_audience_hf(text, audience="kids"):
    """Rewrites a text for a specific audience using Hugging Face's BART model.
    Chunks the text, rephrases each chunk for a specific audience, and combines the rephrased results."""
    # Chunk the text into manageable pieces
    chunks = chunk_text(text)

    # Create placeholders for progress updates and progress bar
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Initialize an empty list to store rewritten chunks
    rewritten_chunks = []

    # Process each chunk
    for i, chunk in enumerate(chunks):
         # Update the progress bar and placeholder
        progress_bar.progress((i + 1) / len(chunks))
        progress_placeholder.write(f"Rephrasing chunk {i + 1}/{len(chunks)} for {audience} audience...")

        # Create a prompt based on the target audience
        prompt = f"Rewrite the following text for a {audience} audience, making it engaging and suitable for their understanding:\n\n{chunk}"

        try:
            # Call the Hugging Face API to rewrite the chunk
            response = inference(inputs=prompt)

            # Check and extract the rewritten text
            if isinstance(response, list) and 'summary_text' in response[0]:
                rewritten_chunk = response[0]['summary_text']
                rewritten_chunks.append(rewritten_chunk)
            else:
                print(f"Unexpected response format: {response}")
                rewritten_chunks.append("")  # Append an empty string in case of failure
        except Exception as e:
            print(f"Error rephrasing chunk {i + 1}: {e}")
            rewritten_chunks.append("")  # Handle errors gracefully

    # Combine all rewritten chunks into a single text
    final_rewritten_text = "\n".join(rewritten_chunks)
    return final_rewritten_text


# Main function
def main():

    # Ajouter page d'accueil / présentation ?

    # Preloaded PDF path
    # shorter doc for tests (modify for deployment)
    preloaded_pdf_path = "./truncated_document.pdf"
    # complete doc
    preloaded_pdf_path = "./pwc-luxembourg-annual-review-2024.pdf"

    # Create first form (choix du doc)
    with st.form("document_form"):
        st.markdown("<h3 style='font-size: 1.3em; font-weight: bold;'>1) Please choose the document to summarize</h3>", unsafe_allow_html=True)

        # Option to use preloaded document
        use_preloaded = st.radio(
            "Options :",
            ("Use default document (annual-review)", "Upload a PDF file, preferably in English")
        )

        # File uploader for custom PDF (only appears if user chooses to upload)
        uploaded_file = None
        if use_preloaded == "Upload a PDF file, preferably in English":
            uploaded_file = st.file_uploader("Upload PDF file:", type=["pdf"])

        # Pour quel public ?
        st.markdown("<h3 style='font-size: 1.3em; font-weight: bold;'>2) Select the target audience for rewriting</h3>", unsafe_allow_html=True)
        chosen_audience = st.selectbox("Options", ["kids", "experts", "general public"])
        # add free choice

        # Submit button
        submitted = st.form_submit_button("Submit")

        if submitted:
            if use_preloaded == "Use default document (annual-review)":
                st.success("You chose the preselected document: *annual-review*.")
                # Process the preloaded PDF
                # factoriser !
                pdf_content = load_pdf(preloaded_pdf_path)  # Cached function call
                first_summary = summarize_text(pdf_content)
                st.subheader("Standard Summary")
                st.write(first_summary)

                # now rephrase
                rewritten_summary = rewrite_for_audience_hf(first_summary, chosen_audience)
                st.subheader(f"Rewritten Summary for {chosen_audience.capitalize()}:")
                st.write(rewritten_summary)

            elif uploaded_file:
                st.success("PDF file uploaded.")
                st.write(f"Name : {uploaded_file.name}")
                # Process the uploaded PDF
                # Save uploaded file to disk temporarily to allow caching
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                    # facto!
                    pdf_content = load_pdf(temp_path)  # Cached function call
                    first_summary = summarize_text(pdf_content)
                    st.subheader("Standard Summary")
                    st.write(first_summary)

                    # now rephrase
                    # FACTORISER
                    rewritten_summary = rewrite_for_audience_hf(first_summary, chosen_audience)
                    st.subheader(f"Rewritten Summary for {chosen_audience.capitalize()}:")
                    st.write(rewritten_summary)

            else:
                st.error("Please upload a PDF file, or choose default document.")


# Run the app
if __name__ == "__main__":
    main()



