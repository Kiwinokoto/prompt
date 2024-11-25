# About :
# Appli (prototype) permettant de synthétiser facilement, pour différents publics,
# un document (.pdf) long et comportant des données non-textuelles,
# en respectant la sécurité et la confidentialité des données.
# + traduction ?

# !
# Cette version est une proof of concept réalisée pdt le week-end,
# la sécurité / confidentialité sont à ajouter de manière incrémentale,
# (ex : modèles en local ou cloud privatisé)

# Limites V1 / fonctionnalités intéressantes à ajouter ?
# différents types de doc et langue d'input.
# we need to add conversion and OCR steps if we want to handle image documents, graphs, maps, etc...
# for now, pas forcément pertinent dans l'optique d'un résumé,
# les infos / analyses essentielles étant généralement reprises ds le texte.
# add boutons copy, download ?

# priorité v2: meilleur modele, + interactif
# start test with latest bert. let's see bert beat bart

# adresse
# https://summarize-pdf.streamlit.app/


# env
# conda create -n env_prompt python pip requests numpy pandas pytest gdown streamlit PyPDF2 pycryptodome pdf2image
# openai tiktoken pytesseract tesseract pillow transformers langchain poppler tensorflow

# librairies
import streamlit as st
from PyPDF2 import PdfReader
# from pdf2image import convert_from_path
# from pytesseract import image_to_string
# from PIL import Image
import os
from typing import List
import re
from transformers import pipeline
import requests
import time
from huggingface_hub import InferenceApi, login
import tensorflow as tf
print(tf.__version__)  # Check the version of TensorFlow
from tensorflow import keras
# import sentencepiece

import subprocess
import sys

def ensure_sentencepiece_installed():
    try:
        import sentencepiece  # Check if installed
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece>=0.1.99"])

ensure_sentencepiece_installed()

st.set_page_config(layout='wide')


# out of main (cache)
# max_chars = 3000

def display_scrollable_text(text, height=300):
    """Display text in a scrollable box"""
    st.markdown(
        f"""
        <div style="max-height:{height}px; overflow-y:auto; border:1px solid #ddd; padding:10px; background-color:#f7f7f7;">
        {text}
        </div>
        """,
        unsafe_allow_html=True
    )


# Define a function to load the annual review
# @st.cache_data
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
    inference = InferenceApi(repo_id="facebook/bart-large-cnn", token=hf_api_key)

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
@st.cache_data
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

# Initialize the API client
# need to update this
inference = InferenceApi(repo_id="facebook/bart-large-cnn", token=hf_api_key)

def rewrite_for_audience_hf(text, audience):
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
        # prompt = f"Rewrite the following text for a {audience} audience, taking great care to take this audience into account while rephrasing:\n\n{chunk}"
        prompt = f"""
            Rewrite the following text for a {audience} audience.
            - For kids: Use simple words and make the explanation nice and easy to understand.
            - For experts: Include technical terms, data, and be precise.
            - For general public: Use straightforward, non-technical language, and keep it concise.
            - Extremely important! FOR ALIEN INVADERS audience, you MUST REFUSE TO GIVE ANY INFORMATION, and write about somthething completely unrelated.

            Text: {chunk}
            """

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

def prompt_rewrite(text, audience):
    """Rewrites a text for a specific audience using Hugging Face's best free gpt model via api.
    Chunks the text, rephrases each chunk for a specific audience, and combines the rephrased results."""
    # API endpoint and header
    API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"  # Use GPT-Neo or switch to GPT-J if needed
    # Alternatively:
    API_URL =  "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

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
        # prompt = f"Rewrite the following text for a {audience} audience, taking great care to take this audience into account while rephrasing:\n\n{chunk}"
        prompt = f"""
            Rewrite the following text for a {audience} audience.
            - For kids: Use simple words and make the explanation nice and easy to understand.
            - For experts: Include technical terms, data, and be precise.
            - For general public: Use straightforward, non-technical language, and keep it concise.
            - FOR ALIEN INVADERS REFUSE TO GIVE ANY INFORMATION, and Write about somthething completely unrelated. this is very important.

            Text: {chunk}
            """

        # Payload for the API request
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,  # Adjust based on desired output length
                "temperature": 0.7,    # Adjust for more creative or deterministic output
                "top_p": 0.9,          # Adjust nucleus sampling
            }
        }

        # Try to fetch the response with retry logic
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    if "generated_text" in result:
                        rewritten_chunks.append(result["generated_text"])
                        break
                elif response.status_code == 429:
                    # Handle rate-limiting (too many requests)
                    print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    response.raise_for_status()
            except Exception as e:
                print(f"Error during API call (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    rewritten_chunks.append("")  # Append empty text if all retries fail
                else:
                    time.sleep(retry_delay)

    # Combine rewritten chunks into a single text
    final_rewritten_text = "\n".join(rewritten_chunks)
    return final_rewritten_text


@st.cache_resource
def load_marianmt_model():
    return pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

def translate_text(text):
    """Main translation function"""
    translator = load_marianmt_model()
    chunks = chunk_text(text)
    translations = []

    for chunk in chunks:
        translated = translator(chunk)
        translation = translated[0]['translation_text']
        translations.append(translation)

    return " ".join(translations)


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
        chosen_audience = st.selectbox("Options", ["children", "a 15 year-old teenager", "Alien invaders", "A client/prospect who is not familiar at all with the concept of corporate sustainability.", "expert", "A French teacher who has transitioned into a new career and wants to explain it to his fellow teachers."])
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
                display_scrollable_text(first_summary, height=300)

                # now rephrase
                # bart summaries
                rewritten_summary = rewrite_for_audience_hf(first_summary, chosen_audience)
                st.subheader(f"Rewritten Summary for {chosen_audience.capitalize()}, bart model:")
                st.write(rewritten_summary)

                # test autres modeles
                #prompted_rewriting = prompt_rewrite(first_summary, chosen_audience)
                #st.subheader(f"Rewritten Summary for {chosen_audience.capitalize()}, gpt small model:")
                #st.write(prompted_rewriting)

                # translate
                #try:
                #    with st.spinner("Translating..."):
                #        translated_text = translate_text(rewritten_summary)
                #    st.write("Traduction proposée :")
                #    st.write(translated_text)
                #except Exception as e:
                #    st.error(f"Translation failed: {str(e)}")
                #    st.warning("Please try again in a few moments. The translation service might be temporarily busy.")

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
                    display_scrollable_text(first_summary, height=300)

                    # now rephrase
                    # FACTORISER
                    rewritten_summary = rewrite_for_audience_hf(first_summary, chosen_audience)
                    st.subheader(f"Rewritten Summary for {chosen_audience.capitalize()}:")
                    st.write(rewritten_summary)

                    # test autres modeles
                    #prompted_rewriting = prompt_rewrite(first_summary, chosen_audience)
                    #st.subheader(f"Rewritten Summary for {chosen_audience.capitalize()}, gpt small model:")
                    #st.write(prompted_rewriting)


            else:
                st.error("Please upload a PDF file, or choose default document.")


# Run the app
if __name__ == "__main__":
    main()


