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
# conda env export > environment.yml

import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PIL import Image
import os

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

# Main function
def main():

    # Ajouter page d'accueil / présentation ?

    # Preloaded PDF path
    preloaded_pdf_path = "./pwc-luxembourg-annual-review-2024.pdf"

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
                st.text_area("Contenu du PDF :", pdf_content, height=400)
            elif uploaded_file:
                st.success("Fichier PDF uploadé avec succès.")
                st.write(f"Nom du fichier : {uploaded_file.name}")
                # Process the uploaded PDF
                # Save uploaded file to disk temporarily to allow caching
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                    pdf_content = load_pdf(temp_path)  # Cached function call
                    st.text_area("Contenu du PDF :", pdf_content, height=400)
            else:
                st.error("Veuillez uploader un fichier PDF avant de continuer.")


# Run the app
if __name__ == "__main__":
    main()



