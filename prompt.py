# About : Appli (prototype) permettant de synthétiser facilement, pour différents publics,
# un document (.pdf) long et comportant des données non-textuelles,
# en respectant la sécurité et la confidentialité des données.

# limites V1 : pdf only (easy to handle text docs later, just skip conversion / OCR steps),
# discard graphs ? Pas forcément pertinent dans l'optique d'un résumé,
# les infos / analyses importantes sont pbblement reprises ds le texte. (?)
# sécurité / confidentialité à ajouter de manière incrémentale, l'objectif est d'abord de
# réaliser vite un prototype opérationnel (mvp)

# modifier adresse
# https://rosetta-stones.streamlit.app/

# env
# conda create -n env_prompt python pip requests numpy pandas pytest gdown streamlit PyPDF2 pdf2image
# pytesseract tesseract pillow transformers langchain poppler
# conda env export > environment.yml

import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PIL import Image
import os

st.set_page_config(layout='wide')

# Main function
def main():

    # Ajouter page d'accueil / présentation ?

    # Create a form
    with st.form("document_form"):
        st.markdown("<h3 style='font-size: 1.5em; font-weight: bold;'>1) Veuillez choisir le document à synthétiser</h3>", unsafe_allow_html=True)

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
                # Example: load_preloaded_pdf()
            elif uploaded_file:
                st.success("Fichier PDF uploadé avec succès.")
                st.write(f"Nom du fichier : {uploaded_file.name}")
                # Process the uploaded PDF
                # Example: process_uploaded_pdf(uploaded_file)
            else:
                st.error("Veuillez uploader un fichier PDF avant de continuer.")


# Run the app
if __name__ == "__main__":
    main()



