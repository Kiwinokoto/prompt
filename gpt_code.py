import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PIL import Image
import os

# Title and description
st.title("PDF OCR & Chunk Visualizer")
st.write("Upload a PDF, process it page by page, and visualize OCR results.")

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file to disk
    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Read PDF and extract pages
    st.write("Extracting pages from the PDF...")
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    st.write(f"Total pages in PDF: {total_pages}")

    # Page OCR and visualization
    st.write("Processing pages...")
    page_images = convert_from_path(pdf_path, dpi=200)  # Convert PDF to images

    for page_number, image in enumerate(page_images, start=1):
        st.subheader(f"Page {page_number}")

        # Perform OCR on the page
        text = image_to_string(image)
        st.image(image, caption=f"Page {page_number}", use_column_width=True)
        st.text_area(f"OCR Result - Page {page_number}", text, height=200)

        # Add a download button for text (optional)
        st.download_button(
            label=f"Download OCR Text - Page {page_number}",
            data=text,
            file_name=f"page_{page_number}_ocr.txt",
            mime="text/plain"
        )

    # Cleanup temporary file
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
