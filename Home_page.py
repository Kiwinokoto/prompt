# https://rosetta-stones.streamlit.app/
# modif

# conda create -n env_prompt python pip requests numpy pandas pytest gdown streamlit PyPDF2 pdf2image pytesseract tesseract pillow transformers langchain poppler 
# conda env export > environment.yml

import subprocess

import streamlit as st
from audio_recorder_streamlit import audio_recorder
st.set_page_config(layout='wide')

# paths, folders/files
import os, sys, random, re
from io import BytesIO

# math, dataframes
import numpy as np
import pandas as pd

# Visualisation
import matplotlib as matplot
import matplotlib.pyplot as plt
import plotly.express as px

# audio
import librosa
import librosa.display
import soundfile as sf

# model
import whisper
from whisper import load_model

# import pickle


# Main function
def main():

    st.write("Depuis cette page, vous pouvez explorer les données utilisées pour tester le modèle, ")
    st.write("ou le tester vous-même directement !   ")

# Run the app
main()



