import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator
from collections import Counter
import nltk
from nltk.corpus import stopwords

def CreartextConb(df, cols_to_combine):
    def combine_columns(row):
        values = [str(row[col]) for col in cols_to_combine if pd.notna(row[col]) and str(row[col]).lower() != 'nan']
        return ' '.join(values)
    df['Combined_Col'] = df.apply(combine_columns, axis=1)
    return df[['Hole_ID', 'Depth_From', 'Depth_To', 'Combined_Col']].copy()

# Función de traducción
def traducir_al_ingles(texto):
    try:
        return GoogleTranslator(source='auto', target='en').translate(str(texto))
    except Exception as e:
        return f"ERROR: {e}"

# 🔹 Función para limpiar y tokenizar texto
def tokenize(text):
    if pd.isna(text):
        return []
    return re.findall(r'\b\w+\b', text.lower()) 

# 🔹 Función para limpiar, tokenizar y filtrar palabras vacías
def clean_text(text):
    if pd.isna(text):
        return []
    tokens = re.findall(r'\b\w+\b', text.lower())  # Tokeniza en minúsculas
    return [word for word in tokens if word not in stop_words]
# Descargar stopwords en español si es la primera vez
nltk.download("stopwords") 
stop_words = set(stopwords.words("spanish"))  # Lista de palabras vacías
