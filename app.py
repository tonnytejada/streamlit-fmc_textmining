import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from scrpt import scrtext
import nltk
from nltk.corpus import stopwords

st.title("Text Mining Application")
st.write('Created by Alex Delgado - Octubre 2025 for Free Use')
st.write('This app allows you to upload a CSV file, combine selected text columns, translate text to English, and generate word clouds based on the text data.')
st.write('You can also create binary columns indicating the presence of selected words.')
st.write('Obligation of columns in data is: "Hole_ID, Depth_From, Depth_To, Comments" or similar.')
#st.write('Based on Deep Translator, NLTK, WordCloud, Streamlit, Pandas, Matplotlib, Seaborn, Altair, Plotly')

uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    st.subheader("Table", divider="gray")
    #text = uploaded_file.read().decode("utf-8")
    text = pd.read_csv(uploaded_file, encoding='latin-1') 
    # Display raw text
    st.write(text.head())
    st.subheader("Select columns to combine", divider="gray")
    columns = st.multiselect("Select", options=text.columns.tolist())
    
    if  'scrtext' in globals():
        # Example usage of the CreartextConb function
        data = scrtext.CreartextConb(text, columns)
        st.write(data)#.head())
        df = scrtext.traducir_al_ingles(data['Combined_Col'])
        # Further text processing and visualization code would go here
        #st.write(df)#.head())
        # 🔹 Extraer todas las palabras del texto
        st.subheader("Select column to tokenize", divider="gray")
        #name_token = st.multiselect("Select column tokenize", options=data.columns.tolist(), default='Alt_Combined')
        cols = data.columns.tolist()
        #default_idx = cols.index('Alt_Combined') if 'Alt_Combined' in cols else 0
        name_token = st.selectbox("Select", options=cols)#, index=default_idx)
        #st.write(name_token)

        all_words = [word for desc in data[name_token] for word in scrtext.tokenize(desc)]
        word_counts0 = Counter(all_words)  # Contar frecuencia de palabras
        st.subheader("Nube de Palabras", divider="gray")
        if word_counts0:
            wc = WordCloud(width=800, height=500,background_color="white", colormap="copper").generate_from_frequencies(word_counts0)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        # Descargar stopwords en español si es la primera vez
        nltk.download("stopwords")
        stop_words = set(stopwords.words("spanish"))  # Lista de palabras vacías

        # 🔹 Extraer todas las palabras relevantes del texto
        #all_words = [word for desc in data[name_token] for word in scrtext.clean_text(desc)]
        #word_counts = Counter(all_words)  # Contar frecuencia de palabras
        
        # 🔹 Extraer todas las palabras relevantes del texto (sin tokens que contengan números)
        all_words = []
        for desc in data[name_token].fillna("").astype(str):
            try:
                toks = scrtext.clean_text(desc)  # asumimos lista de tokens
            except Exception:
                toks = scrtext.tokenize(desc) if hasattr(scrtext, "tokenize") else re.findall(r"\w+", desc.lower())

            for w in toks:
                if not re.search(r"\d", w):        # descartar cualquier token que contenga dígitos
                    all_words.append(w.lower())

        word_counts = Counter(all_words)  # Contar frecuencia de palabras



        st.subheader("Nube de Palabras (filtradas las mas comunes)", divider="gray")
        
        if word_counts:
            wc = WordCloud(width=800, height=500,background_color="white", colormap="copper").generate_from_frequencies(word_counts)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    TOP_N = 10000  # número de palabras más frecuentes a mostrar
    freq_items = word_counts.most_common(TOP_N)

    default_sel = [w for w, _ in freq_items[:2]]  # por defecto las 6 más frecuentes
    #default_sel = ['cpy']
    selected_words = st.multiselect("Selecciona palabras a marcar (usa acentos exactos si aplica):", options=[w for w, _ in freq_items], default=default_sel)

    if selected_words:
        # Mostrar tabla con los conteos de las seleccionadas
        sel_counts = [{"word": w, "count": word_counts.get(w, 0)} for w in selected_words]
        st.table(pd.DataFrame(sel_counts).sort_values("count", ascending=False))

        # Modo de creación de columnas
        mode = st.radio("Crear columnas:", ("Columna combinada (has_any_selected)", "Columnas individuales por palabra"))

        if st.button("Crear columna(s) binarias"):
            # función para obtener tokens por fila (devuelve set en minúsculas)
            def tokens_from_text(txt):
                txt = "" if pd.isna(txt) else str(txt)
                # intentar clean_text -> tokenize -> fallback regex split
                try:
                    toks = scrtext.clean_text(txt)
                    if not toks:
                        toks = scrtext.tokenize(txt)
                except Exception:
                    try:
                        toks = scrtext.tokenize(txt)
                    except Exception:
                        toks = re.findall(r"\w+", txt.lower())
                # devolver set de tokens en lowercase
                return set([t.lower() for t in toks if isinstance(t, str)])

            # Construir token_sets (serie de sets) - computación por fila
            token_sets = data[name_token].fillna("").astype(str).apply(tokens_from_text)

            sel_lower = [w.lower() for w in selected_words]

            if mode.startswith("Columna combinada"):
                # crea columna booleana 1/0 si aparece cualquiera de las palabras seleccionadas
                data["has_any_selected"] = token_sets.apply(lambda s: int(bool(set(sel_lower) & s)))
                st.success("Columna 'has_any_selected' creada (1 = aparece alguna palabra seleccionada).")
            else:
                # crear columnas individuales: has_<palabra_normalizada>
                created_cols = []
                for w in sel_lower:
                    safe_name = re.sub(r'\W+', '_', w)
                    col_name = f"has_{safe_name}"
                    data[col_name] = token_sets.apply(lambda s, _w=w: int(_w in s))
                    created_cols.append(col_name)
                st.success("Columnas individuales creadas: " + ", ".join(created_cols))

            # Mostrar preview y opción de descarga
            st.subheader("Preview del DataFrame (con nuevas columnas)", divider="gray")
            st.write(data.head())

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV con columnas binarias", data=csv, file_name="data_with_binary_columns.csv", mime="text/csv")
        
        else:
            st.info("Selecciona al menos una palabra para poder crear las columnas binarias.")

    else:
        st.info("Selecciona al menos una palabra para poder crear las columnas binarias.")

        st.error("El módulo 'scrtext' no está disponible. Asegúrate de que 'scrpt' está en el mismo directorio y exporta 'scrtext'.")
