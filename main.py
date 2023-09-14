import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.model_selection import GridSearchCV, KFold
from keras.layers import Dense, Embedding, LSTM, Dropout

from scikeras.wrappers import KerasClassifier


from sklearn.datasets import make_classification
from wordcloud import WordCloud
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # untuk stemming
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)

from streamlit_option_menu import option_menu
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="Analisis Sentimen - Ifan Dwi Cahya", page_icon="☺", layout="wide"
)

##CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    #MainMenu{
        visibility:hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================================================================
# Sidebar
# ========================================================================
with st.sidebar:
    selected = option_menu(
        menu_title="Sentimen",
        options=[
            "Dashboard",
            "Classification With Crawling",
            "Classification Input Text",
        ],
        icons=[
            "grid-fill",
            "file-earmark-bar-graph-fill",
            "file-check-fill",
        ],
        menu_icon="emoji-laughing-fill",
        default_index=1,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#fafafa",
                "margin-top": "0",
                "font-family": "poppins",
            },
            "menu-title": {
                "margin-bottom": "10px",
                "font-weight": "bold",
                "font-size": "20px",
            },
            "menu-icon": {"color": "#003bfd", "font-weight": "bold"},
            "nav": {"margin-top": "10px"},
            "icon": {"color": "#003bfd", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#eee",
                "border-radius": "5px",
            },
            "nav-link-selected": {"background-color": "#003bfd21", "color": "#003bfd"},
        },
    )

# Import Data
data = pd.read_excel(
    "D:/SKRIPSI/fix/Streamlit/model/Fix data ulasan mypertamina fitur pendaftaran.xlsx"
)

# Import LSTM Model

with open("model/stem.pkl", "rb") as f:
    stem_load = pickle.load(f)

new_model = tf.keras.models.load_model("model/lstm model.h5")

# ========================================================================
# Dashboard
# ========================================================================
if selected == "Dashboard":
    st.subheader("Analisis Sentimen")
    st.markdown("by **Ifan Dwi Cahya**", unsafe_allow_html=True)
    st.subheader(
        "Analisis Sentimen pada Google Play Store Menggunakan Metode LSTM (Studi Kasus : Aplikasi MyPertamina)"
    )
    st.write(
        "Sebuah aplikasi yang mampu mengklasifikasikan sentimen suatu ulasan dengan menggunakan metode LSTM. Data latih yang digunakan dalam sistem ini diambil dari ulasan Aplikasi MyPertamina di Goggle Play Store"
    )
    st.markdown("Akurasi dari sistem ini adalah **91%**", unsafe_allow_html=True)


# ========================================================================
# Classification new data
# ========================================================================

from google_play_scraper import Sort, reviews


if selected == "Classification With Crawling":
    st.subheader(f"{selected}")
    with st.form("my_form"):
        st.write("Klasifikasi Data Baru")
        number = st.number_input("masukkan angka data terbaru", min_value=1, step=1)
        result, continuation_token = reviews(
            "com.dafturn.mypertamina",
            lang="id",  # defaults to 'en'
            country="id",  # defaults to 'us'
            sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT you can use Sort.NEWEST to get newst reviews
            count=number,  # defaults to 100
            filter_score_with=None,  # defaults to None(means all score) Use 1 or 2 or 3 or 4 or 5 to select certain score
        )
        submitted = st.form_submit_button("Proses")
        if submitted:
            df = pd.DataFrame(np.array(result), columns=["review"])
            df = df.join(pd.DataFrame(df.pop("review").tolist()))
            df = df[["at", "content"]]
            teks = df["content"]
            teks["casefolding"] = df["content"].str.lower()

        # ===========================Cleaning===========================

            def remove_punct(text):
                # Remove Karakter ASCII, angka, punctuation
                text = text.encode("ascii", "replace").decode("ascii")
                text = re.sub("x(\d+[a-zA-Z]+|[a-zA-Z]+\d+|\d+)", "", text)
                text = re.sub("[0-9]+", "", text)
                text = re.sub(r"[\W\s_]", " ", text)

                # Remove spasi diawal dan akhir, url, hastag, tagar, kata akhiran berlebihan, baris baru, tab
                text = re.sub("^\s+|\s+$", "", text)
                text = re.sub(
                r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", "", text
                )  # url
                text = re.sub("@[A-Za-z0-9]+", "", text)
                text = re.sub("#[A-Za-z0-9]+", "", text)
                # text = re.sub(r'([a-z])\1+', r'\1', text)
                text = (
                text.replace("\\t", " ")
                .replace("\\n", " ")
                .replace("\\u", " ")
                .replace("\\", " ")
            )
                return text

            teks["cleaning"] = teks["casefolding"].apply(lambda x: remove_punct(x))

        # ===========================Noramlization===========================

            normalized_word = pd.read_excel("kamus\kamus perbaikan.xlsx")

            normalized_word_dict = {}

            for index, row in normalized_word.iterrows():
                if row[0] not in normalized_word_dict:
                    normalized_word_dict[row[0]] = row[1]

            def normalized_term(document):
                return " ".join(
                [
                    normalized_word_dict[term] if term in normalized_word_dict else term
                    for term in document.split()
                ]
            )

            teks["normalization"] = teks["cleaning"].apply(lambda x: normalized_term(x))

        # ===========================tokenisasi===========================
            def tokenization(text):
                text = nltk.tokenize.word_tokenize(text)
                return text

            teks["tokenizing"] = teks["normalization"].apply(lambda x: tokenization(x))

        # ===========================Filtering===========================
            list_stopwords = set(stopwords.words("indonesian"))

            with open("kamus/stopword.txt", "r") as file:
                for line in file:
                    line = line.strip()
                    list_stopwords.add(line)

            hapus = ["tidak", "baik", "kurang", "biasa", "saja", "cukup"]
            for i in hapus:
                if i in list_stopwords:
                    list_stopwords.remove(i)

            def stopwords_removal(words):
                return [word for word in words if word not in list_stopwords]

            teks["filtering"] = teks["tokenizing"].apply(lambda x: stopwords_removal(x))

        # ===========================Stemming===========================
        # create stemmer
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()

            def stemming(text):
                text = [stemmer.stem(word) for word in text]
                return text

            teks["stemming"] = teks["filtering"].apply(lambda x: stemming(x))
        # teks[['stemming']]

            corpus = teks["stemming"].values
        # ===========================Predict===========================
            max_len = 120
            trunc_type = "pre"
            padding_type = "pre"
            oov_tok = "<OOV>"
            vocab_size = 1833

            from keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            
            tokenizer = Tokenizer(num_words=vocab_size, char_level=False, oov_token=oov_tok)
            tokenizer.fit_on_texts(stem_load.values)
            text_seq = tokenizer.texts_to_sequences(corpus)
            text_pad = pad_sequences(text_seq,maxlen=max_len,padding=padding_type,truncating=trunc_type,dtype="int32",)
            predicted_sentiment = new_model.predict(text_pad).round()
            df['prediksi'] = predicted_sentiment
            def conversionlabel(label):
                if label == 1:
                    return "Positif"
                else:
                    return "Negatif"
            df['prediksi'] = df["prediksi"].apply(conversionlabel)
            df = df[["at", "content", 'prediksi']]
            st.dataframe(df)

# ========================================================================
# Classification with input text
# ========================================================================

elif selected == "Classification Input Text":
    st.subheader(f"{selected}")

    with st.form("my_form"):
        st.write("Testing Data")
        ulasan = st.text_area("Masukkan Data Testing")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Proses")
    if submitted:
        teks = pd.DataFrame([ulasan], columns=["ulasan"], index=None)

        # ===========================CaseFolding===========================

        teks["casefolding"] = teks["ulasan"].str.lower()

        # ===========================Cleaning===========================

        def remove_punct(text):
            # Remove Karakter ASCII, angka, punctuation
            text = text.encode("ascii", "replace").decode("ascii")
            text = re.sub("x(\d+[a-zA-Z]+|[a-zA-Z]+\d+|\d+)", "", text)
            text = re.sub("[0-9]+", "", text)
            text = re.sub(r"[\W\s_]", " ", text)

            # Remove spasi diawal dan akhir, url, hastag, tagar, kata akhiran berlebihan, baris baru, tab
            text = re.sub("^\s+|\s+$", "", text)
            text = re.sub(
                r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", "", text
            )  # url
            text = re.sub("@[A-Za-z0-9]+", "", text)
            text = re.sub("#[A-Za-z0-9]+", "", text)
            # text = re.sub(r'([a-z])\1+', r'\1', text)
            text = (
                text.replace("\\t", " ")
                .replace("\\n", " ")
                .replace("\\u", " ")
                .replace("\\", " ")
            )
            return text

        teks["cleaning"] = teks["casefolding"].apply(lambda x: remove_punct(x))

        # ===========================Noramlization===========================

        normalized_word = pd.read_excel("kamus\kamus perbaikan.xlsx")

        normalized_word_dict = {}

        for index, row in normalized_word.iterrows():
            if row[0] not in normalized_word_dict:
                normalized_word_dict[row[0]] = row[1]

        def normalized_term(document):
            return " ".join(
                [
                    normalized_word_dict[term] if term in normalized_word_dict else term
                    for term in document.split()
                ]
            )

        teks["normalization"] = teks["cleaning"].apply(lambda x: normalized_term(x))

        # ===========================tokenisasi===========================
        def tokenization(text):
            text = nltk.tokenize.word_tokenize(text)
            return text

        teks["tokenizing"] = teks["normalization"].apply(lambda x: tokenization(x))

        # ===========================Filtering===========================
        list_stopwords = set(stopwords.words("indonesian"))

        with open("kamus/stopword.txt", "r") as file:
            for line in file:
                line = line.strip()
                list_stopwords.add(line)

        hapus = ["tidak", "baik", "kurang", "biasa", "saja", "cukup"]
        for i in hapus:
            if i in list_stopwords:
                list_stopwords.remove(i)

        def stopwords_removal(words):
            return [word for word in words if word not in list_stopwords]

        teks["filtering"] = teks["tokenizing"].apply(lambda x: stopwords_removal(x))

        # ===========================Stemming===========================
        # create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        def stemming(text):
            text = [stemmer.stem(word) for word in text]
            return text

        teks["stemming"] = teks["filtering"].apply(lambda x: stemming(x))
        # teks[['stemming']]

        corpus = teks["stemming"].values
        # ===========================Predict===========================
        max_len = 120
        trunc_type = "pre"
        padding_type = "pre"
        oov_tok = "<OOV>"
        vocab_size = 1833

        from keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        
        tokenizer = Tokenizer(num_words=vocab_size, char_level=False, oov_token=oov_tok)
        tokenizer.fit_on_texts(stem_load.values)
        text_seq = tokenizer.texts_to_sequences(corpus)
        
        text_pad = pad_sequences(
            text_seq,
            maxlen=max_len,
            padding=padding_type,
            truncating=trunc_type,
            dtype="int32",
        )
        predicted_sentiment = new_model.predict(text_pad).round()

        if predicted_sentiment == 1.0:
            st.success("😊😊😊 - Positif")
        else:
            st.error("😊😊😊 - negatif")
