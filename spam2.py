import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="SMS Spam Detector", page_icon="📨")
st.title("📨 SMS Spam Detection")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("SMSSpamCollection.txt", sep="\t", header=None, names=["label", "message"])
        return df
    except FileNotFoundError:
        st.error("❌ File data 'SMSSpamCollection.txt' tidak ditemukan.")
        return pd.DataFrame(columns=["label", "message"])

@st.cache_resource
def train_and_save_model(df):
    X = df['message']
    y = df['label'].map({'ham': 0, 'spam': 1})

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    # Save model & vectorizer
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return model, vectorizer

# Load model or train if not exists
if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
else:
    df = load_data()
    if not df.empty:
        model, vectorizer = train_and_save_model(df)
    else:
        st.stop()

# User input
st.subheader("🔍 Cek apakah SMS termasuk spam:")
user_input = st.text_area("Masukkan pesan SMS Anda di bawah:")

if st.button("Deteksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Tolong masukkan teks terlebih dahulu.")
    else:
        input_vec = vectorizer.transform([user_input])
        pred = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)[0]

        label = "🚫 Spam" if pred == 1 else "✅ Bukan Spam"
        st.success(f"Hasil Deteksi: {label}")
        st.progress(proba[1] if pred == 1 else proba[0])
        st.write(f"🔢 Probabilitas Spam: **{proba[1]*100:.2f}%**")
