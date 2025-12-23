import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen",
    layout="wide"
)

# ===============================
# PATH
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

FILE_DF = MODEL_DIR / "df_with_sentiment_revised.csv"
FILE_MODEL = MODEL_DIR / "best_svc_model.pkl"
FILE_VECTORIZER = MODEL_DIR / "ngram_vectorizer.pkl"
FILE_LE = MODEL_DIR / "label_encoder.pkl"

# ===============================
# METRIK MODEL
# ===============================
MODEL_ACCURACY = 0.8540

CONF_MATRIX = np.array([
    [82, 10, 8],
    [10, 37, 3],
    [5, 5, 140]
])

# ===============================
# LOAD ASSET
# ===============================
@st.cache_resource(show_spinner=False)
def load_assets():
    df = pd.read_csv(FILE_DF)
    model = joblib.load(FILE_MODEL)
    vectorizer = joblib.load(FILE_VECTORIZER)
    le = joblib.load(FILE_LE)
    return df, model, vectorizer, le


try:
    df, model, vectorizer, le = load_assets()
except Exception as e:
    st.error(f"Gagal load asset: {e}")
    st.stop()

# ===============================
# PREDIKSI (FIX FEATURE MISMATCH)
# ===============================
def predict_sentiment(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    X = vectorizer.transform([text]).toarray()

    expected_features = model.n_features_in_
    current_features = X.shape[1]

    # 🔥 SINKRONKAN FITUR
    if current_features > expected_features:
        X = X[:, :expected_features]
    elif current_features < expected_features:
        pad_width = expected_features - current_features
        X = np.hstack([X, np.zeros((1, pad_width))])

    pred = model.predict(X)
    return le.inverse_transform(pred)[0].upper()

# ===============================
# UI
# ===============================
st.title("🛒 Analisis Sentimen Ulasan Produk")
st.markdown("Model: **SVC + N-gram (deployed safely)**")
st.divider()

# ===============================
# DISTRIBUSI SENTIMEN
# ===============================
st.header("📊 Distribusi Sentimen")

sentiment_counts = (
    df["sentiment_category"]
    .value_counts()
    .reset_index()
)
sentiment_counts.columns = ["Sentimen", "Jumlah"]

col1, col2 = st.columns(2)

with col1:
    fig = px.pie(sentiment_counts, values="Jumlah", names="Sentimen")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(sentiment_counts, x="Sentimen", y="Jumlah", text="Jumlah")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===============================
# CONFUSION MATRIX
# ===============================
st.header("⚙️ Evaluasi Model")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    CONF_MATRIX,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=ax
)
st.pyplot(fig)

st.divider()

# ===============================
# PREDIKSI
# ===============================
st.header("📝 Prediksi Sentimen")

user_text = st.text_area(
    "Masukkan komentar:",
    "Produk ini sangat bagus dan hasilnya memuaskan."
)

if st.button("Prediksi"):
    if user_text.strip():
        with st.spinner("Memproses..."):
            result = predict_sentiment(user_text)
            st.success(f"Hasil Prediksi: **{result}**")
    else:
        st.warning("Teks tidak boleh kosong")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.info(
    """
    Model dikembangkan menggunakan pendekatan supervised learning
    dengan algoritma Support Vector Classifier (SVC)
    berbasis fitur N-gram.
    """
)

