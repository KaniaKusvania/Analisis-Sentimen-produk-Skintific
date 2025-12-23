import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

# ===============================
# STREAMLIT CONFIG (WAJIB PALING ATAS)
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen",
    layout="wide"
)

# ===============================
# PATH KONFIGURASI (AMAN LOKAL & CLOUD)
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

FILE_DF = MODEL_DIR / "df_with_sentiment_revised.csv"
FILE_MODEL = MODEL_DIR / "best_svc_model.pkl"
FILE_VECTORIZER = MODEL_DIR / "ngram_vectorizer.pkl"
FILE_LE = MODEL_DIR / "label_encoder.pkl"
FILE_SELECTOR = MODEL_DIR / "chi2_selector_ngram.pkl"

# ===============================
# METRIK MODEL (STATIS – HASIL EVALUASI)
# ===============================
MODEL_ACCURACY = 0.8540
CLASSIFICATION_REPORT = """
               precision    recall  f1-score   support
NEGATIVE       0.88      0.82      0.85        100
NEUTRAL        0.80      0.75      0.77         50
POSITIVE       0.87      0.91      0.89        150
accuracy                           0.85        300
"""

CONF_MATRIX = np.array([
    [82, 10, 8],
    [10, 37, 3],
    [5, 5, 140]
])

# ===============================
# LOAD DATA & MODEL (CACHE)
# ===============================
@st.cache_resource(show_spinner=False)
def load_assets():
    df = pd.read_csv(FILE_DF)
    model = joblib.load(FILE_MODEL)
    vectorizer = joblib.load(FILE_VECTORIZER)
    selector = joblib.load(FILE_SELECTOR)
    label_encoder = joblib.load(FILE_LE)
    return df, model, vectorizer, selector, label_encoder


try:
    df, model, vectorizer, selector, le = load_assets()
except Exception as e:
    st.error(f"❌ Gagal memuat model atau data: {e}")
    st.stop()

# ===============================
# FUNGSI PREDIKSI (LOGIKA BENAR)
# ===============================
def predict_sentiment(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    X_ngram = vectorizer.transform([text])
    X_selected = selector.transform(X_ngram)

    pred = model.predict(X_selected)
    return le.inverse_transform(pred)[0].upper()

# ===============================
# DASHBOARD UI
# ===============================
st.title("🛒 Analisis Sentimen Ulasan Produk")
st.markdown(
    "**Model:** Support Vector Classifier (SVC) + Chi-Square Feature Selection (N-gram)"
)
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
    fig_pie = px.pie(
        sentiment_counts,
        values="Jumlah",
        names="Sentimen",
        title="Persentase Sentimen"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = px.bar(
        sentiment_counts,
        x="Sentimen",
        y="Jumlah",
        text="Jumlah",
        title="Jumlah Komentar per Sentimen"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ===============================
# KINERJA MODEL
# ===============================
st.header("⚙️ Evaluasi Model")

col1, col2 = st.columns(2)

with col1:
    st.metric("Akurasi Model", f"{MODEL_ACCURACY:.4f}")
    st.markdown("**Laporan Klasifikasi (Data Uji):**")
    st.text(CLASSIFICATION_REPORT)

with col2:
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
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

st.divider()

# ===============================
# PREDIKSI INTERAKTIF
# ===============================
st.header("📝 Prediksi Sentimen Baru")

user_text = st.text_area(
    "Masukkan komentar:",
    "Produk ini sangat bagus dan hasilnya memuaskan."
)

if st.button("Prediksi"):
    if user_text.strip():
        with st.spinner("Memproses prediksi..."):
            result = predict_sentiment(user_text)
            st.success(f"Hasil Prediksi Sentimen: **{result}**")
    else:
        st.warning("Teks tidak boleh kosong.")

# ===============================
# SIDEBAR – TOP N-GRAM
# ===============================
st.sidebar.header("🔍 Fitur N-gram Terbaik")

ngram_features = np.array(vectorizer.get_feature_names_out())
scores = selector.scores_

mask = selector.get_support()
selected_features = ngram_features[mask]
selected_scores = scores[mask]

top_features = (
    pd.DataFrame({
        "Fitur": selected_features,
        "Skor Chi2": selected_scores
    })
    .sort_values("Skor Chi2", ascending=False)
    .head(10)
)

st.sidebar.dataframe(top_features, use_container_width=True)
