import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Prediksi Stunting",
    page_icon="🩺",
    layout="centered"
)

# =====================================
# CUSTOM CSS (BIAR KEREN)
# =====================================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3, label {
    color: white !important;
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    background-color: #2563eb;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# CLASS ENSEMBLEMODEL (WAJIB ADA)
# =====================================
class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:,1] >= 0.5).astype(int)

# =====================================
# LOAD MODEL
# =====================================
package = joblib.load("model.pkl")
model = package["model"]
threshold = package["threshold"]
features = package["features"]

# =====================================
# HEADER
# =====================================
st.title("🩺 Prediksi Risiko Stunting Anak")
st.caption("Aplikasi Prediksi Berbasis Machine Learning")

st.markdown("---")

# =====================================
# INPUT CARD
# =====================================
st.subheader("📋 Input Data Anak")

col1, col2 = st.columns(2)

with col1:
    umur_bulan = st.number_input("Umur Bulan", min_value=0.0, value=0.0)
    berat = st.number_input("Berat (kg)", min_value=0.0, value=0.0)
    jk = st.selectbox("Jenis Kelamin", ["L", "P"])

with col2:
    tinggi = st.number_input("Tinggi (cm)", min_value=0.0, value=0.0)
    cara_ukur = st.selectbox("Cara Ukur", ["Berdiri", "Terlentang"])

# encoding
jk_encoded = 1 if jk == "L" else 0
cara_ukur_encoded = 1 if cara_ukur == "Berdiri" else 0

# =====================================
# BUILD INPUT MODEL
# =====================================
input_data = {}

input_data["umur_bulan"] = umur_bulan
input_data["jk_encoded"] = jk_encoded
input_data["berat"] = berat
input_data["tinggi"] = tinggi
input_data["cara_ukur_encoded"] = cara_ukur_encoded

for f in features:
    if f not in input_data:
        input_data[f] = 0

# =====================================
# PREDIKSI BUTTON
# =====================================
if st.button("🔍 Prediksi Sekarang"):

    with st.spinner("Menganalisis data anak..."):
        time.sleep(1.5)  # animasi loading

    df = pd.DataFrame([input_data])

    prob = model.predict_proba(df)[:,1][0]
    pred = int(prob >= threshold)

    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")

    st.progress(float(prob))

    st.write(f"Probabilitas Risiko: **{prob:.2%}**")

    if pred == 1:
        st.error("⚠️ Anak Berisiko Stunting")
    else:
        st.success("✅ Anak Tidak Berisiko Stunting")
