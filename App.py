import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL & FITUR
# =========================
model = joblib.load("model.pkl")   # ganti sesuai nama model Anda
fitur_training = joblib.load("fitur_training.pkl")

# =========================
# JUDUL APP
# =========================
st.title("Prediksi Stunting Anak")

# =========================
# INPUT USER
# =========================
st.subheader("Input Data Anak")

umur_bulan = st.number_input("Umur Bulan", min_value=0.0)
berat = st.number_input("Berat (kg)", min_value=0.0)
tinggi = st.number_input("Tinggi (cm)", min_value=0.0)

jk = st.selectbox("Jenis Kelamin", ["L", "P"])
cara_ukur = st.selectbox("Cara Ukur", ["Berdiri", "Terlentang"])

# =========================
# FEATURE ENGINEERING
# =========================
def feature_engineering(df):

    df = df.copy()

    # ===== fitur turunan (sesuaikan training Anda) =====
    df["age_sq"] = df["umur_bulan"] ** 2
    df["age_log"] = np.log1p(df["umur_bulan"])

    df["rasio_bb_umur"] = df["berat"] / (df["umur_bulan"] + 1)
    df["rasio_bb_tb"] = df["berat"] / (df["tinggi"] + 1)

    df["bmi_proxy"] = df["berat"] / ((df["tinggi"]/100) ** 2)

    # encoding kategori
    df["jk_encoded"] = df["jk"].map({"L": 1, "P": 0})
    df["cara_ukur_encoded"] = df["cara_ukur"].map({
        "Berdiri": 1,
        "Terlentang": 0
    })

    # hapus kolom string
    df.drop(["jk", "cara_ukur"], axis=1, inplace=True)

    return df

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi Sekarang"):

    # buat dataframe dari input
    df = pd.DataFrame([{
        "umur_bulan": umur_bulan,
        "berat": berat,
        "tinggi": tinggi,
        "jk": jk,
        "cara_ukur": cara_ukur
    }])

    # feature engineering
    df = feature_engineering(df)

    # SAMAKAN FITUR DENGAN TRAINING
    df = df.reindex(columns=fitur_training, fill_value=0)

    # prediksi
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    # output hasil
    st.subheader("Hasil Prediksi")

    if pred == 1:
        st.error(f"⚠️ Anak terindikasi STUNTING (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Anak NORMAL (Probabilitas: {prob:.2f})")
