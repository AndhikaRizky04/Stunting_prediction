import streamlit as st
import joblib
import pandas as pd
import numpy as np

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
# UI STREAMLIT
# =====================================
st.title("Prediksi Risiko Stunting Anak")

st.write("Masukkan data anak:")

input_data = {}

for f in features:
    input_data[f] = st.number_input(f, value=0.0)

if st.button("Prediksi"):

    df = pd.DataFrame([input_data])

    prob = model.predict_proba(df)[:,1][0]
    pred = int(prob >= threshold)

    st.write(f"Probabilitas stunting: {prob:.3f}")

    if pred == 1:
        st.error("Berisiko Stunting")
    else:
        st.success("Tidak Berisiko Stunting")