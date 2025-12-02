import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# LOAD MODEL + SCALER + FEATURES
# ===============================
model = pickle.load(open("loan_approval_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_cols = pickle.load(open("feature_columns.pkl", "rb"))


# ===============================
# PREPROCESSING (SAMA PERSIS DGN TRAINING)
# ===============================
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])

    # Dependents
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    # Feature Engineering
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_Log"] = np.log(df["LoanAmount"])
    df["Total_Income_Log"] = np.log(df["Total_Income"])

    # Dropped Columns (harus sama seperti training)
    drop_cols = [
        "Loan_ID", "LoanAmount", "ApplicantIncome",
        "CoapplicantIncome", "Total_Income"
    ]
    df = df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        errors="ignore"
    )

    # One Hot Encoding
    df = pd.get_dummies(df, drop_first=True)

    # Reindex agar match kolom waktu training
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Scaling
    df_scaled = scaler.transform(df)

    return df_scaled


# ===============================
# PREDIKSI
# ===============================
def predict_loan(user_input):
    processed = preprocess_input(user_input)

    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1]

    return ("Approved" if pred == 1 else "Rejected", round(float(prob), 3))


# ===============================
# STREAMLIT UI
# ===============================
st.title("💳 Loan Approval Prediction App")
st.write("Masukkan data nasabah untuk memprediksi apakah pinjaman akan **disetujui** atau **ditolak**.")

st.divider()

# ----- Form Input -----
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=150)
loan_term = st.number_input("Loan Term (months)", min_value=12, value=360)
credit_history = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1, 0])

# Button Prediksi
if st.button("🔮 Prediksi"):
    user_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Property_Area": property_area,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history
    }

    result, prob = predict_loan(user_input)

    st.subheader("Hasil Prediksi")
    st.write(f"**Status** : {result}")
    st.write(f"**Probabilitas Disetujui** : {prob}")

    if result == "Approved":
        st.success("🎉 Pinjaman kemungkinan **disetujui**.")
    else:
        st.error("❌ Pinjaman kemungkinan **ditolak**.")


st.divider()
st.caption("Model: Logistic Regression | Deployment by Streamlit")
