import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Health Risk Prediction", layout="centered")
st.title("🏥 Health Risk Prediction App")
st.write("Predict your risk of **Hypertension (High BP)** or **Diabetes** using basic health indicators.")

# Load models
ht_model = joblib.load("Hypertension.pkl")
diabetes_model = joblib.load("Diabetes.pkl")

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict(X, w, b):
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):
        z_wb = np.dot(w, X[i]) + b
        f_wb = sigmoid(z_wb)
        p[i] = f_wb >= 0.5
    return p

# Sidebar
choice = st.sidebar.radio("Choose Prediction Type", ["Hypertension", "Diabetes"])

if choice == "Hypertension":
    st.header("🩺 Hypertension (High Blood Pressure) Prediction")

    st.markdown("### 🧾 Personal & Health Information")
    age = st.slider("Age (in years)", 1, 120, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex == "Female" else 1

    st.markdown("### ❤️ Heart-Related Information")
    cp = st.selectbox(
        "Type of Chest Pain:",
        [
            "0 - No chest pain",
            "1 - Pressure/tightness (Typical Angina)",
            "2 - Mild discomfort (Atypical Angina)",
            "3 - Chest pain not related to heart (Non-anginal)"
        ])
    cp = int(cp.split(" - ")[0])

    trestbps = st.number_input(
        "Resting Blood Pressure (in mm Hg)",
        help="Your blood pressure while resting (normal ~120 mm Hg)",
        min_value=80, max_value=200, value=120)

    chol = st.number_input(
        "Serum Cholesterol (in mg/dL)",
        help="Total cholesterol in blood (normal < 200 mg/dL)",
        min_value=100, max_value=600, value=200)

    fbs = st.selectbox(
        "Is fasting blood sugar > 120 mg/dL?",
        ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0

    restecg = st.selectbox(
        "Resting ECG result",
        [
            "0 - Normal",
            "1 - Abnormal (e.g. ST-T wave changes)"
        ])
    restecg = int(restecg.split(" - ")[0])

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        help="The highest heart rate during exercise (normal: 100–170)",
        min_value=60, max_value=250, value=150)

    exang = st.selectbox(
        "Did you experience chest pain during exercise?",
        ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.slider(
        "Oldpeak (ST depression after exercise)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        help="Depression in ST segment (ECG) after exercise. 1.0 is average.")

    slope = st.selectbox(
        "Slope of ST segment during exercise",
        [
            "0 - Upward slope (normal)",
            "1 - Flat (possible issue)",
            "2 - Downward slope (serious risk)"
        ])
    slope = int(slope.split(" - ")[0])

    ca = st.selectbox(
        "Number of Major Blood Vessels Colored by X-ray",
        options=[0, 1, 2, 3],
        help="0 = normal, more = higher risk")

    thal = st.selectbox(
        "Thalassemia Test Result",
        [
            "3 - Normal",
            "6 - Fixed Defect (old damage)",
            "7 - Reversible Defect (blood flow issue)"
        ])
    thal = int(thal.split(" - ")[0])

    X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                   exang, oldpeak, slope, ca, thal]])
    X_scaled = ht_model["scaler"].transform(X)

    if st.button("Predict Hypertension"):
        pred = predict(X_scaled, ht_model["w"], ht_model["b"])[0]
        st.warning("⚠️ You may be at risk of Hypertension.") if pred else st.success("✅ You are unlikely to have Hypertension.")

# Diabetes Section
elif choice == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    st.markdown("### 🧾 Basic Health Information")
    pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1)
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        help="Blood sugar level after fasting (normal: 70–130)",
        min_value=0, max_value=200, value=120)

    bp = st.number_input(
        "Blood Pressure (mm Hg)",
        help="Normal ~80, higher values may be risky",
        min_value=0, max_value=150, value=70)

    skin = st.number_input(
        "Skin Thickness (mm)",
        help="Measured at triceps (normal: 20–35 mm)",
        min_value=0, max_value=100, value=20)

    insulin = st.number_input(
        "Insulin Level (mu U/mL)",
        help="Normal range is 16–166",
        min_value=0, max_value=900, value=80)

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        help="Weight-to-height ratio. 18.5–24.9 is healthy.",
        min_value=0.0, max_value=70.0, value=25.0)

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        help="Higher value = higher genetic risk",
        min_value=0.0, max_value=3.0, value=0.5)

    age = st.slider("Age (in years)", 1, 120, 30)

    X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    X_scaled = diabetes_model["scaler"].transform(X)

    if st.button("Predict Diabetes"):
        pred = diabetes_model["model"].predict[0]
        st.warning("⚠️ You may be at risk of Diabetes.") if pred else st.success("✅ You are unlikely to have Diabetes.")
