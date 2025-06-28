import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Health Risk Prediction", layout="centered")
st.title("🏥 Health Risk Prediction App")
st.write("Predict your risk of **Hypertension (High BP)** or **Diabetes** using basic health indicators.")

# Load models
ht_model = joblib.load("Hypertension.pkl")
diabetes_model_data = joblib.load("Diabetes.pkl")
diabetes_model = diabetes_model_data["model"]
diabetes_scaler = diabetes_model_data["scaler"]

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
        
        if pred:
            st.warning("⚠️ You may be at risk of Hypertension.")
        else:
            st.success("✅ You are unlikely to have Hypertension.")


# Diabetes Section
elif choice == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    st.markdown("### 🧾 Basic Personal & Lifestyle Info")

    age = st.slider("Age Category", 1, 13, 5, help="Age buckets (e.g., 1=18-24, 13=80+)")  # Assuming bucketed as in BRFSS
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex == "Female" else 1

    high_chol = st.selectbox("High Cholesterol?", ["No", "Yes"])
    high_chol = 1 if high_chol == "Yes" else 0

    chol_check = st.selectbox("Cholesterol Checked in last 5 years?", ["No", "Yes"])
    chol_check = 1 if chol_check == "Yes" else 0

    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)

    st.markdown("### 🚬 Habits & Heart Health")

    smoker = st.selectbox("Do you smoke?", ["No", "Yes"])
    smoker = 1 if smoker == "Yes" else 0

    heart_disease = st.selectbox("History of Heart Disease or Heart Attack?", ["No", "Yes"])
    heart_disease = 1 if heart_disease == "Yes" else 0

    phys_activity = st.selectbox("Physical Activity in last 30 days?", ["No", "Yes"])
    phys_activity = 1 if phys_activity == "Yes" else 0

    fruits = st.selectbox("Do you consume fruits daily?", ["No", "Yes"])
    fruits = 1 if fruits == "Yes" else 0

    veggies = st.selectbox("Do you consume vegetables daily?", ["No", "Yes"])
    veggies = 1 if veggies == "Yes" else 0

    alcohol = st.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"])
    alcohol = 1 if alcohol == "Yes" else 0

    st.markdown("### 🧠🦵 Health Status")

    gen_health = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
    mental_health = st.slider("Poor Mental Health Days (last 30)", 0, 30, 5)
    physical_health = st.slider("Poor Physical Health Days (last 30)", 0, 30, 5)
    diff_walk = st.selectbox("Difficulty Walking?", ["No", "Yes"])
    diff_walk = 1 if diff_walk == "Yes" else 0

    stroke = st.selectbox("History of Stroke?", ["No", "Yes"])
    stroke = 1 if stroke == "Yes" else 0

    high_bp = st.selectbox("High Blood Pressure?", ["No", "Yes"])
    high_bp = 1 if high_bp == "Yes" else 0

    # Final input array
    X = np.array([[age, sex, high_chol, chol_check, bmi, smoker, heart_disease,
                   phys_activity, fruits, veggies, alcohol, gen_health,
                   mental_health, physical_health, diff_walk, stroke,
                   high_bp]])

    # Scale and predict
    X_scaled = diabetes_scaler.transform(X)

    if st.button("Predict Diabetes"):
        pred = diabetes_model.predict(X_scaled)[0]
        
        if pred:
            st.warning("⚠️ You may be at risk of Diabetes.")
        else:
            st.success("✅ You are unlikely to have Diabetes.")