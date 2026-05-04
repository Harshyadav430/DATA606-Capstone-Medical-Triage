import streamlit as st
import pandas as pd
import numpy as np
from numba import cuda
import torch

st.set_page_config(
    page_title="At-Home Triage System",
    page_icon="🏥",
    layout="wide"
)

print('cuda av: ', torch.cuda.is_available())

@st.cache_data
@cuda.jit
def load_data():
    return pd.read_csv("disease_symptom_medication_dataset_compressed.csv", low_memory=False)

df = load_data()

excluded_cols = ["disease", "Medications", "Emergency", "Doctor_Consult"]
symptom_cols = [col for col in df.columns if col not in excluded_cols]

@st.cache_resource
@cuda.jit
def train_model(dataframe, features):
    from sklearn.ensemble import RandomForestClassifier

    X = dataframe[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = dataframe["disease"].astype(str)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, list(X.columns)

model, feature_names = train_model(df, symptom_cols)

@cuda.jit
def predict_top_diseases(selected_symptoms):
    input_vector = [0] * len(feature_names)
    feature_index = {feature: idx for idx, feature in enumerate(feature_names)}

    for symptom in selected_symptoms:
        if symptom in feature_index:
            input_vector[feature_index[symptom]] = 1

    input_array = np.array(input_vector).reshape(1, -1)

    probabilities = model.predict_proba(input_array)[0]
    classes = model.classes_

    top_idx = np.argsort(probabilities)[::-1][:3]
    return [(classes[i], float(probabilities[i])) for i in top_idx]

@cuda.jit
def get_disease_details(disease_name):
    match = df[df["disease"].astype(str) == str(disease_name)]

    if match.empty:
        return "Consult a physician", "Unknown"

    row = match.iloc[0]
    medications = row["Medications"] if "Medications" in row and pd.notna(row["Medications"]) else "Consult a physician"
    emergency = row["Emergency"] if "Emergency" in row and pd.notna(row["Emergency"]) else "Unknown"

    return medications, emergency

@cuda.jit
def clean_medications(med_value):
    if isinstance(med_value, str):
        meds = [m.strip() for m in med_value.split(",") if m.strip()]
        return meds if meds else ["Consult a physician"]
    return ["Consult a physician"]

@cuda.jit
def get_demo_doctor_info(predicted_disease, emergency_status):
    if isinstance(emergency_status, str) and emergency_status.strip().lower() == "yes":
        return {
            "name": "Dr. Michael Carter",
            "specialty": "Emergency Medicine Specialist",
            "phone": "(410) 555-0182",
            "email": "emergency.consult@healthbridge-demo.com",
            "address": "HealthBridge Urgent Care, 2450 Medical Plaza, Baltimore, MD",
            "hours": "24/7 Emergency Consultation",
            "mode": "Phone and emergency walk-in",
        }

    disease_text = str(predicted_disease).lower()

    if any(word in disease_text for word in ["skin", "fungal", "allergy", "rash"]):
        return {
            "name": "Dr. Sophia Bennett",
            "specialty": "Dermatology",
            "phone": "(410) 555-0147",
            "email": "dermatology@healthbridge-demo.com",
            "address": "HealthBridge Specialty Clinic, 1200 Lakeview Avenue, Baltimore, MD",
            "hours": "Mon-Fri, 9:00 AM - 5:00 PM",
            "mode": "Video and in-person consultation",
        }

    if any(word in disease_text for word in ["anxiety", "depression", "panic", "psychotic"]):
        return {
            "name": "Dr. Emily Foster",
            "specialty": "Internal Medicine and Behavioral Health",
            "phone": "(410) 555-0164",
            "email": "behavioral.health@healthbridge-demo.com",
            "address": "HealthBridge Medical Center, 88 Wellness Drive, Catonsville, MD",
            "hours": "Mon-Sat, 10:00 AM - 6:00 PM",
            "mode": "Video consultation preferred",
        }

    if any(word in disease_text for word in ["heart", "cardiac", "chest", "hypertension"]):
        return {
            "name": "Dr. Daniel Brooks",
            "specialty": "Cardiology",
            "phone": "(410) 555-0119",
            "email": "cardiology@healthbridge-demo.com",
            "address": "HealthBridge Heart Institute, 3100 Central Avenue, Baltimore, MD",
            "hours": "Mon-Fri, 8:30 AM - 4:30 PM",
            "mode": "In-person and telehealth consultation",
        }

    return {
        "name": "Dr. Olivia Reed",
        "specialty": "General Physician",
        "phone": "(410) 555-0128",
        "email": "general.consult@healthbridge-demo.com",
        "address": "HealthBridge Primary Care, 5600 Cedar Lane, Catonsville, MD",
        "hours": "Mon-Fri, 9:00 AM - 6:00 PM",
        "mode": "Video and in-person consultation",
    }


# STYLING 
st.markdown("""
<style>
    .stApp {
        background-color: #0b1220;
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2430 0%, #232838 100%);
        border-right: 1px solid #30384a;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }

    .section-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 1.4rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
        margin-bottom: 1rem;
    }

    .result-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
        margin-top: 1rem;
    }

    .doctor-card {
        background: #0f172a;
        border: 1px solid #263244;
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
        margin-top: 1rem;
    }

    .contact-box {
        background: #111827;
        border: 1px solid #2a3648;
        border-radius: 14px;
        padding: 1rem;
        margin-top: 0.75rem;
    }

    .result-label {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }

    .result-value {
        color: #f8fafc;
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        line-height: 1.3;
    }

    .doctor-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .doctor-specialty {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-bottom: 0.6rem;
    }

    .profile-line {
        color: #e5e7eb;
        font-size: 1rem;
        margin-bottom: 0.45rem;
    }

    .profile-line strong {
        color: #f8fafc;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.72rem 1rem;
        font-weight: 600;
        font-size: 1rem;
        transition: 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #2a3648 !important;
        border-radius: 12px !important;
    }

    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border: 1px solid #2a3648 !important;
        border-radius: 12px !important;
    }

    .stProgress > div > div {
        background-color: #1f2937 !important;
        border-radius: 999px;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #3b82f6) !important;
        border-radius: 999px;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }

    label, .stMarkdown, .stText, p, li {
        color: #e5e7eb !important;
    }

    .stCaption {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="app-title">At-Home Triage and Diagnosis System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">AI-assisted disease screening and medication recommendation based on selected symptoms</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Patient Information")

name = st.sidebar.text_input("Full Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
blood_group = st.sidebar.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
medical_history = st.sidebar.text_area(
    "Medical History",
    placeholder="Enter relevant past conditions, allergies, or current medications"
)

# ---------------- MAIN LAYOUT ----------------
left_col, right_col = st.columns([1.7, 1.2], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Patient Intake")
    st.write("Enter patient details and select the symptoms currently observed.")

    st.markdown("#### Selected Profile")
    st.markdown(f'<div class="profile-line"><strong>Name:</strong> {name if name else "Not provided"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-line"><strong>Age:</strong> {age}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-line"><strong>Gender:</strong> {gender}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-line"><strong>Blood Group:</strong> {blood_group}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="profile-line"><strong>Medical History:</strong> {medical_history if medical_history else "Not provided"}</div>',
        unsafe_allow_html=True
    )

    st.markdown("#### Symptom Selection")
    selected_symptoms = st.multiselect(
        "Choose one or more symptoms",
        symptom_cols,
        placeholder="Search and select symptoms"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Prediction Panel")
    predict_clicked = st.button("Run Diagnosis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        if len(selected_symptoms) == 0:
            st.warning("Please select at least one symptom before running diagnosis.")
        else:
            top_predictions = predict_top_diseases(selected_symptoms)
            primary_disease, primary_confidence = top_predictions[0]
            medications, emergency = get_disease_details(primary_disease)
            meds_list = clean_medications(medications)
            doctor_info = get_demo_doctor_info(primary_disease, emergency)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Most Likely Condition</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-value">{primary_disease}</div>', unsafe_allow_html=True)

            st.markdown('<div class="result-label">Confidence</div>', unsafe_allow_html=True)
            st.progress(float(primary_confidence))
            st.write(f"{primary_confidence * 100:.2f}%")

            st.markdown('<div class="result-label">Recommended Medications</div>', unsafe_allow_html=True)
            for med in meds_list:
                st.write(f"- {med}")

            st.markdown('<div class="result-label">Care Recommendation</div>', unsafe_allow_html=True)
            if isinstance(emergency, str) and emergency.strip().lower() == "yes":
                st.error("Immediate medical attention is recommended.")
            else:
                st.success("Routine consultation or home care may be appropriate based on current symptoms.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Top 3 Predicted Conditions</div>', unsafe_allow_html=True)
            for disease_name, score in top_predictions:
                st.write(f"**{disease_name}** — {score * 100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="doctor-card">', unsafe_allow_html=True)
            st.subheader("Consult a Doctor")
            st.markdown('<div class="small-note">Demo contact information for prototype purposes only.</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="doctor-name">{doctor_info["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="doctor-specialty">{doctor_info["specialty"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="contact-box">', unsafe_allow_html=True)
            st.write(f"**Phone:** {doctor_info['phone']}")
            st.write(f"**Email:** {doctor_info['email']}")
            st.write(f"**Address:** {doctor_info['address']}")
            st.write(f"**Availability:** {doctor_info['hours']}")
            st.write(f"**Consultation Mode:** {doctor_info['mode']}")
            st.markdown('</div>', unsafe_allow_html=True)

            consult_choice = st.radio(
                "Would you like to consult this doctor?",
                ["No", "Yes"],
                horizontal=True
            )

            if consult_choice == "Yes":
                st.info("Consultation request submitted successfully. A representative will contact the patient shortly.")
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("This tool provides preliminary guidance and does not replace professional medical consultation.")
