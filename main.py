import streamlit as st

st.set_page_config(page_title="ATTRA", layout="centered")

# --- Title ---
st.title("The At-Home Triage and Treatment Recommendation App (ATTRA)")

# --- Brief Explanation ---
st.write("""
ATTRA is designed to help individuals make informed decisions about their health 
from the comfort of home. Using evidence-based logic and structured clinical inputs, 
ATTRA provides three core tools:

- **Diagnosis Tool** – Helps identify likely conditions based on symptoms  
- **Triage Tool** – Assesses severity and recommends level of care  
- **Treatment Tool** – Suggests safe, at-home treatment options when appropriate  
""")

# --- Learn More Button ---
if st.button("Learn More"):
    st.switch_page("pages/learn_more.py")

st.markdown("---")


# ---------------------------
# SECTION: Diagnosis Tool
# ---------------------------
st.subheader("Diagnosis Tool")
st.write("""
The Diagnosis Tool analyzes your symptoms and medical history to estimate the most 
likely conditions. It uses structured clinical features and pattern recognition to 
provide a ranked list of possibilities.
""")

if st.button("Get Started – Diagnosis Tool"):
    st.switch_page("pages/diagnosis_main.py")

st.markdown("---")


# ---------------------------
# SECTION: Triage Tool
# ---------------------------
st.subheader("Triage Tool")
st.write("""
The Triage Tool evaluates the urgency of your symptoms and provides guidance on 
whether you will likely be admitted or discharged from an emergency department.
This tool may assist but should NOT be solely relied on to determine whether you
should seek emergency care. 
""")

if st.button("Get Started – Triage Tool"):
    st.switch_page("pages/triage_main.py")

st.markdown("---")


# ---------------------------
# SECTION: Treatment Tool
# ---------------------------
st.subheader("Treatment Tool")
st.write("""
The Treatment Tool offers evidence-based recommendations for managing common 
conditions at home, including over-the-counter options, supportive care, and 
monitoring guidance.
""")

if st.button("Get Started – Treatment Tool"):
    st.switch_page("pages/treatment_tool.py")
