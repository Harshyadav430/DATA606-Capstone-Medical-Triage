# DATA606 - Data Science Capstone
## At-Home Triage and Treatment Recommendation Application (ATTRA)

### Purpose: 
Globally, medical facilities are inundated by the numbers of patients seeking emergency care.  Inefficient triaging causes non-emergency patient care to consume precious resources from those facing life-threatening medical situations.  To help make more efficient use of scarce resources, we offer an at-home triage and treatment recommendation application to offer initial triaging and disease identification, to recommend medications, and advise admission to a hospital, if necessary.  A literature review has demonstrated a plethora of research suggesting the effectiveness in using ML to assist in triaging patients for in-patient and out-patient care.  Healthcare providers are optimistic that if carefully integrated, ML triaging could assist in diverting non-emergency patients from emergency services.

> [!WARNING]
> ATTRA is a project developed for academic purposes. It is not a medical device, has not undergone clinical validation, and must not be used to make medical decisions. Users experiencing concerning symptoms should seek immediate professional medical care.

--- 

### System Components

#### - Diagnosis Tool (DT)
> DESCRIPTION
> - HOW TOOL WORKS
> - OTHER DETAILS

#### - Triage Tool (TT)
> The Triage Tool evaluates the urgency of users symptoms and provides guidance on whether they will likely be admitted or discharged from an emergency department. This tool may assist but should NOT be solely relied on to determine whether you should seek emergency care.
> - The tool utilizes an XGBoost model trained, tested, and validated on a dataset (in RData) collected by Yale School of Medicine and was publically available from research conducted by Hong, Haimovich, and Taylor (2018).
> - Can be found at this link: 
>   - 220/972 attributes and 260k/560k records were used from original dataset.

### Tool Summary

| Component       | Methodology                     | Performance |
|----------------|----------------------------------|-------|
| Triage Model   | XGBoost (Gradient Boosted Trees) | AUC = .85 // thresh = .5 | 
| Diagnosis Tool | Symptom-based classification     | Prototype stage |

---

### Repository Structure:

```
ATTRA/
│
├── main.py                     # Primary Streamlit application entry point
│
├── src/
│   ├── triage_dataset_extract.R   # R script for feature extraction and preprocessing
│   └── triage_xgb.py              # XGBoost model utilities for training and inference
│   
│
├── data/
│   ├── triage/
│   │   ├── XGBoost_Triage_Model_AUC*.model   # Serialized XGBoost model for TT
│   │   ├── feature_*.csv                     # Feature dictionaries for TT questionnaire mapping
│   │   └── response_template.csv             # Input schema for model inference
│   │
│   └── photos/                               # Royalty‑free images for UI
│
├── pages/
│   ├── diagnosis_main.py       # Main script for DT
│   └── triage_main.py          # Main script for TT
│
└── misc/
    └── drugbank_ds_utils.py    # (Unused) DrugBank BaseX query utilities
```

---

## Getting Started
- Install a virtual environment (venv) for Python (recommended)
- Install necessary packages, e.g. streamlit, through homebrew and 'pip install'
- Navigate to local project folder and start venv:
```bash
source venv/bin/activate
```
- Run app:
```bash
streamlit run main.py
```
