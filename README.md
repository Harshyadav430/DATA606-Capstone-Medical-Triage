# DATA606 - Data Science Capstone
### At-Home Triage and Treatment Recommendation Application (ATTRA)
#### Urja Kuppa | Ayushi Mukesh Patel | Tyler Snow | Harsh Mahesh Yadav

---

### *Purpose*: 
Globally, medical facilities are inundated by the numbers of patients seeking emergency care.  Inefficient triaging causes non-emergency patient care to consume precious resources from those facing life-threatening medical situations.  To help make more efficient use of scarce resources, we offer an at-home triage and treatment recommendation application to offer initial triaging and disease identification, to recommend medications, and advise admission to a hospital, if necessary.  A literature review has demonstrated a plethora of research suggesting the effectiveness of using machine learning (ML) to assist in triaging patients for in-patient and out-patient care.  Healthcare providers are optimistic that if carefully integrated, ML triaging could assist in diverting non-emergency patients from emergency services.


### *Research Questions*: 

- What is the potential business value of deploying an at‑home disease prediction and medication recommendation tool based on the users' chief complaints in terms of user adoption, cost savings, and scalable digital healthcare services?

- How can we make the prediction process transparent, understandable, and trustworthy so patients feel confident using the system before booking appointments or spending money on emergency room visits?


> [!WARNING]
> ATTRA is a project developed for academic purposes. It is not a medical device, has not undergone clinical validation, and must not be used to make medical decisions. Users experiencing concerning symptoms should seek immediate professional medical care.

--- 

### *System Components*

#### - Diagnosis Tool (DT)
> DESCRIPTION
> - HOW TOOL WORKS
> - OTHER DETAILS

#### - Triage Tool (TT)
> The Triage Tool evaluates the urgency of users' symptoms and self-obtained vitals to evaluate whether they will likely be admitted or discharged from an emergency department.
> - Dataset was collected by Yale School of Medicine and available from research conducted by Hong, Haimovich, and Taylor (2018):
>   - GitHub: https://github.com/yaleemmlc/admissionprediction/tree/master
>   - Article: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201016
> - User Input: multiple-choice, free-text, and checkbox questions for easy, efficient input.
> - Model Output: binary prediction (admission vs. discharge) and probability (model certainty)

> [!WARNING]
> This tool may assist but should NOT be solely relied on to determine whether you should seek emergency care.

### Tool Performance Summary

| Component       | Methodology                     | Features | Records | Performance |
|----------------|----------------------------------|-------|-------|-------|
| Triage Model   | XGBoost (Gradient Boosted Trees) | 220/972 | 260k/560k | AUC = .85 // thresh = .5 | 
| Diagnosis Tool | Symptom-based classification     | | | Prototype stage |

---

### *Repository Structure*:

```
ATTRA/
│
├── data/
│   ├── photos/                               # Royalty‑free images for UI
│   └── triage/
│       ├── XGBoost_Triage_Model_AUC*.model   # Serialized XGBoost model for TT
│       ├── feature_*.csv                     # Feature dictionaries for TT questionnaire mapping
│       └── response_template.csv             # Input schema for model inference
│
├── main.py                                   # Primary Streamlit application entry point
│
├── misc/
│   └── drugbank_ds_utils.py                  # (Unused) DrugBank BaseX query utilities
│
├── pages/
│   ├── diagnosis_main.py                     # Main script for DT
│   └── triage_main.py                        # Main script for TT
│
└── src/
    ├── triage_dataset_extract.R              # R script for feature extraction and preprocessing
    └── triage_xgb.py                         # XGBoost model utilities for training and inference
```


---

## *Getting Started*
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
