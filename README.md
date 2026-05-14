# DATA606 - Data Science Capstone
## At-Home Triage and Treatment Recommendation Application (ATTRA)

### Purpose: 
Globally, medical facilities are inundated by the numbers of patients seeking emergency care.  Inefficient triaging causes non-emergency patient care to consume precious resources from those facing life-threatening medical situations.  To help make more efficient use of scarce resources, we offer an at-home triage and treatment recommendation application to offer initial triaging and disease identification, to recommend medications, and advise admission to a hospital, if necessary.  A literature review has demonstrated a plethora of research suggesting the effectiveness in using ML to assist in triaging patients for in-patient and out-patient care.  Healthcare providers are optimistic that if carefully integrated, ML triaging could assist in diverting non-emergency patients from emergency services.

### Components

#### - Diagnosis Tool
> DESCRIPTION
> - HOW TOOL WORKS
> - OTHER DETAILS

#### - Triage Tool
> The Triage Tool evaluates the urgency of users symptoms and provides guidance on whether they will likely be admitted or discharged from an emergency department. This tool may assist but should NOT be solely relied on to determine whether you should seek emergency care.
> - The tool utilizes an XGBoost model trained, tested, and validated on a dataset (in RData) collected by Yale School of Medicine and was publically available from research conducted by Hong, Haimovich, and Taylor (2018).
> - Can be found at this link: 
>   - 220/972 attributes and 260k/560k records were used from original dataset.


### Documentation:

> - main.py: main script for StreamLit app.

#### src
> - triage_dataset_extract.R
>     - Filters out selected 219/972 attributes from Yale School of Medicine dataset (named "Yale_Dataset.RData").
> - triage_xgb.py: script that provides xgboost-related functions used in *triage.ipynb* and *triage_main.py*  

#### data
> triage
> - XGBoost_Triage_Model_AUC#*: contains current model to load and use for Triage Tool.
> - feature_.*.csv: dictionaries mapping original features to options for questionnaire
> - response_template.csv: template for inputing user response from questionnaire into model
> 
> photos
> - royalty-free photos used for GUI

#### misc
> - (not used) drugbank_ds_utils.py: script queries a BaseX server with DrugBank database (free educational version) uploaded. No actual database in the script. (For access to free, educational-use database, reach out to a DrugBank representative.)

#### pages
> - diagnosis_main.py: main script for diagnosis tool
> - triage_main.py: main script for triage tool
