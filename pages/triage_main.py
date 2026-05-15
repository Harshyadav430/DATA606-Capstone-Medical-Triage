import streamlit as st
import pandas as pd
import re
from pathlib import Path
from src import triage_xgb as t_xgb

# Configurations
st.set_page_config(page_title="Triage Tool", layout="centered")

# File Paths
data_triage_path=Path("data") / "triage"

# Model
model_path = t_xgb.get_xgb_model_path()
model = t_xgb.get_xgb_model(model_path)

# --- Title ---
st.title("Triage Tool")

df_response = pd.read_csv(f'{data_triage_path}/response_template.csv') # one record from Yale Medicine Dataset to fill in with responses

def save_response(df, orig_feature, response):
    df[orig_feature] = response

# Load features for questions
df_cc = pd.read_csv(f'{data_triage_path}/feature_cc.csv').sort_values(by='orig_feature', ascending=True)
chief_complaints_dict = dict(zip(df_cc['orig_feature'], df_cc['segment_feature']))

df_counts = pd.read_csv(f'{data_triage_path}/feature_counts.csv')
counts_dict = dict(zip(df_counts['orig_feature'], df_counts['segment_feature']))

df_demographics = pd.read_csv(f'{data_triage_path}/feature_demographics.csv')
demographics_list = list(zip(df_demographics['orig_feature'], df_demographics['segment_feature'], df_demographics['options']))

df_vitals = pd.read_csv(f'{data_triage_path}/feature_vitals.csv')
vitals_dict = dict(zip(df_vitals['orig_feature'], df_vitals['replace_feature']))


# Ask for demographical information
st.write("### Provide the following information:")
for feature in demographics_list:
    if feature[0] == 'age':
        value = st.number_input(
            f"{feature[1].title()}",
            min_value=0,
            max_value=120,
            step=1,
            format="%d"
        )
        save_response(df_response, feature[0], int(value))

    else:
        choices = feature[2][1:-1].replace("'","").split(',')
        choice = st.selectbox(f"{feature[1].title()}", choices)

        save_response(df_response, feature[0], str(choice)) 

# Ask for counts for count-based medical history
st.write("### Enter number of:")
for feature in counts_dict:
    value = st.number_input(
        f"{counts_dict[feature].title()}",
        min_value=0,
        max_value=100,
        step=1,
        format="%d"
    )
    save_response(df_response, feature, int(value))
    
# Ask for vitals
st.write("### Enter your current:")
for feature in vitals_dict:
    if vitals_dict[feature] == 'temperature':
      value = st.number_input(
          f"{vitals_dict[feature].capitalize()}",
          min_value=0.0,
          max_value=140.0,
          format="%f"
      )
      save_response(df_response, feature, float(value))

    else:
      value = st.number_input(
          f"{vitals_dict[feature].upper()}",
          min_value=0,
          max_value=500,
          step=1,
          format="%d"
      )
      save_response(df_response, feature, int(value))


# Ask for chief complaints
st.write("### Select all conditions that apply:")

selected = {}

# Scrollable container
with st.container():
    st.markdown('<div class="scroll-box">', unsafe_allow_html=True)

    for feature in chief_complaints_dict:
        
        if chief_complaints_dict[feature].isupper(): # keeps acronyms capitalized
            selected[feature] = st.checkbox(chief_complaints_dict[feature])
        else:
            selected[feature] = st.checkbox(chief_complaints_dict[feature].title()) # title case for remaining cc features

    st.markdown('</div>', unsafe_allow_html=True)

for orig_feature in selected:
    save_response(df_response, orig_feature, selected[orig_feature])

if st.button("Submit"):
    # your final actions here
    user_response_path = data_triage_path / "user_response.csv"
    df_response.to_csv(user_response_path, index=False)
    st.success("Predicting.")

    t_xgb.cast_response(df_response)

    disp, pred_percent = t_xgb.predict_discharge(model,df_response, .45)

    # --- Display colored result box ---
    if not disp:
        st.markdown(
            """
            <div style="
                background-color:#B22222;
                padding:20px;
                border-radius:8px;
                font-size:20px;
                font-weight:bold;
                text-align:center;">
                Admission.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                background-color:#228B22;
                padding:20px;
                border-radius:8px;
                font-size:20px;
                font-weight:bold;
                text-align:center;">
                Discharge.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Explanation text ---
    st.write(f"It is determined that you are {pred_percent} likely to be discharged.")

if st.button("Return Home"):
    st.switch_page("main.py")


