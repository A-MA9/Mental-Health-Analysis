import streamlit as st
import pandas as pd
import pickle
import numpy as np

local_pickle_path = "/home/ec2-user/stress_level_models.pkl"

with open(local_pickle_path, "rb") as f:
    model_data = pickle.load(f)
    model = model_data["xgboost"]
    scaler = model_data["scaler"]
    selected_features = model_data["selected_features"]

categorical_mappings = {
    "Course": {'Engineering': 1, 'Business': 2, 'Medical': 3, 'Law': 4, 'Others': 5},
    "Sleep_Quality": {'Good': 3, 'Average': 2, 'Poor': 1},
    "Physical_Activity": {'High': 3, 'Moderate': 2, 'Low': 1},
    "Diet_Quality": {'Good': 3, 'Average': 2, 'Poor': 1},
    "Social_Support": {'High': 3, 'Moderate': 2, 'Low': 1},
    "Relationship_Status": {'Married': 1, 'Single': 2, 'In a Relationship': 3},
    "Substance_Use": {'Never': 1, 'Occasionally': 2, 'Frequently': 3},
    "Counseling_Service_Use": {'Never': 1, 'Occasionally': 2, 'Frequently': 3},
    "Family_History": {'Yes': 1, 'No': 2},
    "Chronic_Illness": {'Yes': 1, 'No': 2},
    "Extracurricular_Involvement": {'High': 1, 'Moderate': 2, 'Low': 3},
    "Residence_Type": {'With Family': 1, 'Off-Campus': 2, 'On-Campus': 3},
    "Gender": {'Male': 1, 'Female': 2}
}

range_mappings = {
    "Stress_Level": list(range(1, 6)),
    "Anxiety_Score": list(range(1, 6)),
    "Financial_Stress": list(range(1, 6))
}

st.title("Depression Score Predictor")

user_data = {}
for col in selected_features:
    if col in categorical_mappings:
        user_data[col] = categorical_mappings[col][st.selectbox(f"Select {col}", list(categorical_mappings[col].keys()), key=f"{col}_select")]
    elif col in range_mappings:
        user_data[col] = st.selectbox(f"Select {col}", range_mappings[col], key=f"{col}_select")
    elif col in ["Age", "Semester_Credit_Load"]:
        user_data[col] = st.number_input(f"Enter {col}", value=0, step=1, key=f"{col}_input")
    else:
        user_data[col] = st.number_input(f"Enter {col}", value=0.0, key=f"{col}_input")

input_df = pd.DataFrame([user_data])
input_df[selected_features] = scaler.transform(input_df[selected_features])

if st.button("Analyze Mental Health"):
    prediction = model.predict(input_df)[0]
    if prediction < 2:
        st.success("The student is in a healthy mental state.")
    elif 2.1 <= prediction <= 3.5:
        st.warning("Some signs of distress. Consider monitoring mental health and seeking support if needed.")
    else:
        st.error("High risk of depression. Immediate professional help is recommended.")

