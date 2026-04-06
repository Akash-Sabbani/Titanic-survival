# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load Model and Columns
@st.cache_resource
def load_model():
    try:
        model = joblib.load("titanic_model.pkl")
        return model
    except FileNotFoundError:
        return None
model = load_model()

model_columns = joblib.load("model_columns.pkl")  # save this from your training script

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# 2. App Title

st.title("🚢 Titanic Survival Predictor")
st.markdown(
    """
    Predict whether a passenger would have survived the Titanic disaster.
    Fill in the details below and click **Predict**.
    """
)

# 3. Sidebar Input

st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 3 = 3rd)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.number_input("Fare", value=50.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])
family_size = st.sidebar.number_input("Family Size", min_value=1, max_value=11, value=1)
is_alone = 1 if family_size == 1 else 0
title = st.sidebar.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])


# 4. Prepare Input for Model

def preprocess_input(pclass, sex, age, fare, embarked, family_size, is_alone, title, model_columns):
    # Age group
    if age <= 12:
        age_group = 0
    elif age <= 20:
        age_group = 1
    elif age <= 40:
        age_group = 2
    elif age <= 60:
        age_group = 3
    else:
        age_group = 4

    # Build dict for all possible features
    input_dict = {
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare],
        'Family_Size': [family_size],
        'Is_Alone': [is_alone],
        'Age_Group': [age_group],
        'Sex_male': [1 if sex=='male' else 0],
        'Embarked_C': [1 if embarked=='C' else 0],
        'Embarked_Q': [1 if embarked=='Q' else 0],
        'Title_Mr': [1 if title=='Mr' else 0],
        'Title_Miss': [1 if title=='Miss' else 0],
        'Title_Mrs': [1 if title=='Mrs' else 0],
        'Title_Master': [1 if title=='Master' else 0],
        'Title_Rare': [1 if title=='Rare' else 0]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_dict)

    # Add any missing columns with 0
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[model_columns]

    return input_df

input_df = preprocess_input(pclass, sex, age, fare, embarked, family_size, is_alone, title, model_columns)

if st.sidebar.button("Predict Survival"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("This passenger would SURVIVE!")
    else:
        st.error("This passenger would NOT survive.")

st.markdown("---")
st.subheader("Model Information")

st.metric("Algorithm", "ML Classification")
st.metric("Accuracy", "~82%")
st.metric("Dataset", "800+ passengers")