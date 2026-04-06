# Titanic Survival Prediction

Predict whether a passenger would have survived the Titanic disaster using machine learning and a user-friendly Streamlit app.

## Project Overview

This project leverages historical Titanic passenger data to build a machine learning model that predicts survival. It includes:

- **Data preprocessing**: handling missing values, encoding categorical features, feature engineering (`Family_Size`, `Is_Alone`, `Age_Group`)  
- **Modeling**: Random Forest classifier with tuned hyperparameters
- **Deployment**: Interactive **Streamlit app** for real-time prediction  

The app allows users to input passenger details and outputs whether the passenger would survive or not.

## Features

- Extracted **Title** from names (Mr, Miss, Mrs, Master, Rare)  
- Encoded categorical features (`Sex`, `Embarked`)  
- Created new features:
  - `Family_Size` = `SibSp + Parch + 1`  
  - `Is_Alone`  
  - `Age_Group` bins  
- Randomforestclassifier model for high accuracy (~82–85%)  
- Streamlit app with sidebar inputs and prediction output  

## Tech Stack

- Python  
- pandas, numpy – Data manipulation  
- scikit-learn – Preprocessing and evaluation  
- Random forest Classifier – Machine learning model  
- Streamlit – Web app deployment  
- joblib – Model serialization  

## How to Use

1. **Clone the repository**:

git clone https://github.com/Akash-Sabbani/Titanic-survival
cd titanic-survival-predictor

2. **Install dependencies**:
pip install -r requirements.txt

3.**Run the Streamlit app**:
streamlit run app.py