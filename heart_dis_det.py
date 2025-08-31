# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 10:31:02 2025

@author: Sunita
"""

import streamlit as st
import pickle
import numpy as np

# load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction")

# input fields
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Other)", [0, 1, 2, 3])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Yes, disease may be present")
    else:
        st.success("No disease detected")
