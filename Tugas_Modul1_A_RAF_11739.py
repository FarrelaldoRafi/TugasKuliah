import streamlit as st
import pandas as pd
import pickle
import os

model_directory = r'C:\Users\Admin\Music\Machine Learning\Tugas_A_11739'

model_path = os.path.join(model_directory, 'RF_diabetes_model.pkl')

if os.path.exists(model_path):
    try:
        with open(model_path,'rb') as F:
            loaded_model = pickle.load(F)

            RF_model = loaded_model[0]

            st.title("Prediksi Diabetes")

            st.write("Aplikasi ini digunakan untuk membantu memprediksi penyakit diabetes pada seseorang")

            pregnancies = st.slider("Pregnancies", min_value=0, max_value=17, step=1)
            glucose = st.slider("Glucose (mg/dl)", min_value=0.0, max_value=199.0, step=0.1)
            bloodPressure = st.slider("Blood Pressure (mmHg)", min_value=0, max_value=122, step=2)
            skinThickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=99, step=2)
            insulin = st.slider("Insulin (µU/mL)", min_value=0, max_value=846, step=10)
            bmi = st.slider("BMI", min_value=0.0, max_value=67.1, step=0.1)
            diabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", min_value=0.07, max_value=2.42, step=0.1)
            age = st.slider("Age", min_value=21, max_value=81, step=1)

            input_data = [[pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,diabetesPedigreeFunction,age]]

            if st.button("Prediksi!"):
                RF_model_prediction = RF_model.predict(input_data)
                outcome_names = {0: 'Tidak Diabetes', 1: 'Diabetes'}
                st.write(F"Orang tersebut diprediksi **{outcome_names[RF_model_prediction[0]]}** oleh RF")

    except Exception as E:
        st.error(F"Terjadi Kesalahan: {E}")  
else:
    print("File 'RF_diabetes_model.pkl' tidak ditemukan di direktori.")