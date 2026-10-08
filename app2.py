import pickle
import pandas as pd
import streamlit as st
#Load model
with open("model.pkl","rb") as f:
    model=pickle.load(f)
# try loading scalar(if exists)
try:
    with open("scaler.pkl","rb") as f:
        scaler=pickle.load(f)
except:
    scaler=None
def preprocess_and_predict(features):

#convert to dataframe

    input_data=pd.DataFrame([features])
#get req clmn
    required_columns=model.feature_names_in_

#add missing clmn

    for col in required_columns:
        if col not in input_data.columns:
           input_data[col]=0

#Arrange clmn
    input_data  =input_data[required_columns]



#apply scaling
    if scaler is not None:
           input_data=scaler.transform(input_data)

#prediction
    prediction=model.predict(input_data)
    probability=model.predict_proba(input_data)[:,1]
    return prediction[0],probability[0]


# UI
st.title("Purchase Prediction App")
Age = st.number_input("Age", value=25, step=1)
Gender = st.selectbox("Gender", ["Male", "Female"])
Gender = 1 if Gender == "Male" else 0
AnnualIncome = st.number_input("AnnualIncome", value=34000, step=1)
SpendingScore = st.number_input("SpendingScore", value=25, step=1)
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married"])
MaritalStatus = 1 if MaritalStatus == "Married" else 0




# input dictionary
features = {
    "Age": Age,
    "Gender": Gender,
    "AnnualIncome": AnnualIncome,
    "SpendingScore": SpendingScore,
    "MaritalStatus": MaritalStatus
}


# Prediction
if st.button("Predict"):
    prediction, probability = preprocess_and_predict(features)
    if prediction == 1:
        st.error(f"HIGH chance of purchasing (Probability: {probability:.2f})")
    else:
        st.success(f"LOW chance of purchasing (Probability: {probability:.2f})")
