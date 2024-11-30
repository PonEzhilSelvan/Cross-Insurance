# import necessary libraries
import streamlit as st
import pandas as pd
import joblib

st.title("Cross Insurance Prediction")

# read the dataset to fill list values
df = pd.read_csv('data/train.csv')
 
# create input fields 
gender = st.selectbox("gender", pd.unique(df['Gender']))
age = st.number_input('age')
driving_license = st.number_input('driving_license')
region_code = st.number_input('region_code')
previously_insured = st.number_input('previously_insured')
vehicle_age = st.selectbox("vehicle_age", pd.unique(df['Vehicle_Age']))
vehicle_damage = st.selectbox("vehicle_damage", pd.unique(df['Vehicle_Damage']))
annual_premium = st.number_input('annual_premium')
policy_sales_channel = st.number_input('policy_sales_channel')
vintage = st.number_input('vintage')
 
# convert the input values to dict

inputs = {
   
  "Gender": gender,
  "Age": age,
  "Driving_License": driving_license,
  "Region_Code": region_code,
  "Previously_Insured": previously_insured,
  "Vehicle_Age": vehicle_age,
  "Vehicle_Damage": vehicle_damage,
  "Annual_Premium": annual_premium,
  "Policy_Sales_Channel": policy_sales_channel,
  "Vintage": vintage 
}

# on click
if st.button("Predict"):
    # load the pickle model 
    model = joblib.load('linear_regression.pkl')

    X_input = pd.DataFrame(inputs,index=[0])
    # predict the target using the loaded model
    prediction = model.predict(X_input)
    # display the result
    st.write(prediction)
