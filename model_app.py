#importing libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd 
import joblib

app = FastAPI()

class Input(BaseModel):
    gender  : object
    age     : int
    driving_license : int
    region_code : float
    previously_insured : int
    vehicle_age     : object
    vehicle_damage  : object
    annual_premium  : float
    policy_sales_channel  : float
    vintage               : int

    #Gender  : object
    #Age     : int
    #Driving_License : int
    #Region_Code : float
    #Previously_Insured : int
    #Vehicle_Age     : object
    #Vehicle_Damage  : object
    #Annual_Premium  : float
    #Policy_Sales_Channel  : float
    #Vintage               : int

class Output(BaseModel):
    response      : int

@app.post("/predict")
def predict(data: Input) -> Output :
    X_input = pd.DataFrame([[data.gender, data.age, data.driving_license,
                             data.region_code,data.previously_insured, data.vehicle_age,
                             data.vehicle_damage, data.annual_premium,data.policy_sales_channel,
                             data.vintage]])

    X_input.columns = ['Gender','Age','Driving_License','Region_Code', 'Previously_Insured',
                       'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel','Vintage']
                       
    #load model
    model = joblib.load('linear_regression.pkl')
    
    #predict Model
    prediction = model.predict(X_input)

    #output
    return Output(response = prediction)
    
    
    
    
    


#uvicorn model_app:app --reload  