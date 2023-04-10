import pickle

import numpy as np
import pandas as pd
# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import get_config

config = get_config()
app = FastAPI()

def needed_files_exists():
    paths = config["PATH"]
    return paths["model_path"] and paths["encoder_path"] and paths["lb_path"]

class Employee(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race:str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    class Config:
        schema_extra = {
            "example" : {
                "age": 32,
                "workclass": "Private",
                "fnglt": 287988,
                "education": "11th",
                "education_num": 10,
                "marital_status": "Never-married",
                "occupation": "Transport-moving",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Male",
                "capital_gain": 100,
                "capital_loss": 100,
                "hours_per_week": 1000,
                "native_country": "United-States"
            }
        }
@app.get("/")
async def home():
    return "Census Income Prediction app"

@app.post("/inference")
async def predict(employee: Employee):
    data = pd.DataFrame(
        employee.dict(), index=[0]
    )
    data.columns = [s.replace("_","-") for s in data.columns]
    if needed_files_exists():
        with open(config["PATH"]["model_path"], "rb") as model_file:
            model = pickle.load(model_file)
        with open(config["PATH"]["encoder_path"], "rb") as encoder_file:
            encoder = pickle.load(encoder_file)
        with open(config["PATH"]["lb_path"], "rb") as lb_file:
            lb = pickle.load(lb_file)
    proc_data, _, _, _ = process_data(
                data,
                categorical_features=config["DATA_PROC"]["cat_features"],
                training=False,
                lb=lb,
                encoder=encoder        
    )
    
    prediction = int(inference(model=model, X=proc_data)[0])
    return "<=50K" if prediction == 0 else ">50K"

