import json

import requests

if __name__=="__main__":
    data = {
        "age": 66, 
        "workclass": " Self-emp-not-inc", 
        "fnlgt": 102686, 
        "education": " Masters", 
        "education_num": 14, 
        "marital_status": " Married-civ-spouse", 
        "occupation": " Exec-managerial", 
        "relationship": " Husband", 
        "race": " White", 
        "sex": " Male", 
        "capital_gain": 0, 
        "capital_loss": 0, 
        "hours_per_week": 20, 
        "native_country": " United-States"
    }

    r = requests.post("https://churn-prediction-app-ek0o.onrender.com/inference", data=json.dumps(data))
    print(f"Status code : {r.status_code}")
    print(f"Income prediction : {r.json()}")
