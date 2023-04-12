import pytest
import sys
from fastapi.testclient import TestClient
sys.path.insert(0,"./")

from main import Employee, app

client = TestClient(app)

@pytest.fixture
def example_less_50():
    data = {
        "age": 32,
        "workclass": "Private",
        "fnlgt": 287988,
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
    return Employee(**data)
@pytest.fixture
def example_greater_50():
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
    return Employee(**data)
def test_home_page():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Census Income Prediction app"

def test_post_prediction_class_0(example_less_50):
    r = client.post("/inference", json=example_less_50.dict())
    assert r.status_code == 200
    assert r.json() == "<=50K"

def test_post_prediction_class_1(example_greater_50):
    r = client.post("/inference", json=example_greater_50.dict())
    assert r.status_code == 200
    assert r.json() == ">50K"
