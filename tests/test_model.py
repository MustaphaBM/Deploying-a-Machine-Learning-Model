import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from main import get_config
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model

sys.path.insert(0,"./")



@pytest.fixture
def config():
    return get_config()


@pytest.fixture
def model(config):
    with open(config["PATH"]["model_path"], "rb") as model_file:
        model = pickle.load(model_file)
    return model


@pytest.fixture
def get_data(config):
    df = pd.read_csv("data/census.csv")
    df.columns = [s.replace(" ","") for s in df.columns]
    train,test = train_test_split(df, test_size=config["DATA_PROC"]["test_size"])
    X_train, y_train, encoder,lb = process_data(train,
                                                categorical_features=config["DATA_PROC"]["cat_features"],
                                                training=True, 
                                                label="salary")
    X_test,y_test, _, _ = process_data(test,
                                        categorical_features=config["DATA_PROC"]["cat_features"],
                                        label="salary",
                                        training=False,
                                        encoder=encoder,
                                        lb=lb)
    return X_train, y_train, X_test, y_test,

def test_train_model(get_data,config):
    X_train, y_train, _, _ = get_data
    dummy_model = train_model(X_train=X_train,y_train=y_train, model_config=config["MODEL_PARAMS"])
    assert isinstance(dummy_model, RandomForestClassifier)

def test_inference(get_data,model):
    _,_,X_test,_ = get_data
    preds = inference(model=model, X=X_test)
    assert isinstance(preds, np.ndarray)
    assert len(X_test) == len(preds)

def test_compute_model_metrics(get_data,model):
    _,_,X_test,y_test = get_data
    preds = inference(model=model, X=X_test)
    precision,recall,fbeta = compute_model_metrics(y_test,preds)

    assert 0 < precision < 1
    assert 0 < recall < 1
    assert 0 < fbeta < 1
