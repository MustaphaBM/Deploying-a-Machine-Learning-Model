# Script to train machine learning model.

import logging
import pickle

# Add the necessary imports for the starter code.
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model

pd.set_option('display.max_columns', 16)
logging.basicConfig(
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

def get_config():
    with open("starter/config.yaml", "r") as stream:
        try:
            config  = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
    return config

def run_pipeline(data_path, categorical_features, target_column, model_hyperparams, model_path, encoder_path, lb_path, test_size=0.2):
    """Processes the data, train a decision tree model and save it to disk

    Args:
        data_path (str): Path to raw data
        categorical_features (list): List of categorical features
        target_column (str) : the label
        model_hyperparams (dict) : the hyper parameters of the decision tree model
        model_path (str) : where to save the trained model
        encoder_path (str): the path of the encoder
        lb_path (str): Label Binarizer path
        test_size (float, optional): test size

    """
    # Add code to load in the data.
    data = pd.read_csv(data_path)
    # remove extra spaces from column names in the data
    data.columns = [s.replace(" ","") for s in data.columns]
    logging.info(f"Data was loaded sucessfully from {data_path}.")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=test_size)
    logging.info("Data was splitted successfully to train and test set.")
    print(train)
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=categorical_features, label=target_column, training=True
    )
    
    logging.info("Training data was successfully processed.")
    # Train and save a model.
    model = train_model(X_train=X_train, y_train=y_train, model_config=model_hyperparams)
    logging.info("Model was successfully trained.")

    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(encoder_path, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    with open(lb_path, "wb") as lb_file:
        pickle.dump(lb, lb_file)
    
    X_test, y_test, _, _ = process_data(test,categorical_features=categorical_features,training=False, label=target_column,encoder=encoder,lb=lb)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test,preds)

    logging.info(f"{precision=}.")
    logging.info(f"{recall=}.")
    logging.info(f"{fbeta=}.")

    
    logging.info(f"Model was successfully saved in {model_file}.")


if __name__ == "__main__":

    config = get_config()
    run_pipeline(
        data_path=config["PATH"]["relative_data_path"], 
        categorical_features=config["DATA_PROC"]["cat_features"], 
        target_column=config["DATA_PROC"]["label"], 
        model_hyperparams=config["MODEL_PARAMS"], 
        model_path=config["PATH"]["model_path"], 
        encoder_path=config["PATH"]["encoder_path"], 
        lb_path=config["PATH"]["lb_path"], 
        test_size=config["DATA_PROC"]["test_size"]
    )


