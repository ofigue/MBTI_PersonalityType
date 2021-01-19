# Ref. https://github.com/abhishekkrthakur/mlframework
# to run it: % sh test_ensemble.sh <technique>

import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(test_data_path)
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    pred = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "xgb_pred"]) # <----------- PARAMETER
    return pred
    

if __name__ == "__main__":
    #model = dispatcher.MODELS[MODEL]
    predictions = predict(test_data_path=TEST_DATA, 
                         model_type="XGBClassifier",  # <---------- PARAMETER
                         model_path="models/")
    predictions.loc[:, "id"] = predictions.loc[:, "id"].astype(int)
    predictions.to_csv(f"model_preds/test_xgb.csv", index=False) # <----------- PARAMETER