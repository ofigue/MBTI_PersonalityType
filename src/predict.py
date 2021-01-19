# Ref. https://github.com/abhishekkrthakur/mlframework
# to run it: % sh predict.sh

import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

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

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub
    

if __name__ == "__main__":
    submission = predict(test_data_path="input/testSet.csv", 
                         model_type="XGBClassifier", 
                         model_path="models/")
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"models/xgb_submission.csv", index=False)