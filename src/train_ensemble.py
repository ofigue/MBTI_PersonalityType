# Stacking
# Ref. https://www.youtube.com/watch?v=TuIgtitqJho&t=6s
# to run it: sh train_ensemble.sh <technique>

import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
#TEST_DATA = os.environ.get("TEST_DATA")
#FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")


def run_training(fold):
    df = pd.read_csv(TRAINING_DATA)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    xtrain = df_train.drop(["id", "target", "kfold"], axis=1)
    xvalid = df_valid.drop(["id", "target", "kfold"], axis=1)
    
    clf = dispatcher.MODELS[MODEL]
    clf.fit(xtrain, ytrain)
    pred = clf.predict_proba(xvalid)[:, 1]

    auc = metrics.roc_auc_score(yvalid, pred)
    print(f"fold={fold}, auc={auc}")

    joblib.dump(clf, f"models/{MODEL}_{fold}.pkl")
    joblib.dump(xtrain.columns, f"models/{MODEL}_{fold}_columns.pkl")

    df_valid.loc[:, "xgb_pred"] = pred # <------ PARAMETER
    return df_valid[["id", "target", "kfold", "xgb_pred"]] # <------- PARAMETER

if __name__ == "__main__":
    dfs = []
    for j in range(5):
        temp_df = run_training(j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    fin_valid_df.to_csv("model_preds/train_xgb.csv", index=False) # <------- PARAMETER