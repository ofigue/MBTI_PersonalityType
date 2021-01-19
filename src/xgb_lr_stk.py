# to run it: python xgb_stk.py

import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression # ADDED

#import xgboost as xgb

def run_train(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["rf_pred", "xgb_pred", "xtr_pred"]].values
    xvalid = valid_df[["rf_pred", "xgb_pred", "xtr_pred"]].values

    #clf = xgb.XGBClassifier()
    clf = LogisticRegression()
    clf.fit(xtrain, train_df.target.values)
    preds = clf.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(valid_df.target.values, preds)
    print(f"{fold}, {auc}")
    
    valid_df.loc[:, "xgb_prediction"] = preds

    return valid_df

def run_test(pred_df, df_test):
    xtrain = pred_df[["rf_pred", "xgb_pred", "xtr_pred"]].values

    #clf = xgb.XGBClassifier()
    clf = LogisticRegression()
    clf.fit(xtrain, pred_df.target.values)

    xtest = df_test[["rf_pred", "xgb_pred", "xtr_pred"]].values
    preds_test = clf.predict_proba(xtest)[:, 1]

    data = {'id': df_test.id, 'xgb_test_pred': preds_test}
    test_preds = pd.DataFrame(data, columns = ['id', 'xgb_test_pred'])

    return test_preds

if __name__ == "__main__":
    files = glob.glob("../model_preds/train*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how="left")
        
    test_files = glob.glob("../model_preds/test*.csv")
    df_test = None
    for f in test_files:
        if df_test is None:
            df_test = pd.read_csv(f)
        else:
            temp_df1 = pd.read_csv(f)
            df_test = df_test.merge(temp_df1, on="id", how="left")

    dfs_train = []
    dfs_test = []

    for j in range(5):
        temp_df= run_train(df, j)
        dfs_train.append(temp_df)

    fin_valid_df = pd.concat(dfs_train)
    print(metrics.roc_auc_score(fin_valid_df.target.values, fin_valid_df.xgb_prediction.values))
     
    fin_test_df = run_test(df, df_test)
    fin_test_df.to_csv("../model_preds/sub_season.csv", index=False) # PARAMETER

