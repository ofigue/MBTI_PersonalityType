# to run: python blending.py
import glob
import pandas as pd
import numpy as np
from sklearn import metrics

if __name__ == "__main__":
    files = glob.glob("../model_preds/train*.csv")
    df_train = None
    for f in files:
        if df_train is None:
            df_train = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df_train = df_train.merge(temp_df, on="id", how="left")

    files = glob.glob("../model_preds/test*.csv") # PARAMETER
    df_test = None
    for f in files:
        if df_test is None:
            df_test = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df_test = df_test.merge(temp_df, on="id", how="left")
    
    id = df_test.loc[:, 'id']    
    targets = df_train.target.values
    pred_cols = ['rf_pred', 'xgb_pred', 'xtr_pred']

    for col in pred_cols:
        auc = metrics.roc_auc_score(df_train.target.values, df_train[col].values)
        print(f"{col}, overall_auc={auc}")
    

    print("Average")
    # Train
    avg_pred = np.mean(df_train[['rf_pred', 'xgb_pred', 'xtr_pred']].values, axis=1)
    auc_score = metrics.roc_auc_score(targets, avg_pred)
    print(auc_score)

    # Test
    avg_pred = np.mean(df_test[['rf_pred', 'xgb_pred', 'xtr_pred']].values, axis=1)
    #print(avg_pred)
    #test_df = {'id': id, 'avg_test': avg_pred}
    #test_df = pd.DataFrame(test_df, columns=['id', 'avg_test'])
    #test_df.to_csv("../model_preds/test_submission.csv", index=False) # PARAMETER

    print("Weighted Average")
    # Train
    rf_pred = df_train.rf_pred.values
    xgb_pred = df_train.xgb_pred.values
    xtr_pred = df_train.xtr_pred.values
    avg_pred = (rf_pred + 3 * xgb_pred + xtr_pred)/5
    print(metrics.roc_auc_score(targets, avg_pred))
    
    # Test
    rf_pred = df_test.rf_pred.values
    xgb_pred = df_test.xgb_pred.values
    xtr_pred = df_test.xtr_pred.values
    wgt_pred = (rf_pred + 3 * xgb_pred + xtr_pred)/5
    #print(wgt_pred)
    #test_df = {'id': id, 'wt_avg_test': wgt_pred}
    #test_df = pd.DataFrame(test_df, columns=['id', 'wt_avg_test'])
    #test_df.to_csv("../model_preds/test_sub_Season.csv", index=False) # PARAMETER
  
    print("Rank Averaging")
    # Train
    rf_pred = df_train.rf_pred.rank().values
    xgb_pred = df_train.xgb_pred.rank().values
    xtr_pred = df_train.xtr_pred.rank().values
    avg_pred = (rf_pred + xgb_pred + xtr_pred)/3
    print(metrics.roc_auc_score(targets, avg_pred))

    # Test
    rf_pred = df_test.rf_pred.rank().values
    xgb_pred = df_test.xgb_pred.rank().values
    xtr_pred = df_test.xtr_pred.rank().values
    rnk_pred = (rf_pred + xgb_pred + xtr_pred)/3
    #print(rnk_pred)
    #test_df = {'id': id, 'rnk_avg_test': rnk_pred}
    #test_df = pd.DataFrame(test_df, columns=['id', 'rnk_avg_test'])
    #test_df.to_csv("../model_preds/test_submission.csv", index=False) # PARAMETER

    print("weighted rank averaging")
    # Train
    rf_pred = df_train.rf_pred.rank().values
    xgb_pred = df_train.xgb_pred.rank().values
    xtr_pred = df_train.xtr_pred.rank().values
    avg_pred = (rf_pred + 3 * xgb_pred + xtr_pred)/5
    print(metrics.roc_auc_score(targets, avg_pred))
    
    # Test
    rf_pred = df_test.rf_pred.rank().values
    xgb_pred = df_test.xgb_pred.rank().values
    xtr_pred = df_test.xtr_pred.rank().values
    wgt_rnk_pred = (rf_pred + 3 * xgb_pred + xtr_pred)/5
    #print(wgt_rnk_pred)
    #test_df = {'id': id, 'wtrk_avg_test': wgt_rnk_pred}
    #test_df = pd.DataFrame(test_df, columns=['id', 'wtrk_avg_test'])
    #test_df.to_csv("../model_preds/test_submission.csv", index=False) # PARAMETER

