# Ref. https://github.com/abhishekkrthakur/mlframework
# to run it: % sh train.sh <technique>

import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import xgboost as XGBClassifier


from . import dispatcher

from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

TRAINING_DATA = os.environ.get("TRAINING_DATA")
#TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

#FOLD_MAPPPING = {
#    0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
#    1: [0, 2, 3, 4, 5, 6, 7, 8, 9],
#    2: [0, 1, 3, 4, 5, 6, 7, 8, 9],
#    3: [0, 1, 2, 4, 5, 6, 7, 8, 9],
#    4: [0, 1, 2, 3, 5, 6, 7, 8, 9],
#    5: [0, 1, 2, 3, 4, 6, 7, 8, 9],
#    6: [0, 1, 2, 3, 4, 5, 7, 8, 9],
#    7: [0, 1, 2, 3, 4, 5, 6, 8, 9],
#    8: [0, 1, 2, 3, 4, 5, 6, 7, 9],
#    9: [0, 1, 2, 3, 4, 5, 6, 7, 8]
#}

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    cntizer = CountVectorizer(analyzer="word", max_features=1000, tokenizer=None, preprocessor=None, stop_words=None, max_df=0.5, min_df=0.1) 
    tfizer = TfidfTransformer()

    # Xgboost 
    # setup parameters for xgboost
    param = {}
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    param['eta'] = 0.6
    param['ntrees'] = 300
    param['subsample'] = 0.93
    param['max_depth'] = 2
    param['silent'] = 1
    param['n_jobs'] = 8
    param['num_class'] = 3  #len(unique_type_list)
    #xgb_class = xgb.XGBClassifier(**param)

    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)
    
    #ytrain = train_df.target.values
    #yvalid = valid_df.target.values

    #train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    #valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    #valid_df = valid_df[train_df.columns]

    posts_trainX = train_df.loc[:, 'posts'].to_numpy()
    posts_validX = valid_df.loc[:, 'posts'].to_numpy()

    personality_trainY = train_df.loc[:, 'target'].to_numpy()
    personality_validY = valid_df.loc[:, 'target'].to_numpy()

    probs = np.ones((len(personality_validY), 3))
    
    X_train = cntizer.fit_transform(posts_trainX)
    X_valid = cntizer.transform(posts_validX)
    
    #X_train_tfidf = tfizer.fit_transform(X_train_cnt)
    #X_valid_tfidf = tfizer.transform(X_valid_cnt)

    xg_train = XGBClassifier.DMatrix(X_train, label=personality_trainY)
    xg_test = XGBClassifier.DMatrix(X_valid, label=personality_validY)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    
    clf = dispatcher.MODELS[MODEL]

    num_round = 30
    bst = clf.fit(param, xg_train, num_round, watchlist, early_stopping_rounds=6)
    
    preds = bst.predict(xg_test)
    probs = np.multiply(probs, preds)
    preds = np.array([np.argmax(prob) for prob in preds])
    
    score = f1_score(personality_validY, preds, average='weighted')
    #print('%s : %s' % (str(model).split('(')[0], score))
    print(score)

    # data is ready to train
    #clf = dispatcher.MODELS[MODEL]
    #clf.fit(train_df, ytrain)
    #preds = clf.predict_proba(valid_df)[:, 1]
    #print(metrics.roc_auc_score(yvalid, preds))

    #joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    #joblib.dump(bst, f"models/{MODEL}_{FOLD}.pkl")
    #joblib.dump(xg_train.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
