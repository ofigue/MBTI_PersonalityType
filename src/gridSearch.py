# to run it: % sh gridSearch.sh <technique>

import os
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression # ADDED
from sklearn import svm
from sklearn.model_selection import GridSearchCV

TRAINING_DATA = os.environ.get("TRAINING_DATA")
MODEL = os.environ.get("MODEL")

model_params = {
    'svm': {'model': svm.SVC(gamma='auto'),
    'params' : {'C': [1,10,20], 'kernel': ['rbf']}},
    'random_forest': {'model': ensemble.RandomForestClassifier(),
    'params' : {'n_estimators': [100, 200, 300]}},    #, 'max_depth': [1,3,5], 'criterion': ['gini', 'entropy']}},
    'logistic_regression' : {'model': LogisticRegression(solver='liblinear',multi_class='auto'),
    'params': {'C': [1,5,10]}}
}

if __name__ == "__main__":
    train = pd.read_csv(TRAINING_DATA)

    xtrain = train.drop(["id", "target"], axis=1)
    ytrain = train.target
    
    clf =  model_selection.GridSearchCV(
        estimator=model_params[MODEL]['model'],
        param_grid=model_params[MODEL]['params'],
        scoring='accuracy',
        cv=5,
        verbose=10,
        return_train_score=False
        )
    
    clf.fit(xtrain,ytrain)

    print(clf.best_score_)
    print(clf.best_estimator_.get_params())
