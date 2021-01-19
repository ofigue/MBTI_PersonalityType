# Ref. https://github.com/abhishekkrthakur/mlframework

from sklearn import ensemble
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression # ADDED
from sklearn import svm


MODELS = {
    #"randomforest": ensemble.RandomForestClassifier(),
    "randomforest": ensemble.RandomForestClassifier(n_estimators=450, n_jobs=-1),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=500, n_jobs=-1, verbose=0),
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier(),
    "XGBClassifier": XGBClassifier(learning_rate=0.01, objetive='binary:logistic', n_estimators=1500, max_depth=4,
    min_child_weight=6, subsample=0.85, colsample_bytree =0.85, nthread=4),
    "XGBClassifier1": XGBClassifier(learning_rate=0.01, objetive='multi:softprob', n_estimators=1500, max_depth=4,
    min_child_weight=6, subsample=0.85, colsample_bytree =0.85, nthread=4),
    "logisticregression": LogisticRegression() # ADDED
    #"SupVectorMachine": svm.SVC(probability=True) # Get probabilities
}