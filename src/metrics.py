# Ref. https://github.com/abhishekkrthakur/mlframework

# Ref. https://www.youtube.com/watch?v=k6q6Sn60ngc
# Example of use in ipython
# from metrics import RegressionMetrics
# y1 = [1.5, 2.5, 0.5, 0.8]
# y2 = [1.2, 2.0, 0.7, 0.9]
# rm = RegressionMetrics()
# rm(“mse”, y1, y2)
# 0.0975
#

import numpy as np
from sklearn import metrics as skmetrics

class RegressionMetrics:
    def __init__(self):
        self.metrics = {
            "mae": self._mae,
            "mse": self._mse,
            "rmse": self._rmse,
            "msle": self._msle,
            "rmsle": self._rmsle,
            "r2": self._r2
        }
    
    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception("Metric not implemented")
        if metric == "mae":
            return self._mae(y_true, y_pred)
        if metric == "mse":
            return self._mse(y_true, y_pred)
        if metric == "rmse":
            return self._rmse(y_true, y_pred)
        if metric == "msle":
            return self._msle(y_true, y_pred)
        if metric == "rmsle":
            return self._rmsle(y_true, y_pred)
        if metric == "r2":
            return self._r2(y_true, y_pred)
    
    @staticmethod
    def _mae(y_true, y_pred):
        return skmetrics.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return skmetrics.mean_squared_error(y_true, y_pred)

    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true, y_pred))

    @staticmethod
    def _msle(y_true, y_pred):
        return skmetrics.mean_squared_log_error(y_true, y_pred)
 
    def _rmsle(self, y_true, y_pred):
        return np.sqrt(self._msle(y_true, y_pred))
 
    @staticmethod
    def _r2(y_true, y_pred):
         return skmetrics.r2_score(y_true, y_pred)


# Ref. https://www.youtube.com/watch?v=uuQ4XLeyWG0
# Example of use in ipython:
# t = [0, 0, 1, 0, 1, 1] 
# p = [0, 0, 1, 0, 1, 0]
# cm = ClassificationMetrics()
# cm("accuracy", t, p)
# 0.833333333334
#


class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "precision": self._precision,
            "recall": self._recall,
            "auc": self._auc,
            "logloss": self._logloss
        }
    
    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception("Metric not implemented")
        if metric == "auc":
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for AUC")
        elif metric == "logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for logloss")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)