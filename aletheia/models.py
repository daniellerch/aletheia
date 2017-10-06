
import numpy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import xgboost


def xgb(X_cover, X_stego):
    X=numpy.vstack((X_cover, X_stego))
    y=numpy.hstack(([0]*len(X_cover), [1]*len(X_stego)))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
    #print X_train.shape, y_train.shape, X_val.shape, y_val.shape

    clf = xgboost.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        #min_child_weight=1,
        max_depth=100,
        #gamma=0,
        subsample=1,
        colsample_bytree=0.01
        )
    clf.fit(X_train, y_train)

    y_val_prob = clf.predict(X_val)
    y_val_pred = [round(v) for v in y_val_prob]
    val_score=accuracy_score(y_val, y_val_pred)

    return clf, val_score
