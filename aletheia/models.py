
import numpy
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count


class EnsembleSVM:

    def __init__(self, n_estimators=20, max_samples=500, max_features=2000,
                 n_randomized_search_iter=10):

        self.n_estimators=n_estimators
        self.max_samples=max_samples
        self.max_features=max_features
        self.n_randomized_search_iter=n_randomized_search_iter

    def _prepare_classifier(self, params):

        X_train, y_train = params

        tuned_parameters = [{
            'kernel': ['rbf'], 
            'gamma': [1e-4,1e-3,1e-2,1e-1,1e+0,1e+1,1e+2,1e+3,1e+4],
            'C': [1e+0,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6,1e+7,1e+8,1e+9]
        }]

        clf=RandomizedSearchCV(svm.SVC(), tuned_parameters[0], 
                               n_iter=self.n_randomized_search_iter, n_jobs=1)
        clf.fit(X_train, y_train)
              
        params=clf.best_params_
        clf=svm.SVC(kernel=params['kernel'], C=params['C'], 
            gamma=params['gamma'], probability=True)
        clf.fit(X_train, y_train)

        return clf


    def fit(self, X_train, y_train):
        
        self.selector = SelectKBest(f_classif, k=self.max_features)
        self.selector.fit(X_train, y_train)

        X_train=self.selector.transform(X_train)

        param_list=[]
        idx = range(len(y_train))
        for i in range(self.n_estimators):
            random.shuffle(idx)
            param_list.append((X_train[idx[:self.max_samples]], 
                               y_train[idx[:self.max_samples]]))

        pool = ThreadPool(cpu_count())
        self.clf_list = pool.map(self._prepare_classifier, param_list)
        pool.close()
        pool.join()


    def predict_proba(self, X):
        X_val=self.selector.transform(X)
        y_val_pred=[0]*len(X_val)
        for clf in self.clf_list:
            P=clf.predict_proba(X_val)
            for i in range(len(P)):
                y_val_pred[i]+=P[i][0]
        return y_val_pred


    def score(self, X, y):
        y_pred=self.predict_proba(X)
        ok=0
        for i in range(len(y)):
            p=float(y_pred[i])/len(self.clf_list)
            if  p > 0.5 and y[i]==0: ok+=1
            elif p <= 0.5 and y[i]==1: ok+=1

        return float(ok)/len(y)

   
# }}}






