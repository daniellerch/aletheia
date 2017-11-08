
import numpy
import random
import os
import tempfile
import shutil
import subprocess


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm

from scipy.io import savemat, loadmat

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

# {{{ EnsembleSVM
class EnsembleSVM:

    def __init__(self, n_estimators=50, max_samples=1000, max_features=2000,
                 n_randomized_search_iter=20):

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
        y_pred=self._predict_cover_proba(X)
        return [ [float(x)/100, 1-float(x)/100] for x in y_pred ]

    def _predict_cover_proba(self, X):
        X_val=self.selector.transform(X)
        y_val_pred=[0]*len(X_val)
        for clf in self.clf_list:
            P=clf.predict_proba(X_val)
            for i in range(len(P)):
                y_val_pred[i]+=P[i][0]
        return y_val_pred


    def score(self, X, y):
        y_pred=self._predict_cover_proba(X)
        ok=0
        for i in range(len(y)):
            p=float(y_pred[i])/len(self.clf_list)
            if  p > 0.5 and y[i]==0: ok+=1
            elif p <= 0.5 and y[i]==1: ok+=1

        return float(ok)/len(y)

   
# }}}

# {{{ Ensemble4Stego

#M_BIN="/usr/local/MATLAB/R2013a/bin/matlab -nodesktop -nojvm -nosplash -r"
M_BIN="octave -q --no-gui --eval"

class Ensemble4Stego:

    def fit(self, X, y):
        
        currdir=os.path.dirname(__file__)
        basedir=os.path.abspath(os.path.join(currdir, os.pardir))
        m_path=os.path.join(basedir, 'external', 'octave')

        self.__tmpdir=tempfile.mkdtemp()

        y=numpy.array(y)
        Xc=X[y==0]
        Xs=X[y==1]
        
        if len(Xc)>len(Xs):
            Xs=Xs[:len(Xc)]

        if len(Xs)>len(Xc):
            Xc=Xc[:len(Xs)]

        pcover=self.__tmpdir+"/F_train_cover.mat"
        savemat(pcover, mdict={'F': numpy.array(Xc)}, oned_as='column')

        pstego=self.__tmpdir+"/F_train_stego.mat"
        savemat(pstego, mdict={'F': numpy.array(Xs)}, oned_as='column')

        pclf=self.__tmpdir+"/clf.mat"
    
        m_code=""
        m_code+="cd "+self.__tmpdir+";"
        m_code+="addpath('"+m_path+"');"
        m_code+="warning('off');"
        m_code+="ensemble_fit('"+pcover+"', '"+pstego+"', '"+pclf+"');"
        m_code+="exit"
        p=subprocess.Popen(M_BIN+" \""+m_code+"\"", shell=True)
        output, err = p.communicate()
        status = p.wait()

        self.__mat_clf=loadmat(pclf)
        shutil.rmtree(self.__tmpdir)

    def predict_proba(self, X):

        currdir=os.path.dirname(__file__)
        basedir=os.path.abspath(os.path.join(currdir, os.pardir))
        m_path=os.path.join(basedir, 'external', 'octave')

        self.__tmpdir=tempfile.mkdtemp()

        prob=[]

        path=self.__tmpdir+"/F_test.mat"
        savemat(path, mdict={'F': numpy.array(X)}, oned_as='column')

        pclf=self.__tmpdir+"/clf.mat"
        savemat(pclf, self.__mat_clf)

        pvotes=self.__tmpdir+"/votes.txt"

        m_code=""
        m_code+="cd "+self.__tmpdir+";"
        m_code+="addpath('"+m_path+"');"
        m_code+="warning('off');"
        m_code+="ensemble_predict('"+pclf+"', '"+path+"', '"+pvotes+"');"
        m_code+="exit"
        p=subprocess.Popen(M_BIN+" \""+m_code+"\"", shell=True)
        output, err = p.communicate()
        status = p.wait()


        with open(pvotes, 'r') as f:
            lines=f.readlines()
        f.close()

        shutil.rmtree(self.__tmpdir)

        for l in lines:
            votes=(1+float(l)/500)/2
            prob.append( [1-votes, votes] )

        return prob


    def predict(self, X):
        results=[]
        proba=self.predict_proba(X)
        for p in proba:
            if p[0]>=0.5:
                results.append(0)
            else:
                results.append(1)
        return numpy.array(results)

    def score(self, X, y):
        Z=self.predict(X)
        result=numpy.count_nonzero(Z==y)
        return round(float(result)/len(y), 2)


    def save(self, path):
        savemat(path, self.__mat_clf)

    def load(self, path):
        self.__mat_clf=loadmat(path)

# }}}

