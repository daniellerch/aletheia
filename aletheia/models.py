
import numpy
import random
import os
import tempfile
import shutil
import subprocess
import glob
import sys

from aletheia import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm

import hdf5storage
from scipy.io import savemat, loadmat
from scipy import misc, signal # ndimage

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count


from keras.models import Model
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Input, Conv2D, AveragePooling2D
from keras.layers import Lambda, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import np_utils



# {{{ EnsembleSVM
class EnsembleSVM:

    def __init__(self, n_estimators=50, max_samples=1000, max_features=2000,
                 n_randomized_search_iter=20, random_state=123):

        random.seed(random_state)
        self.random_state=random_state
        self.n_estimators=n_estimators
        self.max_samples=max_samples
        self.max_features=max_features
        self.n_randomized_search_iter=n_randomized_search_iter

    def _prepare_classifier(self, params, n_jobs=1):

        X_train, y_train = params

        tuned_parameters = [{
            'kernel': ['rbf'], 
            'gamma': [1e-4,1e-3,1e-2,1e-1,1e+0,1e+1,1e+2,1e+3,1e+4],
            'C': [1e+0,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6,1e+7,1e+8,1e+9]
        }]

        clf=RandomizedSearchCV(svm.SVC(random_state=self.random_state), 
                               tuned_parameters[0], 
                               n_iter=self.n_randomized_search_iter, 
                               n_jobs=n_jobs, random_state=self.random_state)
        clf.fit(X_train, y_train)
              
        params=clf.best_params_
        clf=svm.SVC(kernel=params['kernel'], C=params['C'], 
            gamma=params['gamma'], probability=True, 
            random_state=self.random_state)
        clf.fit(X_train, y_train)

        return clf


    def fit(self, X, y):
        
        self.selector = SelectKBest(f_classif, k=self.max_features)
        self.selector.fit(X, y)

        X_train=self.selector.transform(X)
        y_train=y

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

        """
        X2=[]
        for clf in self.clf_list:
            P=clf.predict_proba(X_train)
            if len(X2)==0:
                X2=P[:, 0]
            else:
                X2=numpy.vstack((X2, P[:, 0]))
        X2=numpy.swapaxes(X2, 0, 1)
        print "X2:", X2.shape

        from sklearn.ensemble import RandomForestClassifier
        self.clf2=RandomForestClassifier(n_estimators=100)
        self.clf2.fit(X2, y_train)
        """

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

        """
        X2=[]
        Xt=self.selector.transform(X)
        for clf in self.clf_list:
            P=clf.predict_proba(Xt)
            if len(X2)==0:
                X2=P[:, 0]
            else:
                X2=numpy.vstack((X2, P[:, 0]))
        X2=numpy.swapaxes(X2, 0, 1)
        print "X2 predict:", X2.shape

        return self.clf2.predict_proba(X2)[:,0]
        """

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
        os.chdir(m_path)

        self.__tmpdir=tempfile.mkdtemp()

        y=numpy.array(y)
        Xc=X[y==0]
        Xs=X[y==1]
        
        if len(Xc)>len(Xs):
            Xs=Xs[:len(Xc)]

        if len(Xs)>len(Xc):
            Xc=Xc[:len(Xs)]

        pcover=self.__tmpdir+"/F_train_cover.mat"
        #savemat(pcover, mdict={'F': numpy.array(Xc)}, oned_as='column')
        hdf5storage.write({u'F': numpy.array(Xc)}, '.', pcover, matlab_compatible=True)

        pstego=self.__tmpdir+"/F_train_stego.mat"
        #savemat(pstego, mdict={'F': numpy.array(Xs)}, oned_as='column')
        hdf5storage.write({u'F': numpy.array(Xs)}, '.', pstego, matlab_compatible=True)

        pclf=self.__tmpdir+"/clf.mat"
    
        del Xc
        del Xs
        del X

        m_code=""
        m_code+="cd "+self.__tmpdir+";"
        m_code+="addpath('"+m_path+"');"
        m_code+="warning('off');"
        m_code+="ensemble_fit('"+pcover+"', '"+pstego+"', '"+pclf+"');"
        m_code+="exit"
        p=subprocess.Popen(M_BIN+" \""+m_code+"\"", stdout=subprocess.PIPE, shell=True)
        # output, err = p.communicate()
        status = p.wait()

        self.__mat_clf=loadmat(pclf)
        shutil.rmtree(self.__tmpdir)

    def predict_proba(self, X):

        currdir=os.path.dirname(__file__)
        basedir=os.path.abspath(os.path.join(currdir, os.pardir))
        m_path=os.path.join(basedir, 'external', 'octave')
        os.chdir(m_path)

        self.__tmpdir=tempfile.mkdtemp()

        prob=[]

        path=self.__tmpdir+"/F_test.mat"
        #savemat(path, mdict={'F': numpy.array(X)}, oned_as='column')
        hdf5storage.write({u'F': numpy.array(X)}, '.', path, matlab_compatible=True)

        pclf=self.__tmpdir+"/clf.mat"
        savemat(pclf, self.__mat_clf)

        pvotes=self.__tmpdir+"/votes.txt"

        m_code=""
        m_code+="cd "+self.__tmpdir+";"
        m_code+="addpath('"+m_path+"');"
        m_code+="warning('off');"
        m_code+="ensemble_predict('"+pclf+"', '"+path+"', '"+pvotes+"');"
        m_code+="exit"
        p=subprocess.Popen(M_BIN+" \""+m_code+"\"", stdout=subprocess.PIPE, shell=True)
        #output, err = p.communicate()
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
        savemat(path, self.__mat_clf, appendmat=False)

    def load(self, path):
        self.__mat_clf=loadmat(path, appendmat=False)

# }}}

# {{{ SaveBestModelCallback()
g_best_accuracy=0

class SaveBestModelCallback(Callback):
    def __init__(self, data, model, name):
        self.data = data
        self.name = name
        self.model = model
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.data
        loss, acc = self.model.evaluate(x, y, verbose=0)

        global g_best_accuracy
        if acc>g_best_accuracy:
            g_best_accuracy=acc
            self.model.save_weights(self.name+"_"+str(round(acc,2))+".h5")
# }}}

# {{{ XuNet
class XuNet:

    def __init__(self):
        self.model=self._create_model(256)

    # {{{ _create_model()
    def _create_model(self, n):

        inputs = Input(shape=(1, n, n))
        x = inputs

        x = Conv2D(8, (5,5), padding="same", strides=1, data_format="channels_first")(x)
        x = BatchNormalization()(x)
        x = Lambda(K.abs)(x)
        x = Activation("tanh")(x)   
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")(x)
        print x

        x = Conv2D(16, (5,5), padding="same", data_format="channels_first")(x)
        x = BatchNormalization()(x)
        x = Activation("tanh")(x)   
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")(x)
        print x

        x = Conv2D(32, (1,1), padding="same", data_format="channels_first")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)   
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")(x)
        print x

        x = Conv2D(64, (1,1), padding="same", data_format="channels_first")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)   
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")(x)
        print x

        x = Conv2D(128, (1,1), padding="same", data_format="channels_first")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)   
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")(x)
        print x
     


        x = GlobalAveragePooling2D(data_format="channels_first")(x) 
        print x

        x = Dense(2)(x)
        x = Activation('softmax')(x)

        predictions = x

        model = Model(inputs=inputs, outputs=predictions)

        return model
    # }}}

    # {{{ _load_images()
    def _load_images(self, image_path):

        F0 = numpy.array(
           [[-1,  2,  -2,  2, -1],
            [ 2, -6,   8, -6,  2],
            [-2,  8, -12,  8, -2],
            [ 2, -6,   8, -6,  2],
            [-1,  2,  -2,  2, -1]])


        # Read filenames
        files=[]
        if os.path.isdir(image_path):
            for dirpath,_,filenames in os.walk(image_path):
                for f in filenames:
                    path=os.path.abspath(os.path.join(dirpath, f))
                    if not utils.is_valid_image(path):
                        print "Warning, please provide a valid image: ", f
                    else:
                        files.append(path)
        else:
            files=[image_path]

        files=sorted(files)

        X=[]
        for f in files:
            I = misc.imread(f)
            I=signal.convolve2d(I, F0, mode='same')  
            I=I.astype(numpy.int16)
            X.append( [ I ] )

        X=numpy.array(X)

        return X
    # }}}

    # {{{ train()
    def train(self, cover_path, stego_path, val_size=0.10, name='xu-net'):

        C = self._load_images(cover_path)
        S = self._load_images(stego_path)

        idx=range(len(C))
        random.shuffle(idx)
        C=C[idx]
        S=S[idx]

        l=int(len(C)*(1-val_size))
        Xc_train=C[:l]
        Xs_train=S[:l]
        Xc_val=C[l:]
        Xs_val=S[l:]

        X_train = numpy.vstack((Xc_train, Xs_train))
        y_train = numpy.hstack(([0]*len(Xc_train), [1]*len(Xs_train)))
        y_train = np_utils.to_categorical(y_train, 2)

        X_val = numpy.vstack((Xc_val, Xs_val))
        y_val = numpy.hstack(([0]*len(Xc_val), [1]*len(Xs_val)))
        y_val = np_utils.to_categorical(y_val, 2)


        self.model.compile(loss='binary_crossentropy', optimizer="adam", 
                           metrics=['accuracy'])

        self.model.fit(X_train, y_train, batch_size=32, epochs=1000, 
                       callbacks=[SaveBestModelCallback((X_val, y_val), 
                                                   self.model, name)],
                       validation_data=(X_val, y_val), shuffle=True)

    # }}}



# }}}

