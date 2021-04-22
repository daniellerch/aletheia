
import numpy
import random
import os
import tempfile
import shutil
import subprocess
import glob
import sys

from aletheialib import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm

#import hdf5storage
from scipy.io import savemat, loadmat
from scipy import signal # ndimage
from imageio import imread

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count



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

# {{{ Tensorflow NN

import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn
import tensorflow.keras.layers as L
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks


class NN: 

    def __init__(self, network, model_name=None, shape=(512,512,3)):
        # {{{
        self.model_dir = 'models'
        self.model_name = model_name
        self.shape = shape
        if network == "effnetb0":
            self.model = self.create_model_effnetb0()
        else:
            print("NN __init__ Error: network not found")
            sys.exit(0)

        if model_name:
            path = self.model_dir+'/'+self.model_name+'-best.h5'
            if os.path.exists(path):
                print("Loading", path, "...")
                self.model.load_weights(path)

        self.replace_method = False

        if self.replace_method:
            print("WARNING! replace_method enabled")
            self.replace_base_str = "NSF5"
            self.replace_list = ['NSF5', 'JUNIW', 'STEGHIDE', 'OUTGUESS']

        # }}}

    def create_model_effnetb0(self):
        # {{{
        input_shape = self.shape

        model = tf.keras.Sequential([
            efn.EfficientNetB0(
                input_shape=input_shape,
                weights='imagenet',
                include_top=False
                ),
            L.GlobalAveragePooling2D(),
            L.Dense(2, activation='softmax')
            ])
        return model
        # }}}

    def rot_flip(self, I):
        # {{{
        rot = random.randint(0,3)
        if random.random() < 0.5:
            I = np.rot90(I, rot)
        else:
            I = np.flip(np.rot90(I, rot))
        return I
        # }}}

    def train_generator(self, cover_list, stego_list, batch):
        # {{{
        while True:
            bs = batch//2
            C, S, y = [], [], []
            while bs>0:
                try:
                    C_path = random.choice(cover_list)
                    S_path = random.choice(stego_list)

                    if self.replace_method:
                       S_path = S_path.replace(self.replace_base_str, 
                                               random.choice(self.replace_list))

                    Ic = self.rot_flip(imread(C_path))
                    Is = self.rot_flip(imread(S_path))
                    if Ic.shape!=self.shape or Is.shape!=self.shape:
                        print("WARNING: wrong shape", Is.shape)
                        continue
                    C.append(Ic)
                    y.append([1, 0])
                    S.append(Is)
                    y.append([0, 1])
                    bs -= 2
                except Exception as e:
                    print("NN train_generator Warning: cannot read image:", C_path, S_path)
                    print(e)
                    continue

            X = np.vstack((C,S)).astype('float32')/255
            y = np.hstack(([0]*len(C), [1]*len(S)))
            Y = to_categorical(y, 2)
            yield X, Y
        # }}}

    def valid_generator(self, cover_list, stego_list, batch):
        # {{{
        if len(cover_list)!=len(stego_list):
            print("NN valid_generator error: we expect same number of cover and stego images")
            sys.exit(0)
        if len(cover_list)*2 % batch != 0:
            print("NN valid_generator error: wrong batch size")
            sys.exit(0)

        C, S = [], []
        bs = batch//2
        while True:
            for i in range(len(cover_list)):
                if bs>0:
                    try:
                        C_path = cover_list[i]
                        S_path = stego_list[i]

                        if self.replace_method:
                           S_path = S_path.replace(self.replace_base_str, 
                                                   random.choice(self.replace_list))

                        Ic = imread(C_path)
                        Is = imread(S_path)
                        if Ic.shape!=self.shape or Is.shape!=self.shape:
                            print("NN valid_generator warning: wrong shape:", C_path, S_path)
                            continue
                        C.append(Ic)
                        S.append(Is)
                        bs -= 2
                    except KeyboardInterrupt:
                        sys.exit(0)
                    except:
                        print("NN valid_generator warning: cannot read image:", C_path, S_path, i)
                        continue
                else:
                    X = np.vstack((C,S)).astype('float32')/255
                    y = np.hstack(([0]*len(C), [1]*len(S)))
                    Y = to_categorical(y, 2)
                    yield X, Y
                    C, S = [], []
                    bs = batch//2
        # }}}

    def pred_generator(self, image_list, batch):
        # {{{
        images = []
        for f in image_list:
            try:
                img = imread(f)
                if img.shape!=self.shape:
                    print("NN pred_generator warning: wrong shape:", f)
                    continue
                images.append(img[:self.shape[0], :self.shape[1], :])
            except Exception as e:
                #print(e)
                print("NN pred_generator warning: cannot read image:", f)
                continue

            if len(images)==batch:
                X = np.array(images).astype('float32')/255
                yield X
                images = []

        if len(images)>0:
            X = np.array(images).astype('float32')/255
            yield X
        # }}}

    def train(self,
              trn_C_list, trn_S_list, trn_batch,
              val_C_list, val_S_list, val_batch,
              max_epochs, early_stopping):
        # {{{
        opt = optimizers.Adamax(lr=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        cb_checkpoint = callbacks.ModelCheckpoint(
            self.model_dir+"/"+self.model_name+'-{epoch:03d}-{accuracy:.4f}-{val_accuracy:.4f}.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )

        cb_checkpoint_best = callbacks.ModelCheckpoint(
            self.model_dir+"/"+self.model_name+'-best.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )

        cb_earlystopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            verbose=2,
            patience=early_stopping
        )

        callbacks_list = [
            cb_checkpoint,
            cb_checkpoint_best,
            cb_earlystopping
        ]

        steps_train = int((len(trn_C_list)+len(trn_S_list))/trn_batch)
        g_train = self.train_generator(trn_C_list, trn_S_list, trn_batch)
        steps_train = 1000 # XXX

        steps_valid = int((len(val_C_list)+len(val_S_list))/val_batch)
        g_valid = self.valid_generator(val_C_list, val_S_list, val_batch)

        self.model.fit(g_train, steps_per_epoch=steps_train,
                  validation_data=g_valid, validation_steps=steps_valid,
                  callbacks=callbacks_list, epochs=max_epochs)
        # }}}

    def filter_images(self, files):
        # {{{
        files_ok = []
        for f in files:
            img = imread(f)
            if len(img.shape)!=3 or img.shape[2] != self.shape[2]:
                print("WARNING: image ignored:", f, ", expected number of channels:", 
                       self.shape[2])
                continue

            if (img.shape[0] < self.shape[0] or 
                img.shape[1] < self.shape[1]):
                print("WARNING: image ignored:", f, ", image too small, expected:", 
                       self.shape[0], "x", self.shape[1])
                continue
            files_ok.append(f)
        return files_ok
        # }}}

    def load_model(self, model_path, quiet=False):
        # {{{
        if os.path.exists(model_path):
            if not quiet:
                print("Loading", model_path, "...")
            self.model.load_weights(model_path)
        # }}}

    def predict(self, files, batch):
        # {{{
        verbose = 1
        if len(files)<batch:
            batch=1
            verbose = 0
        steps = len(files)//batch
        #print("steps:", steps, "batch:", batch)
        #print("files:", files[:steps*batch])
        g = self.pred_generator(files[:steps*batch], batch)
        pred = self.model.predict(g, steps=steps, verbose=verbose)[:,-1]
        if steps*batch<len(files):
            g = self.pred_generator(files[steps*batch:], batch)
            pred = pred.tolist() + self.model.predict(g, steps=1, verbose=1)[:,-1].tolist()
        return np.array(pred)
        # }}}
 

# }}}


