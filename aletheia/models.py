
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

# {{{ TF-CNN



import glob
import time
import random
import threading

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


from scipy import misc
from functools import partial
from sklearn.metrics import accuracy_score

from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope



# {{{ AdamaxOptimizer()
# Implementation of Adamax optimizer, taken from : 
# https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

class AdamaxOptimizer(optimizer.Optimizer):
    """
    Optimizer that implements the Adamax algorithm. 
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)


    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
# }}}

# {{{ _average_summary()
class _average_summary:
    def __init__(self, variable, name, num_iterations):
        self.sum_variable = tf.get_variable(name, shape=[], \
                                initializer=tf.constant_initializer(0), \
                                dtype=variable.dtype.base_dtype, \
                                trainable=False, \
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])
        with tf.control_dependencies([variable]):
            self.increment_op = tf.assign_add(self.sum_variable, variable)
        self.mean_variable = self.sum_variable / float(num_iterations)
        self.summary = tf.summary.scalar(name, self.mean_variable)
        with tf.control_dependencies([self.summary]):
            self.reset_variable_op = tf.assign(self.sum_variable, 0)

    def add_summary(self, sess, writer, step):
        s, _ = sess.run([self.summary, self.reset_variable_op])
        writer.add_summary(s, step)
# }}}

# {{{ _train_data_generator()
def _train_data_generator(cover_files, stego_files, data_augm=False, 
                   shuffle=True, crop_size=256):

    cover_list = sorted(cover_files)
    stego_list = sorted(stego_files)
    nb_data = len(cover_list)

    if len(cover_list) != len(stego_list) or len(cover_list)==0:
        print("Error, check the number of files:", 
            len(cover_list), "!=", len(stego_list))
        sys.exit(0)

    img = misc.imread(cover_list[0])[:crop_size,:crop_size]
    batch = np.empty((2, img.shape[0], img.shape[1],1), dtype='uint8')
    iterable = list(zip(cover_list, stego_list))
    while True:
        if shuffle:
            random.shuffle(iterable)
        for cover_path, stego_path in iterable:
            labels = np.array([0, 1], dtype='uint8')
            batch[0,:,:,0] = misc.imread(cover_path)[:crop_size,:crop_size]
            batch[1,:,:,0] = misc.imread(stego_path)[:crop_size,:crop_size]

            if data_augm:
                rot = random.randint(0,3)
                if random.random() < 0.5:
                    yield [np.rot90(batch, rot, axes=[1,2]), 
                           np.array([0,1], dtype='uint8')]
                else:
                    yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), 
                        np.array([0,1], dtype='uint8')]

            yield [batch, labels]
# }}}        

# {{{ _test_data_generator()
def _test_data_generator(files, crop_size=256):

    nb_data = len(files)
    img = misc.imread(files[0])[:crop_size,:crop_size]
    batch = np.empty((1, img.shape[0], img.shape[1],1), dtype='uint8')
    while True:
        for path in files:
            labels = np.array([0], dtype='uint8')
            batch[0,:,:,0] = misc.imread(path)[:crop_size,:crop_size]
            yield [batch, labels]
# }}}        

# {{{ _GeneratorRunner()
class _GeneratorRunner():
    """
    This class manage a multithreaded queue filled with a generator
    """
    def __init__(self, generator, capacity):
        """
        inputs: generator feeding the data, must have thread_idx 
            as parameter (but the parameter may be not used)
        """
        self.generator = generator
        _input = generator().__next__()
        if type(_input) is not list:
            raise ValueError("generator doesn't return" \
                             "a list: %r" %  type(_input))
        input_batch_size = _input[0].shape[0]
        if not all(_input[i].shape[0] == input_batch_size for i in range(len(_input))):
            raise ValueError("all the inputs doesn't have the same batch size,"\
                             "the batch sizes are: %s" % [_input[i].shape[0] for i in range(len(_input))])
        self.data = []
        self.dtypes = []
        self.shapes = []
        for i in range(len(_input)):
            self.shapes.append(_input[i].shape[1:])
            self.dtypes.append(_input[i].dtype)
            self.data.append(tf.placeholder(dtype=self.dtypes[i], \
                                shape=(input_batch_size,) + self.shapes[i]))
        self.queue = tf.FIFOQueue(capacity, shapes=self.shapes, \
                                dtypes=self.dtypes)
        self.enqueue_op = self.queue.enqueue_many(self.data)
        self.close_queue_op = self.queue.close(cancel_pending_enqueues=False)
        
    def get_batched_inputs(self, batch_size):
        """
        Return tensors containing a batch of generated data
        """
        batch = self.queue.dequeue_many(batch_size)
        return batch
    
    def thread_main(self, sess, thread_idx=0, n_threads=1):
        try:
            #for data in self.generator(thread_idx, n_threads):
            for data in self.generator():
                sess.run(self.enqueue_op, feed_dict={i: d \
                            for i, d in zip(self.data, data)})
                if self.stop_threads:
                    return
        except RuntimeError:
            pass
        except tf.errors.CancelledError:
            pass
    
    def start_threads(self, sess, n_threads=1):
        self.stop_threads = False
        self.threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess, n, n_threads))
            t.daemon = True
            t.start()
            self.threads.append(t)
        return self.threads

    def stop_runner(self, sess):
        self.stop_threads = True
        sess.run(self.close_queue_op)

def queueSelection(runners, sel, batch_size):
    selection_queue = tf.FIFOQueue.from_list(sel, [r.queue for r in runners])
    return selection_queue.dequeue_many(batch_size)
# }}}

# {{{ _Model()
class _Model():

    def __init__(self, is_training=None):

        if tf.test.is_gpu_available():
            data_format='NCHW'
        else:
            data_format='NHWC'

        self.data_format = data_format                                           
        if is_training is None:                                                  
            self.is_training = tf.get_variable('is_training', dtype=tf.bool, 
                                    initializer=tf.constant_initializer(True),
                                    trainable=False) 
        else:                                                                    
            self.is_training = is_training

    def _build_losses(self, labels):
        self.labels = tf.cast(labels, tf.int64)                                  
        with tf.variable_scope('loss'):                                          
            oh = tf.one_hot(self.labels, 2)                                      
            xen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(  
                                                    labels=oh,logits=self.outputs))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)   
            self.loss = tf.add_n([xen_loss] + reg_losses)                        
        with tf.variable_scope('accuracy'):                                      
            am = tf.argmax(self.outputs, 1)                                      
            equal = tf.equal(am, self.labels)                                    
            self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))           
        return self.loss, self.accuracy         
# }}}

# {{{ SRNet()
class SRNet(_Model):

    def _build_model(self, inputs):
        self.inputs = inputs
        if self.data_format == 'NCHW':
            reduction_axis = [2,3]
            _inputs = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
        else:
            reduction_axis = [1,2]
            _inputs = tf.cast(inputs, tf.float32)
        with arg_scope([layers.conv2d], num_outputs=16,
                       kernel_size=3, stride=1, padding='SAME',
                       data_format=self.data_format,
                       activation_fn=None,
                       weights_initializer=layers.variance_scaling_initializer(),
                       weights_regularizer=layers.l2_regularizer(2e-4),
                       biases_initializer=tf.constant_initializer(0.2),
                       biases_regularizer=None),\
            arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True, 
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format),\
            arg_scope([layers.avg_pool2d],
                       kernel_size=[3,3], stride=[2,2], padding='SAME',
                       data_format=self.data_format):
            with tf.variable_scope('Layer1'): 
                conv=layers.conv2d(_inputs, num_outputs=64, kernel_size=3)
                actv=tf.nn.relu(layers.batch_norm(conv))
            with tf.variable_scope('Layer2'): 
                conv=layers.conv2d(actv)
                actv=tf.nn.relu(layers.batch_norm(conv))
            with tf.variable_scope('Layer3'): 
                conv1=layers.conv2d(actv)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn2=layers.batch_norm(conv2)
                res= tf.add(actv, bn2)
            with tf.variable_scope('Layer4'): 
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn2=layers.batch_norm(conv2)
                res= tf.add(res, bn2)
            with tf.variable_scope('Layer5'): 
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                res= tf.add(res, bn)
            with tf.variable_scope('Layer6'): 
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                res= tf.add(res, bn)
            with tf.variable_scope('Layer7'): 
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                res= tf.add(res, bn)
            with tf.variable_scope('Layer8'): 
                convs = layers.conv2d(res, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer9'):  
                convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res, num_outputs=64)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=64)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer10'): 
                convs = layers.conv2d(res, num_outputs=128, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res, num_outputs=128)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=128)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer11'): 
                convs = layers.conv2d(res, num_outputs=256, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1=layers.conv2d(res, num_outputs=256)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=256)
                bn=layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res= tf.add(convs, pool)
            with tf.variable_scope('Layer12'): 
                conv1=layers.conv2d(res, num_outputs=512)
                actv1=tf.nn.relu(layers.batch_norm(conv1))
                conv2=layers.conv2d(actv1, num_outputs=512)
                bn=layers.batch_norm(conv2)
                avgp = tf.reduce_mean(bn, reduction_axis,  keepdims=True )
        ip=layers.fully_connected(layers.flatten(avgp), num_outputs=2,
                    activation_fn=None, normalizer_fn=None,
                    weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                    biases_initializer=tf.constant_initializer(0.), scope='ip')
        self.outputs = ip
        return self.outputs
# }}}

# {{{ nn_configure_device()
def nn_configure_device(dev_id):
    if dev_id == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"]="";
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
        os.environ["CUDA_VISIBLE_DEVICES"]=dev_id;
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# }}}

# {{{ nn_fit()
def nn_fit(model_class, data, checkpoint_name,
              batch_size=32, load_checkpoint=None, valid_interval=100,
              optimizer=AdamaxOptimizer(0.0001), 
              log_path='log', checkpoint_path='checkpoint',
              max_iter=1000000, num_runner_threads=10, early_stopping=100):

    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    if not os.path.isdir(checkpoint_path+'/'+checkpoint_name):
        os.mkdir(checkpoint_path+'/'+checkpoint_name)
        if load_checkpoint != None:
            print("Checkpoint file does not exist. Creating a new one!")
        load_checkpoint = None

    train_cover_files, train_stego_files, \
    valid_cover_files, valid_stego_files = data
    train_ds_size = len(train_cover_files)+len(train_stego_files)
    valid_ds_size = len(valid_cover_files)+len(valid_stego_files)

    train_gen = partial(_train_data_generator, 
                        train_cover_files, train_stego_files, True)
    valid_gen = partial(_train_data_generator, 
                        valid_cover_files, valid_stego_files, False)

    tf.reset_default_graph()
    train_runner = _GeneratorRunner(train_gen, batch_size * 10)
    valid_runner = _GeneratorRunner(valid_gen, batch_size * 10)


    is_training = tf.get_variable('is_training', dtype=tf.bool, 
                                  initializer=True, trainable=False)

    tf_batch_size = tf.get_variable('batch_size', dtype=tf.int32, 
                                    initializer=batch_size, trainable=False, 
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES])

    disable_training_op = tf.group(tf.assign(is_training, False), 
                                   tf.assign(tf_batch_size, batch_size))

    enable_training_op = tf.group(tf.assign(is_training, True), 
                            tf.assign(tf_batch_size, batch_size))

    img_batch, label_batch = queueSelection([valid_runner, train_runner], 
                                             tf.cast(is_training, tf.int32), 
                                             batch_size)

    model = model_class(is_training)
    model._build_model(img_batch)

    loss, accuracy = model._build_losses(label_batch)
    train_loss_s = _average_summary(loss, 'train_loss', valid_interval)
    train_accuracy_s = _average_summary(accuracy, 'train_accuracy', 
                                       valid_interval)

    valid_loss_s = _average_summary(loss, 'valid_loss', 
                                   float(valid_ds_size) / float(batch_size))

    valid_accuracy_s = _average_summary(accuracy, 'valid_accuracy', 
                                       float(valid_ds_size) / float(batch_size))

    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], 
                                  initializer=tf.constant_initializer(0), 
                                  trainable=False)

    minimize_op = optimizer.minimize(loss, global_step)

    train_op = tf.group(minimize_op, train_loss_s.increment_op, 
                        train_accuracy_s.increment_op)

    increment_valid = tf.group(valid_loss_s.increment_op, 
                               valid_accuracy_s.increment_op)

    init_op = tf.group(tf.global_variables_initializer(), 
                       tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=10000)

    with tf.Session() as sess:
        sess.run(init_op)
        if load_checkpoint is not None:
            checkpoint_file = checkpoint_path+'/'+load_checkpoint+'/model.ckpt'
            print("Loading checkpoint", checkpoint_file, "...")
            saver.restore(sess, checkpoint_file)

        train_runner.start_threads(sess, num_runner_threads)
        valid_runner.start_threads(sess, 1)
        writer = tf.summary.FileWriter(log_path, sess.graph)
        start = sess.run(global_step)
        sess.run(disable_training_op)
        sess.run([valid_loss_s.reset_variable_op, 
                  valid_accuracy_s.reset_variable_op, 
                  train_loss_s.reset_variable_op, 
                  train_accuracy_s.reset_variable_op])

        _time = time.time()
        for j in range(0, valid_ds_size, batch_size):
            sess.run([increment_valid])
        _acc_val = sess.run(valid_accuracy_s.mean_variable)
        valid_accuracy_s.add_summary(sess, writer, start)
        valid_loss_s.add_summary(sess, writer, start)
        sess.run(enable_training_op)

        early_stopping_cnt = early_stopping
        best_acc = 0.0
        last_val_time = time.time()
        for i in range(start+1, max_iter+1):
            sess.run(train_op)
            if i % valid_interval == 0:
                # train
                train_acc = round(sess.run(train_accuracy_s.mean_variable), 4)
                train_loss_s.add_summary(sess, writer, i)
                train_accuracy_s.add_summary(sess, writer, i)

                # validation
                sess.run(disable_training_op)
                for j in range(0, valid_ds_size, batch_size):
                    sess.run([increment_valid])
                valid_acc = round(sess.run(valid_accuracy_s.mean_variable), 4)
                valid_loss_s.add_summary(sess, writer, i)
                valid_accuracy_s.add_summary(sess, writer, i)
                sess.run(enable_training_op)

                # log & checkpoint
                t = round(time.time()-last_val_time)
                print(i, early_stopping_cnt, "Accuracy:", train_acc, valid_acc, " : ", t, "seconds")
                last_val_time = time.time()
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    saver.save(sess, checkpoint_path+'/'+checkpoint_name+'/model_'+
                               str(round(valid_acc,4))+'_'+str(i)+'.ckpt')
                    saver.save(sess, checkpoint_path+'/'+checkpoint_name+'/model.ckpt')
                    early_stopping_cnt = early_stopping

                # Early stopping
                if early_stopping_cnt == 0:
                    print("Early stopping condition!")
                    print(i, "Best accuracy:", best_acc, " : ", t, "seconds")
                    return
                early_stopping_cnt -= 1



# }}}

# {{{ nn_predict()
def nn_predict(model_class, files, checkpoint_dir, batch_size=32):

    test_ds_size = len(files)
    gen = partial(_test_data_generator, files)

    tf.reset_default_graph()
    runner = _GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False)

    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = _average_summary(loss, 'loss',  
                                   float(test_ds_size) / float(batch_size))
    accuracy_summary = _average_summary(accuracy, 'accuracy',  
                                   float(test_ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op, 
                            accuracy_summary.increment_op)
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], 
                                  initializer=tf.constant_initializer(0), 
                                  trainable=False)
    init_op = tf.group(tf.global_variables_initializer(), 
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)

    outputs_arr = np.empty([test_ds_size, 
                            model.outputs.get_shape().as_list()[1]])
    checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, checkpoint_file)
        runner.start_threads(sess, 1)
        for j in range(0, test_ds_size, batch_size):
            outputs_arr[j:j+batch_size] = sess.run(model.outputs)
    pred = np.argmax(outputs_arr, axis=1)

    return pred
# }}}


# }}}


