
import numpy
import random
import os
import tempfile
import shutil
import subprocess
import glob
import sys

from aletheialib import utils

from scipy.io import savemat, loadmat
from scipy import signal # ndimage
from imageio import imread

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count



# {{{ Ensemble4Stego

#M_BIN="/usr/local/MATLAB/R2013a/bin/matlab -nodesktop -nojvm -nosplash -r"
M_BIN="octave -q --no-gui --eval"

class Ensemble4Stego:

    def fit(self, X, y):

        import hdf5storage

        currdir=os.path.dirname(__file__)
        basedir=os.path.abspath(os.path.join(currdir, os.pardir))        
        m_path=os.path.join(basedir, 'aletheia-cache', 'octave')
        
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

        import hdf5storage

        currdir=os.path.dirname(__file__)
        basedir=os.path.abspath(os.path.join(currdir, os.pardir))      
        m_path=os.path.join(basedir, 'aletheia-cache', 'octave')
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


# {{{ AccumulatingModel
import tensorflow as tf

class AccumulatingModel(tf.keras.Model):
    def __init__(self, accum_steps=2, **kwargs):
        super().__init__(**kwargs)
        self.accum_steps = tf.constant(accum_steps, dtype=tf.int64)
        self._accum_step_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._grad_accums = None

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss_real = self.compute_loss(x, y, y_pred, training=True)
            loss = loss_real / tf.cast(self.accum_steps, loss_real.dtype)

        grads = tape.gradient(loss, self.trainable_variables)

        if self._grad_accums is None:
            self._grad_accums = [
                tf.Variable(tf.zeros_like(v), trainable=False)
                for v in self.trainable_variables
            ]

        for ga, g in zip(self._grad_accums, grads):
            if g is not None:
                ga.assign_add(g)

        self._accum_step_counter.assign_add(1)

        def _apply():
            grads_and_vars = [(ga, v) for ga, v in zip(self._grad_accums, self.trainable_variables)]
            self.optimizer.apply_gradients(grads_and_vars)

            for ga in self._grad_accums:
                ga.assign(tf.zeros_like(ga))
            self._accum_step_counter.assign(0)
            return tf.constant(0)

        def _no_apply():
            return tf.constant(0)

        tf.cond(tf.equal(self._accum_step_counter, self.accum_steps), _apply, _no_apply)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss_real)
            else:
                metric.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}
        logs.setdefault("loss", loss_real)
        return logs


# }}}


class NN:

    def __init__(self, network, model_name=None, shape=(512,512,3)):
        # {{{
        self.model_dir = 'aletheia-models'
        self.model_name = model_name
        self.shape = shape
        self.network = network

        self.acc_grad = 1
        self.div255 = True
        self.subset = None
        self.opt = optimizers.Adamax(learning_rate=1e-3) 
        self.steps_train_limit = None
        self.use_pairs = False
        self.use_pairs_prob = 0.8
    
        if network == "effnetb0":
            self.model = self.create_model_effnetb0()

        elif network == "srnet":
            self.model = self.create_model_srnet()
            self.opt = optimizers.Adamax(learning_rate=1e-3) 
            self.acc_grad = 1
            self.use_pairs = False
            self.use_pairs_prob = 0.8
            self.subset = None
            self.steps_train_limit = 1000

        else:
            print("NN __init__ Error: network not found")
            sys.exit(0)


        if self.acc_grad>1:
            self.model = AccumulatingModel(accum_steps=acc_grad, inputs=self.model.inputs, outputs=self.model.outputs)

        if model_name:
            path_h5 = self.model_dir+'/'+self.model_name+'-best.h5'
            path = self.model_dir+'/'+self.model_name+'-best.keras'
            if os.path.exists(path):
                print("Loading", path, "...")
                self.model.load_weights(path)
            elif os.path.exists(path_h5): 
                print("Loading", path_h5, "...")
                self.model.load_weights(path_h5)
            else:
                print("New model:", path)

        self.replace_method = False

        if self.replace_method:
            print("WARNING! replace_method enabled")
            self.replace_base_str = "NSF5"
            self.replace_list = ['NSF5', 'JUNIW', 'STEGHIDE', 'OUTGUESS']

        # }}}

    def create_model_effnetb0(self):
        # {{{
        print("-- EFFNET --")
        input_shape = self.shape

        from tensorflow.keras import layers as L, regularizers

        tf.config.optimizer.set_jit(False) 
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')


        model = tf.keras.Sequential([
            efn.EfficientNetB0(
                input_shape=input_shape,
                weights='imagenet',
                include_top=False
                ),
            L.GlobalAveragePooling2D(),

            L.Dense(2, activation='softmax', dtype="float32")
            #L.Dense(1, activation="sigmoid", dtype="float32", name="out")
            ])
        return model
        # }}}

    def create_model_srnet(self):
        # {{{
        print("-- SRNET --")

        #tf.config.optimizer.set_jit(False)
        #from tensorflow.keras import mixed_precision
        #mixed_precision.set_global_policy('mixed_float16')

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import add, Dense, Dropout, Activation, Input, BatchNormalization
        from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
        from tensorflow.keras import optimizers
        from tensorflow.keras import initializers
        from tensorflow.keras import regularizers                                                                  

        # Deep Residual Network for Steganalysis of Digital Images. M. Boroumand, 
        # M. Chen, J. Fridrich. https://ws2.binghamton.edu/fridrich/Research/SRNet.pdf

        input_shape = self.shape


        inputs = Input(shape=input_shape)
        x = inputs

        conv2d_params = {
            'padding': 'same',
            'data_format': 'channels_last',
            'bias_initializer': initializers.Constant(0.2),
            'bias_regularizer': None,
            'kernel_initializer': initializers.VarianceScaling(),
            'kernel_regularizer': regularizers.l2(2e-4),
        }

        avgpool_params = {
            'padding': 'same',
            'data_format': 'channels_last',
            'pool_size': (3,3),
            'strides': (2,2)
        }

        bn_params = {
            'momentum': 0.9,
            'center': True,
            'scale': True
        }


        x = Conv2D(64, (3,3), strides=1, **conv2d_params)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation("relu")(x)

        x = Conv2D(16, (3,3), strides=1, **conv2d_params)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation("relu")(x)

        for i in range(5):
            y = x
            x = Conv2D(16, (3,3), **conv2d_params)(x)
            x = BatchNormalization(**bn_params)(x)
            x = Activation("relu")(x)
            x = Conv2D(16, (3,3), **conv2d_params)(x)
            x = BatchNormalization(**bn_params)(x)
            x = add([x, y])
            y = x


        for f in [16, 64, 128, 256]:
            y = Conv2D(f, (1,1), strides=2, **conv2d_params)(x)
            y = BatchNormalization(**bn_params)(y)
            x = Conv2D(f, (3,3), **conv2d_params)(x)
            x = BatchNormalization(**bn_params)(x)
            x = Activation("relu")(x)
            x = Conv2D(f, (3,3), **conv2d_params)(x)
            x = BatchNormalization(**bn_params)(x)
            x = AveragePooling2D(**avgpool_params)(x)
            x = add([x, y])

        x = Conv2D(512, (3,3), **conv2d_params)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation("relu")(x)
        x = Conv2D(512, (3,3), **conv2d_params)(x)
        x = BatchNormalization(**bn_params)(x)

        x = GlobalAveragePooling2D()(x)


        x = Dense(2, 
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.01)
        )(x)

        x = Activation('softmax')(x)

        predictions = x

        model = Model(inputs=inputs, outputs=predictions)




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

    def rot_flip_pair(self, I1, I2):
        # {{{
        rot = random.randint(0,3)
        if random.random() < 0.5:
            I1 = np.rot90(I1, rot)
            I2 = np.rot90(I2, rot)
        else:
            I1 = np.flip(np.rot90(I1, rot))
            I2 = np.flip(np.rot90(I2, rot))
        return I1, I2
        # }}}

    def train_generator(self, cover_list, stego_list, batch):
        # {{{
        while True:
            bs = batch//2
            C, S = [], []

            while bs>0:
                try:
                    C_path = random.choice(cover_list)
                    S_path = random.choice(stego_list)
                    if self.use_pairs and random.random() < self.use_pairs_prob:
                        basename = os.path.basename(C_path)
                        dirname = os.path.dirname(S_path)
                        S_path = os.path.join(dirname, basename)
                        Ic, Is = self.rot_flip_pair(imread(C_path), imread(S_path))
                    else:
                        if self.replace_method:
                           S_path = S_path.replace(self.replace_base_str,
                                                   random.choice(self.replace_list))
                        Ic = self.rot_flip(imread(C_path))
                        Is = self.rot_flip(imread(S_path))

                    if Ic.shape!=self.shape or Is.shape!=self.shape:
                        #print("WARNING: wrong shape", Is.shape)
                        continue
                    C.append(Ic)
                    S.append(Is)
                    bs -= 1
                except Exception as e:
                    #print("NN train_generator Warning: cannot read image:", C_path, S_path)
                    print(e)
                    continue

            if self.div255:
                X = np.vstack((C,S)).astype('float32')/255
            else:
                X = np.vstack((C,S)).astype('float32')

            y = np.hstack(([0]*len(C), [1]*len(S)))

            if self.use_pairs:
                # Break cover–stego pairing at batch level to avoid 
                # shortcut learning and early overfitting.
                p = np.random.permutation(len(y))
                X = X[p]
                y = y[p]

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
                            #print("NN valid_generator warning: wrong shape:", C_path, S_path)
                            continue
                        C.append(Ic)
                        S.append(Is)
                        bs -= 1
                    except KeyboardInterrupt:
                        sys.exit(0)
                    except:
                        #print("NN valid_generator warning: cannot read image:", C_path, S_path, i)
                        continue
                else:
                    if self.div255:
                        X = np.vstack((C,S)).astype('float32')/255
                    else:
                        X = np.vstack((C,S)).astype('float32')
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
            img = np.zeros(self.shape)
            try:
                I = imread(f)

                # This function must support images with variable size
                # Note that with big images we are only analyzing a small part
                d0 = min(I.shape[0], self.shape[0])
                d1 = min(I.shape[1], self.shape[1])
                d2 = min(I.shape[2], self.shape[2])
                img[:d0, :d1, :d2] = I[:d0, :d1, :d2]
                #img = self.rot_flip(img) # XXX

            except Exception as e:
                print(str(e))
                print("NN pred_generator warning: cannot read image:", f)

            images.append(img)

            if len(images)==batch:
                if self.div255:
                    X = np.array(images).astype('float32')/255
                else:
                    X = np.array(images).astype('float32')
                yield (X,)
                images = []

        if len(images)>0:
            if self.div255:
                X = np.array(images).astype('float32')/255
            else:
                X = np.array(images).astype('float32')
            yield (X,)
        # }}}

    def train(self,
              trn_C_list, trn_S_list, trn_batch,
              val_C_list, val_S_list, val_batch,
              max_epochs, early_stopping):
        # {{{
      
        if self.subset != None:
            trn_C_list = trn_C_list[:self.subset]
            trn_S_list = trn_S_list[:self.subset]

        #opt = optimizers.Adamax(learning_rate=0.0001) # XXX
        self.model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        #self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])

        cb_checkpoint = callbacks.ModelCheckpoint(
            self.model_dir+"/"+self.model_name+'-{epoch:03d}-{accuracy:.4f}-{val_accuracy:.4f}.keras',
            #self.model_dir+"/"+self.model_name+'-{epoch:03d}-{loss:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.keras',
            save_best_only=True,
            monitor='val_accuracy',
            #monitor='val_loss',
            mode='max'
            #mode='min',
        )

        cb_checkpoint_best = callbacks.ModelCheckpoint(
            self.model_dir+"/"+self.model_name+'-best.keras',
            save_best_only=True,
            monitor='val_accuracy',
            #monitor='val_loss',
            mode='max'
            #mode='min',
        )

        cb_checkpoint_last = callbacks.ModelCheckpoint(
            self.model_dir + "/" + self.model_name + "-last.keras",
            save_best_only=False,   # guarda en cada epoch
            save_weights_only=False,
            verbose=0
        )


        cb_earlystopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            #monitor='val_loss',
            mode='max',
            #mode='min',
            verbose=2,
            patience=early_stopping
        )


        callbacks_list = [
            cb_checkpoint,
            cb_checkpoint_best,
            cb_checkpoint_last,
            cb_earlystopping
        ]

        #callbacks_list = [] # XXX

        steps_train = int((len(trn_C_list)+len(trn_S_list))/trn_batch)
        g_train = self.train_generator(trn_C_list, trn_S_list, trn_batch)
        steps_train = (len(trn_C_list)+len(trn_S_list))//trn_batch
        steps_train -= steps_train%2

        if self.steps_train_limit:
            steps_train = self.steps_train_limit

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
            try:
                img = imread(f)
            except:
                print("WARNING: cannot read, image ignored:", f)
                continue

            if len(img.shape)!=3 or img.shape[2] != self.shape[2]:
                print("WARNING: image ignored:", f, ", expected number of channels:",
                       self.shape[2])
                continue

            """
            if (img.shape[0] < self.shape[0] or
                img.shape[1] < self.shape[1]):
                print("WARNING: image ignored:", f, ", image too small, expected:",
                       self.shape[0], "x", self.shape[1])
                continue
            """
            files_ok.append(f)
        return files_ok
        # }}}

    def load_model(self, model_path, quiet=False):
        # {{{
        if os.path.exists(model_path):
            if not quiet:
                print("Loading", model_path, "...")
            self.model.load_weights(model_path)
        elif not quiet:
            print("WARNING: model file not found:", model_path)
        # }}}

    def predict(self, files, batch, verbose=None):
        # {{{
        verb = 1
        if len(files)<batch:
            batch=1
            verb = 0
        if verbose != None:
            verb = verbose
        steps = len(files)//batch
        #print("steps:", steps, "batch:", batch)
        #print("files:", files[:steps*batch])
        g = self.pred_generator(files[:steps*batch], batch)
        pred = self.model.predict(g, steps=steps, verbose=verb)[:,-1]
        if steps*batch<len(files):
            g = self.pred_generator(files[steps*batch:], batch)
            pred = pred.tolist() + self.model.predict(g, steps=1, verbose=verb)[:,-1].tolist()
        return np.array(pred)
        # }}}

    def get_gradients_from_array(self, arr):
        # {{{
        batch = len(arr)
        targets = [[1,0]] * batch
        labels = tf.reshape(targets, (batch, 2))

        images = tf.cast(arr, tf.float32)/255
        loss_object = tf.keras.losses.BinaryCrossentropy()
        with tf.GradientTape() as tape:
            tape.watch(images)
            prediction = self.model(images, training=False)
            loss = loss_object(labels, prediction)
        gradient = tape.gradient(loss, images).numpy()

        return gradient
        # }}}



    def get_gradients(self, files, batch):
        # {{{
        if len(files)<batch:
            batch = 1

        targets = [[1,0]] * batch
        labels = tf.reshape(targets, (batch, 2))

        steps = len(files)//batch
        g1 = self.pred_generator(files[:steps*batch], batch)

        g2 = []
        if steps*batch<len(files):
            g2 = self.pred_generator(files[steps*batch:], 1)

        cnt = 0
        I = imread(files[0])
        gradients = np.zeros((len(files),)+I.shape).astype('float32') 
        for g in [g1, g2]:
            for images_batch in g:
                images_batch = tf.cast(images_batch, tf.float32)
                loss_object = tf.keras.losses.BinaryCrossentropy()
                with tf.GradientTape() as tape:
                    tape.watch(images_batch)
                    prediction = self.model(images_batch, training=False)
                    loss = loss_object(labels, prediction)
                gradient = tape.gradient(loss, images_batch).numpy()
                """
                if gradients is None:
                    gradients = gradient
                else:
                    gradients = np.concatenate((gradients, gradient), axis=0)
                """
                gradients[cnt:cnt+gradient.shape[0]] = gradient
                print("gradients:", cnt, "                \r", end='')
                cnt += gradient.shape[0]

        return gradients
        # }}}



# }}}

# {{{ UTILS

def load_model(nn, model_name):
    # {{{
    # Get the directory where the models are installed
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, os.pardir, 'aletheia-models')

    model_path = os.path.join(dir_path, model_name+".keras")
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}\n")
        sys.exit(-1)
    nn.load_model(model_path, quiet=True)
    # }}}
    return nn


def dci_si_method(image_path, method, modelA=None, modelB=None):
    # {{{
    import shutil
    import tempfile
    import aletheialib.stegosim
    import aletheialib.models
    import numpy as np
    from imageio import imread

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fn_sim=aletheialib.stegosim.embedding_fn(method)
    method = method.replace("-sim", "")

    embed_fn_saving = False
    if method in aletheialib.stegosim.EMB_FN_SAVING_METHODS:
        embed_fn_saving = True

    # Make some replacements to adapt the name of the method with the name
    # of the model file
    method = method.replace("-color", "")
    method = method.replace("j-uniward", "juniw")
    method = method.replace("s-uniward", "uniw")


    A_dir=tempfile.mkdtemp()
    shutil.copy(image_path, A_dir)

    B_dir=tempfile.mkdtemp()
    aletheialib.stegosim.embed_message(fn_sim, A_dir, "0.40", B_dir, 
                                       embed_fn_saving=embed_fn_saving, show_debug_info=False)

    A_nn = aletheialib.models.NN("effnetb0")
    B_nn = aletheialib.models.NN("effnetb0")
    A_files = glob.glob(os.path.join(A_dir, '*'))
    B_files = glob.glob(os.path.join(B_dir, '*'))

    if len(B_files)==0:
        print("ERROR: Cannot prepare the B set", image_path, B_files, method, "embed_fn_saving=", embed_fn_saving)
        sys.exit(0)

    
    if modelA!=None and modelB!=None:
        A_nn.load_model(modelA, quiet=True)
        B_nn.load_model(modelB, quiet=True)
        
    else:
        A_nn = load_model(A_nn, "effnetb0-A-alaska2-"+method)
        B_nn = load_model(B_nn, "effnetb0-B-alaska2-"+method)

    A = imread(A_files[0])
    B = imread(B_files[0])
    # This function must support images with variable size
    # Note that with big images we are only analyzing a small part
    d0 = min(A.shape[0], A_nn.shape[0])
    d1 = min(A.shape[1], A_nn.shape[1])
    d2 = min(A.shape[2], A_nn.shape[2])
    Ai = np.zeros(A_nn.shape)
    Bi = np.zeros(A_nn.shape)
    Ai[:d0, :d1, :d2] = A[:d0, :d1, :d2]
    Bi[:d0, :d1, :d2] = B[:d0, :d1, :d2]

    aa = []
    ab = []
    bb = []
    ba = []
    for flip in [0, 1]:
        for rot in [0, 1, 2, 3]:
            for roll in [0, 128, 256, 384]:

                if flip:
                    Ai = np.flip(Ai)
                    Bi = np.flip(Bi)
                if rot:
                    Ai = np.rot90(Ai, rot)
                    Bi = np.rot90(Bi, rot)
                if roll:
                    Ai = np.roll(Ai, roll, axis=1)
                    Bi = np.roll(Bi, roll, axis=1)

                Ax = np.array([Ai]).astype('float32')/255
                Bx = np.array([Bi]).astype('float32')/255

                p_aa = A_nn.model.predict(Ax, verbose=0)[0,1]
                p_ab = A_nn.model.predict(Bx, verbose=0)[0,1]
                p_bb = B_nn.model.predict(Bx, verbose=0)[0,1]
                p_ba = B_nn.model.predict(Ax, verbose=0)[0,1]
                aa.append(p_aa)
                ab.append(p_ab)
                bb.append(p_bb)
                ba.append(p_ba)

    aa_mean = np.mean(aa)
    aa = np.array(np.round(aa)).astype('uint8')
    ab = np.array(np.round(ab)).astype('uint8')
    bb = np.array(np.round(bb)).astype('uint8')
    ba = np.array(np.round(ba)).astype('uint8')

    inc = ( (aa!=bb) | (ba!=0) | (ab!=1) ).astype('uint8') 
    dci_pred_score = round(1-float(np.sum(inc==1))/(2*len(aa)),3)


    shutil.rmtree(A_dir)
    shutil.rmtree(B_dir)

    return dci_pred_score, aa_mean
    # }}}


# }}}

