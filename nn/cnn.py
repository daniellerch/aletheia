#!/usr/bin/python
# -*- coding: utf-8 -*-



import sys
import glob
from scipy import misc, ndimage, signal
import numpy
import random
import ntpath
import os
from skimage.util.shape import view_as_blocks, view_as_windows

from keras.layers import Merge, Lambda, Layer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.layers.core import Reshape
from keras import optimizers
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import backend as K

# Crop squared blocks of the image with size:
n=512

F0 = numpy.array(
   [[-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1]])


# {{{ create_cnn_model()
def create_model():

    model = Sequential()

    F = numpy.reshape(F0, (F0.shape[0],F0.shape[1],1,1) )
    bias=numpy.array([0])


    model.add(Conv2D(1, (5,5), padding="same", data_format="channels_first", 
                     input_shape=(1,n,n), activation='relu', weights=[F, bias]))


# {{{ Group 1
    model.add(Conv2D(8, (5,5), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Lambda(K.abs)) 
    model.add(Activation("tanh"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")) 
# }}}

# {{{ Group 2
    model.add(Conv2D(16, (5,5), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")) 
# }}}

# {{{ Group 3
    model.add(Conv2D(32, (1,1), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")) 
# }}}

# {{{ Group 4
    model.add(Conv2D(64, (1,1), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")) 
# }}}

# {{{ Group 5
    model.add(Conv2D(128, (1,1), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(GlobalAveragePooling2D(data_format="channels_first")) 
# }}}

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model
# }}}

# {{{ load_images()
def load_images(path_pattern):

    files=glob.glob(path_pattern)

    X=[]
    for f in files:
        I = misc.imread(f)
        patches = view_as_blocks(I, (n, n))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append( [ patches[i,j] ] )


    X=numpy.array(X)

    return X
# }}}

# {{{ train()
def train(tr_cover_dir, tr_stego_dir, ts_cover_dir, ts_stego_dir, model_file):

    Xc = load_images(tr_cover_dir+'/*.pgm')
    Xs = load_images(tr_stego_dir+'/*.pgm')
    Yc = load_images(ts_cover_dir+'/*.pgm')
    Ys = load_images(ts_stego_dir+'/*.pgm')

    X = numpy.vstack((Xc, Xs))
    Y = numpy.vstack((Yc, Ys))


    Xt = numpy.hstack(([0]*len(Xc), [1]*len(Xs)))
    Yt = numpy.hstack(([0]*len(Yc), [1]*len(Ys)))


    Xt = np_utils.to_categorical(Xt, 2)
    Yt = np_utils.to_categorical(Yt, 2)

    idx=list(range(len(X)))
    random.shuffle(idx)

    X=X[idx]
    Xt=Xt[idx]


    model = create_model()
    model.fit(X, Xt, batch_size=64, epochs=100, validation_data=(Y, Yt), shuffle=True)
    model.save_weights(model_file)
# }}}





