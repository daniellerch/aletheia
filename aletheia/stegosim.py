import os
import sys
import scipy
import logging
import numpy
import tempfile
import shutil
import numpy
import subprocess
import random

from scipy.io import savemat, loadmat
from scipy import misc
from PIL import Image

from aletheia import utils
from aletheia.octave_interface import _embed

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count


# {{{ embed_message()
def embed_message(embed_fn, path, payload, output_dir, 
                  embed_fn_saving=False):

    path=utils.absolute_path(path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir=utils.absolute_path(output_dir)

    # Read filenames
    files=[]
    if os.path.isdir(path):
        for dirpath,_,filenames in os.walk(path):
            for f in filenames:
                path=os.path.abspath(os.path.join(dirpath, f))
                if not utils.is_valid_image(path):
                    print("Warning, please provide a valid image: ", f)
                else:
                    files.append(path)
    else:
        files=[path]
    
    # remove fileas already generated in a previous execution
    filtered_files = []
    for f in files:
        basename=os.path.basename(f)
        dst_path=os.path.join(output_dir, basename)
        if os.path.exists(dst_path):
            print("Warning! file already exists, ignored:", dst_path)
            continue
        filtered_files.append(f)
    files = filtered_files
    del filtered_files

    def embed(path):
        basename=os.path.basename(path)
        dst_path=os.path.join(output_dir, basename)

        if embed_fn_saving:
            embed_fn(path, payload, dst_path)
        else:
            X=embed_fn(path, payload)
            try:
                scipy.misc.toimage(X, cmin=0, cmax=255).save(dst_path)
            except Exception as e:
                print(str(e))

    # Process thread pool in batches
    batch=1000
    for i in range(0, len(files), batch):
        files_batch = files[i:i+batch]
        n_core=cpu_count()
        print("Using", n_core, "threads")
        pool = ThreadPool(n_core)
        results = pool.map(embed, files_batch)
        pool.close()
        pool.terminate()
        pool.join()

    """
    for path in files:
        I=scipy.misc.imread(path)
        basename=os.path.basename(path)
        dst_path=os.path.join(output_dir, basename)
        if embed_fn_saving:
            print path, payload, dst_path
            embed_fn(path, payload, dst_path)
        else:
            X=embed_fn(path, payload)
            try:
                scipy.misc.toimage(X, cmin=0, cmax=255).save(dst_path)
            except Exception, e:
                print str(e)
    """
   
# }}}


def wow(path, payload):
    return _embed('wow', path, payload)

def s_uniward(path, payload):
    return _embed('s_uniward', path, payload)

def j_uniward(path, payload, dst_path):
    return _embed('j_uniward', path, payload, dst_path)

def j_uniward_color(path, payload, dst_path):
    return _embed('j_uniward_color', path, payload, dst_path)

def hugo(path, payload):
    return _embed('hugo', path, payload)

def hill(path, payload):
    return _embed('hill', path, payload)

def ebs(path, payload, dst_path):
    return _embed('ebs', path, payload, dst_path)

def ebs_color(path, payload, dst_path):
    return _embed('ebs_color', path, payload, dst_path)

def ued(path, payload, dst_path):
    return _embed('ued', path, payload, dst_path)

def ued_color(path, payload, dst_path):
    return _embed('ued_color', path, payload, dst_path)

def nsf5(path, payload, dst_path):
    return _embed('nsf5', path, payload, dst_path)

def nsf5_color(path, payload, dst_path):
    return _embed('nsf5_color', path, payload, dst_path)

def experimental(path, payload):
    return _embed('experimental', path, payload)

# {{{ lsbm()
def lsbm(path, payload):
    X = misc.imread(path)
    sign=[1, -1]
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            if random.randint(0,99)>int(float(payload)*100):
                continue
 
            if len(X.shape)==2:
                k=sign[random.randint(0, 1)]
                if X[i, j]==0: k=1
                if X[i, j]==255: k=-1
                if X[i, j]%2!=random.randint(0,1): # message
                    X[i, j]+=k
            else:
                kr=sign[random.randint(0, 1)]
                kg=sign[random.randint(0, 1)]
                kb=sign[random.randint(0, 1)]
                if X[i, j][0]==0: kr=1
                if X[i, j][1]==0: kg=1
                if X[i, j][2]==0: kb=1
                if X[i, j][0]==255: kr=-1
                if X[i, j][1]==255: kg=-1
                if X[i, j][2]==255: kb=-1
                # message
                if X[i, j][0]%2==random.randint(0,1): kr=0
                if X[i, j][1]%2==random.randint(0,1): kg=0
                if X[i, j][2]%2==random.randint(0,1): kb=0
                X[i, j]=(X[i,j][0]+kr, X[i,j][1]+kg, X[i,j][2]+kb)
    return X
# }}}

# {{{ lsbr()
def lsbr(path, payload):
    X = misc.imread(path)
    sign=[1, -1]
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            if random.randint(0,99)>int(float(payload)*100):
                continue
 
            if len(X.shape)==2:
                k=sign[random.randint(0, 1)]
                if X[i, j]==0: k=1
                if X[i, j]==255: k=-1
                if X[i, j]%2!=random.randint(0,1): # message
                    if X[i, j]%2==0: X[i, j]+=1
                    else: X[i, j]-=1
            else:
                # message
                kr=0; Kg=0; kb=0

                if X[i, j][0]%2==0: kr=1
                else: kr=-1

                if X[i, j][1]%2==0: kg=1
                else: kg=-1

                if X[i, j][2]%2==0: kb=1
                else: kb=-1

                if X[i, j][0]%2==random.randint(0,1): kr=0
                if X[i, j][1]%2==random.randint(0,1): kg=0
                if X[i, j][2]%2==random.randint(0,1): kb=0
                X[i, j]=(X[i,j][0]+kr, X[i,j][1]+kg, X[i,j][2]+kb)
    return X
# }}}

# {{{ embedding_fn()
def embedding_fn(name):
    if name=="lsbm-sim":
        return lsbm
    if name=="lsbr-sim":
        return lsbr
    if name=="hugo-sim":
        return hugo
    if name=="wow-sim":
        return wow
    if name=="s-uniward-sim":
        return s_uniward
    if name=="j-uniward-sim":
        return j_uniward
    if name=="j-uniward-color-sim":
        return j_uniward_color
    if name=="hill-sim":
        return hill
    if name=="nsf5-sim":
        return nsf5
    if name=="nsf5-color-sim":
        return nsf5_color
    if name=="ebs-sim":
        return ebs
    if name=="ebs-color-sim":
        return ebs_color
    if name=="ued-sim":
        return ued
    if name=="ueb-color-sim":
        return ued_color

    print("Unknown simulator:", name)
    sys.exit(0)
# }}}






