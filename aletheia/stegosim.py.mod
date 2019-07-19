import os
import sys
import scipy
import logging
import tempfile
import shutil
import subprocess
import random

import numpy as np
from scipy.io import savemat, loadmat
from scipy import misc, ndimage, signal
from PIL import Image
import hdf5storage

from aletheia import utils

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count

M_BIN="octave -q --no-gui --eval"


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


def _embed(sim, path, payload, dst_path=None):
    X=np.array([])

    im=Image.open(path)
    if (im.mode!='L' and sim in ["wow", "hugo", "hill", "s_uniward"]):
        print("Error,", sim, "must be used with grayscale images")
        return X

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    m_path=os.path.join(basedir, 'external', 'octave')

    tmpdir=tempfile.mkdtemp()
    X_path=tmpdir+"/X.mat"

    m_code=""
    m_code+="cd "+tmpdir+";"
    m_code+="addpath('"+m_path+"');"
    m_code+="warning('off');"
    m_code+="pkg load image;"
    m_code+="pkg load signal;"

    if sim=='wow':
        m_code+="X=WOW('"+path+"',"+payload+");"
    elif sim=='hugo':
        m_code+="X=HUGO('"+path+"',"+payload+");"
    elif sim=='s_uniward':
        m_code+="X=S_UNIWARD('"+path+"',"+payload+");"
    elif sim=='hill':
        m_code+="X=HILL('"+path+"',"+payload+");"
    elif sim=='j_uniward':
        m_code+="J_UNIWARD('"+path+"',"+payload+",'"+dst_path+"');"
    elif sim=='j_uniward_color':
        m_code+="J_UNIWARD_COLOR('"+path+"',"+payload+",'"+dst_path+"');"
    elif sim=='nsf5':
        m_code+="NSF5('"+path+"',"+payload+",'"+dst_path+"');"
    elif sim=='nsf5_color':
        m_code+="NSF5_COLOR('"+path+"',"+payload+",'"+dst_path+"');"
    elif sim=='ebs':
        m_code+="EBS('"+path+"',"+payload+",'"+dst_path+"');"
    elif sim=='ebs_color':
        m_code+="EBS_COLOR('"+path+"',"+payload+",'"+dst_path+"');"
    elif sim=='ued':
        m_code+="UED('"+path+"',"+payload+",'"+dst_path+"');"
    elif sim=='ued_color':
        m_code+="UED_COLOR('"+path+"',"+payload+",'"+dst_path+"');"
    elif sim=='experimental':
        m_code+="X=EXPERIMENTAL('"+path+"',"+payload+");"

    if not dst_path:
        m_code+="save('-mat7-binary', '"+X_path+"','X');"
    m_code+="exit"

    p=subprocess.Popen(M_BIN+" \""+m_code+"\"", stdout=subprocess.PIPE, shell=True)
    #output, err = p.communicate()
    status = p.wait()

    if not dst_path:
        data=loadmat(X_path)
        X=data['X']
        shutil.rmtree(tmpdir)
        return X
     
    shutil.rmtree(tmpdir)
    return

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

def testing(path, payload):

    # {{{ hill_cost()
    def hill_cost(I):
        HF1 = np.array([
            [-1, 2,-1],
            [ 2,-4, 2],
            [-1, 2,-1]
        ])
        H2 = np.ones((3, 3)).astype(np.float)/3**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
        W1 = signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm') 
        rho=1./(W1+10**(-10))
        cost = signal.convolve2d(rho, HW, mode='same', boundary='symm') 
       
        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1
    # }}}

    # {{{ f2_hill_cost()
    def f2_hill_cost(I):
        HF1 = np.array([
            [ 1, 1, 1],
            [ 1,-8, 1],
            [ 1, 1, 1]
        ])
        H2 = np.ones((3, 3)).astype(np.float)/3**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
        W1 = signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm') 
        rho=1./(W1+10**(-10))
        cost = signal.convolve2d(rho, HW, mode='same', boundary='symm') 
       
        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1

    # }}}

    # {{{ hill5x5_cost()
    def hill5x5_cost(I):
        HF1 = np.array([
            [-1, 2,-2, 2,-1],
            [ 2,-6, 8,-6, 2],
            [-2, 8,-12,8,-2],
            [ 2,-6, 8,-6, 2],
            [-1, 2,-2, 2,-1]
        ])
        H2 = np.ones((5, 5)).astype(np.float)/5**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
        W1 = signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm') 
        rho=1./(W1+10**(-10))
        cost = signal.convolve2d(rho, HW, mode='same', boundary='symm') 
    
        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1
    # }}}

    # {{{ hill4x4_cost()
    def hill4x4_cost(I):
        HF1 = np.array([
            [-1, 1,-1, 1],
            [ 1,-1, 1,-1],
            [-1, 1,-1, 1],
            [ 1,-1, 1,-1]
        ])
        H2 = np.ones((4, 4)).astype(np.float)/5**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
        W1 = signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm') 
        rho=1./(W1+10**(-10))
        cost = signal.convolve2d(rho, HW, mode='same', boundary='symm') 
    
        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1
    # }}}

    # {{{ mx9_hill_cost()
    def mx9_hill_cost(I):
        H2 = np.ones((5, 5)).astype(np.float)/5**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        kernels = []

        k = np.array([
            [-1, 2,-2, 2,-1],
            [ 2,-6, 8,-6, 2],
            [-2, 8,-12,8,-2],
            [ 2,-6, 8,-6, 2],
            [-1, 2,-2, 2,-1]
        ])
        kernels.append(k)

        k = np.array([
            [-1, 2,-2, 0, 0],
            [ 2,-6, 8, 0, 0],
            [-2, 8,-12,0, 0],
            [ 2,-6, 8, 0, 0],
            [-1, 2,-2, 0, 0]
        ])
        kernels.append(k)

        k = np.array([
            [ 0, 0,-2, 2,-1],
            [ 0, 0, 8,-6, 2],
            [ 0, 0,-12,8,-2],
            [ 0, 0, 8,-6, 2],
            [ 0, 0,-2, 2,-1]
        ])
        kernels.append(k)

        k = np.array([
            [ 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0],
            [-2, 8,-12,8,-2],
            [ 2,-6, 8,-6, 2],
            [-1, 2,-2, 2,-1]
        ])
        kernels.append(k)

        k = np.array([
            [-1, 2,-2, 2,-1],
            [ 2,-6, 8,-6, 2],
            [-2, 8,-12,8,-2],
            [ 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0]
        ])
        kernels.append(k)


        k = np.array([
            [-1, 2,-2, 2,-1],
            [ 2,-6, 8,-6, 0],
            [-2, 8,-12,0, 0],
            [ 2,-6, 0, 0, 0],
            [-1, 0, 0, 0, 0]
        ])
        kernels.append(k)

        k = np.array([
            [ 0, 0, 0, 0,-1],
            [ 0, 0, 0,-6, 2],
            [ 0, 0,-12,8,-2],
            [ 0,-6, 8,-6, 2],
            [-1, 2,-2, 2,-1]
        ])
        kernels.append(k)

        k = np.array([
            [-1, 0, 0, 0, 0],
            [ 2,-6, 0, 0, 0],
            [-2, 8,-12,0, 0],
            [ 2,-6, 8,-6, 0],
            [-1, 2,-2, 2,-1]
        ])
        kernels.append(k)

        k = np.array([
            [-1, 2,-2, 2,-1],
            [ 0,-6, 8,-6, 2],
            [ 0, 0,-12,8,-2],
            [ 0, 0, 0,-6, 2],
            [ 0, 0, 0, 0,-1]
        ])
        kernels.append(k)



        cost_list = []
        for HF1 in kernels:
            R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
            W1 = signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm') 

            W1 = W1 - np.min(W1)
            W1 /= np.max(W1)

            rho=1./(W1+10**(-10))
            cost = signal.convolve2d(rho, HW, mode='same', boundary='symm') 
            cost_list.append(np.expand_dims(cost, axis=2))

        costs = np.concatenate((cost_list), axis=2)
        cost = np.max(costs, axis=2)

        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1
    # }}}

    # {{{ usm()
    def usm(image, radius, amount):
        blurred = ndimage.gaussian_filter(image, sigma=radius, mode='reflect')
        result = image + (image - blurred) * amount
        return result
    # }}}

    # {{{ pm_hill_cost()
    # 100% detected!
    def pm_hill_cost(I):
        HF1 = np.array([
            [-1, 2,-1],
            [ 2,-4, 2],
            [-1, 2,-1]
        ])
        H2 = np.ones((3, 3)).astype(np.float)/3**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
        R1neg = R1.copy()
        R1pos = R1.copy()

        R1neg[R1neg<0] = 0
        R1pos[R1pos>0] = 0

        W1pos = signal.convolve2d(R1pos, H2, mode='same', boundary='symm') 
        W1neg = signal.convolve2d(R1neg, H2, mode='same', boundary='symm') 
        rhopos=1./(W1pos+10**(-10))
        rhoneg=1./(W1neg+10**(-10))
        costpos = signal.convolve2d(rhopos, HW, mode='same', boundary='symm') 
        costneg = signal.convolve2d(rhoneg, HW, mode='same', boundary='symm') 
       
        wet_cost = 10**8
        costpos[costpos>wet_cost] = wet_cost
        costneg[costneg>wet_cost] = wet_cost

        costP1 = costpos
        costM1 = costneg

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost

        return costP1, costM1
    # }}}

    # {{{ mx2_hill_cost()
    def mx2_hill_cost(I):
        H2 = np.ones((3, 3)).astype(np.float)/3**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        kernels = []

        k = np.array([
            [-1, 2,-1],
            [ 2,-4, 2],
            [-1, 2,-1]
        ])
        kernels.append(k)

        F = np.array([
            [ 2,-1, 2],
            [-1,-4,-1],
            [ 2,-1, 2]
        ])
        kernels.append(k)



        cost_list = []
        for HF1 in kernels:
            R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
            W1 = signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm') 

            W1 = W1 - np.min(W1)
            W1 /= np.max(W1)

            rho=1./(W1+10**(-10))
            cost = signal.convolve2d(rho, HW, mode='same', boundary='symm') 
            cost_list.append(np.expand_dims(cost, axis=2))

        costs = np.concatenate((cost_list), axis=2)
        cost = np.max(costs, axis=2)

        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1
    # }}}

    # {{{ mx4_3x3_hill_cost()
    def mx4_3x3_hill_cost(I):
        H2 = np.ones((3, 3)).astype(np.float)/3**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        kernels = []

        k = np.array([
            [-1, 2,-1],
            [ 2,-4, 2],
            [-1, 2,-1]
        ])
        kernels.append(k)

        F = np.array([
            [ 2,-1, 2],
            [-1,-4,-1],
            [ 2,-1, 2]
        ])
        kernels.append(k)

        k = np.array([
            [ 1, 0,-1],
            [ 2, 0,-2],
            [ 1, 0,-1]
        ])
        kernels.append(k)

        k = np.array([
            [ 1, 2, 1],
            [ 0, 0, 0],
            [-1,-2,-1]
        ])
        kernels.append(k)





        cost_list = []
        for HF1 in kernels:
            R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
            W1 = signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm') 

            W1 = W1 - np.min(W1)
            W1 /= np.max(W1)

            rho=1./(W1+10**(-10))
            cost = signal.convolve2d(rho, HW, mode='same', boundary='symm') 
            cost_list.append(np.expand_dims(cost, axis=2))

        costs = np.concatenate((cost_list), axis=2)
        cost = np.max(costs, axis=2)

        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1
    # }}}

    # {{{ mx4_hill_cost()
    def mx4_hill_cost(I):
        H2 = np.ones((3, 3)).astype(np.float)/3**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        kernels = []

        k = np.array([
            [ 0, 0, 0],
            [ 1,-2, 1],
            [ 0, 0, 0]
        ])
        kernels.append(k)

        k = np.array([
            [ 0, 1, 0],
            [ 0,-2, 0],
            [ 0, 1, 0]
        ])
        kernels.append(k)

        k = np.array([
            [ 1, 0, 0],
            [ 0,-2, 0],
            [ 0, 0, 1]
        ])
        kernels.append(k)

        k = np.array([
            [ 0, 0, 1],
            [ 0,-2, 0],
            [ 1, 0, 0]
        ])
        kernels.append(k)


        cost_list = []
        for HF1 in kernels:
            R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
            W1 = signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm') 

            W1 = W1 - np.min(W1)
            W1 /= np.max(W1)

            rho=1./(W1+10**(-10))
            cost = signal.convolve2d(rho, HW, mode='same', boundary='symm') 
            cost_list.append(np.expand_dims(cost, axis=2))

        costs = np.concatenate((cost_list), axis=2)
        cost = np.max(costs, axis=2)

        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1
    # }}}

    # {{{ mx2_5x5_hill_cost()
    def mx2_5x5_hill_cost(I):
        H2 = np.ones((5, 5)).astype(np.float)/5**2
        HW = np.ones((15, 15)).astype(np.float)/15**2

        kernels = []

        k = np.array([
            [-1, 2,-2, 2,-1],
            [ 2,-6, 8,-6, 2],
            [-2, 8,-12,8,-2],
            [ 2,-6, 8,-6, 2],
            [-1, 2,-2, 2,-1]
        ])
        kernels.append(k)

        k = np.array([
            [-2, 2,-1,  2,-2],
            [ 2, 8,-6,  8, 2],
            [-1,-6,-12,-6,-1],
            [ 2, 8,-6,  8, 2],
            [-2, 2,-1,  2,-2]
        ])
        kernels.append(k)



        cost_list = []
        for HF1 in kernels:
            R1 = signal.convolve2d(I, HF1, mode='same', boundary='symm') 
            W1 = signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm') 

            W1 = W1 - np.min(W1)
            W1 /= np.max(W1)

            rho=1./(W1+10**(-10))
            cost = signal.convolve2d(rho, HW, mode='same', boundary='symm') 
            cost_list.append(np.expand_dims(cost, axis=2))

        costs = np.concatenate((cost_list), axis=2)
        cost = np.max(costs, axis=2)

        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1
    # }}}

    # {{{ f7x7_cost()
    def f7x7_cost(I):
        HF1 = np.array([
            [-1, 1,-1, 1,-1, 1,-1],
            [ 1,-1, 1,-1, 1,-1, 1],
            [-1, 1,-1, 2,-1, 1,-1],
            [ 1,-1, 2,-4, 2,-1, 1],
            [-1, 1,-1, 2,-1, 1,-1],
            [ 1,-1, 1,-1, 1,-1, 1],
            [-1, 1,-1, 1,-1, 1,-1]
        ])
        W = np.abs(signal.convolve2d(I, HF1, mode='same', boundary='symm'))
        cost=1./(W+10**(-10))
       
        wet_cost = 10**8
        cost[cost>wet_cost] = wet_cost
        costP1 = cost.copy()
        costM1 = cost.copy()

        costM1[I==255] = wet_cost
        costP1[I==0] = wet_cost
        return costP1, costM1
    # }}}


    I = misc.imread(path).astype(np.float)
    #Is = usm(I, radius=1, amount=1)
    #Ir = np.round(Is).astype(np.int)

    #if float(payload)==0:
    #    return Ir

    costP1, costM1 = f7x7_cost(I)
    #cost = hill2_cost(I)

    return ml_stc(I, costP1, costM1, payload)


def experimental(path, payload):
    return _embed('experimental', path, payload)

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




def embedding_fn(name):
    if name=="lsbm-sim":
        return stegosim.lsbm
    if name=="lsbr-sim":
        return stegosim.lsbr
    if name=="hugo-sim":
        return stegosim.hugo
    if name=="wow-sim":
        return stegosim.wow
    if name=="s-uniward-sim":
        return stegosim.s_uniward
    if name=="j-uniward-sim":
        return stegosim.j_uniward
    if name=="j-uniward-color-sim":
        return stegosim.j_uniward_color
    if name=="hill-sim":
        return stegosim.hill
    if name=="nsf5-sim":
        return stegosim.nsf5
    if name=="nsf5-color-sim":
        return stegosim.nsf5_color
    if name=="ebs-sim":
        return stegosim.ebs
    if name=="ebs-color-sim":
        return stegosim.ebs_color
    if name=="ued-sim":
        return stegosim.ued
    if name=="ueb-color-sim":
        return stegosim.ued_color

    print("Unknown simulator:", name)
    sys.exit(0)



def ml_stc(I, costP1, costM1, payload):

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    m_path=os.path.join(basedir, 'external', 'octave')

    for err in range(3):
        try:
            tmpdir=tempfile.mkdtemp()
            X_path=tmpdir+"/X.mat"
            I_path=tmpdir+"/I.mat"
            costP1_path=tmpdir+"/costP1.mat"
            costM1_path=tmpdir+"/costM1.mat"

            hdf5storage.write({u'M': I}, '.', I_path, matlab_compatible=True)
            hdf5storage.write({u'M': costP1}, '.', costP1_path, matlab_compatible=True)
            hdf5storage.write({u'M': costM1}, '.', costM1_path, matlab_compatible=True)
            break
        except:
            print("Error, retry", err)
            if err==2:
                print("Too many errors!")
            pass


    m_code=""
    m_code+="cd "+tmpdir+";"
    m_code+="addpath('"+m_path+"');"
    m_code+="warning('off');"
    m_code+="pkg load image;"
    m_code+="pkg load signal;"
    m_code+="X=ML_STC('"+I_path+"','"+costP1_path+"','"+costM1_path+"',"+payload+");"
 
    m_code+="save('-mat7-binary', '"+X_path+"','X');"
    m_code+="exit"

    p=subprocess.Popen(M_BIN+" \""+m_code+"\"", stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    status = p.wait()

    data=loadmat(X_path)
    X=data['X']
    shutil.rmtree(tmpdir)
    return X
 



