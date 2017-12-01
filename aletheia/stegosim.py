import os
import sys
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
 

M_BIN="octave -q --no-gui --eval"


def _embed(sim, path, payload):

    X=numpy.array([])

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

    if sim=='wow':
        m_code+="X=WOW('"+path+"',"+payload+");"
    elif sim=='hugo':
        m_code+="X=HUGO('"+path+"',"+payload+");"
    elif sim=='s_uniward':
        m_code+="X=S_UNIWARD('"+path+"',"+payload+");"
    elif sim=='hill':
        m_code+="X=HILL('"+path+"',"+payload+");"

    m_code+="save('-mat7-binary', '"+X_path+"','X');"
    m_code+="exit"

    p=subprocess.Popen(M_BIN+" \""+m_code+"\"", shell=True)
    output, err = p.communicate()
    status = p.wait()

    data=loadmat(X_path)
    shutil.rmtree(tmpdir)
    
    X=data['X']

    return X
     

def wow(path, payload):
    return _embed('wow', path, payload)

def s_uniward(path, payload):
    return _embed('s_uniward', path, payload)

def hugo(path, payload):
    return _embed('hugo', path, payload)

def hill(path, payload):
    return _embed('hill', path, payload)

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


