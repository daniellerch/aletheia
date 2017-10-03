import os
import sys
import PIL
import logging
import numpy
import tempfile
import shutil
import numpy
from scipy.io import savemat, loadmat
 

M_BIN="octave -q --no-gui --eval"


def _embed(sim, path, payload):

    X=numpy.array([])

    cwd=os.getcwd()
    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    m_path=os.path.join(basedir, 'external', 'octave')
    os.chdir(m_path)

    tmpdir=tempfile.mkdtemp()
    X_path=tmpdir+"/X.mat"

    m_code=""
    m_code+="warning('off');"
    m_code+="pkg load image;"

    if sim=='wow':
        m_code+="X=WOW('"+path+"',"+payload+");"
    elif sim=='hugo':
        m_code+="X=HUGO('"+path+"',"+payload+");"
    elif sim=='s_uniward':
        m_code+="X=S_UNIWARD('"+path+"',"+payload+");"

    m_code+="save('-mat7-binary', '"+X_path+"','X');"
    m_code+="exit"

    p=os.popen(M_BIN+" \""+m_code+"\"")
    p.read()
    os.chdir(cwd)

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



