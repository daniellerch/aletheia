import os
import sys
import numpy
import logging
import tempfile
import shutil
from PIL import Image
from scipy.io import loadmat

M_BIN="octave -q --no-gui --eval"


# {{{ SRM_extract()
def SRM_extract(path):
    cwd=os.getcwd()
    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    m_path=os.path.join(basedir, 'external', 'octave')

    X=numpy.array([])
    im=Image.open(path)
    if im.mode=='L':
        tmpdir=tempfile.mkdtemp()
        os.chdir(tmpdir)
        data_path=tmpdir+"/data.mat"
        m_code=""
        m_code+="addpath('"+m_path+"');"
        m_code+="warning('off');"
        m_code+="data=SRM('"+path+"', 1);"
        m_code+="save('-mat7-binary', '"+data_path+"','data');"
        m_code+="exit"
        p=os.popen(M_BIN+" \""+m_code+"\"")
        p.read()
        p.close()
        data=loadmat(data_path)
        os.chdir(cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)
        for submodel in data["data"][0][0]:
            X = numpy.hstack((X,submodel.reshape((submodel.shape[1]))))

    elif im.mode in ['RGB', 'RGBA', 'RGBX']:
        tmpdir=tempfile.mkdtemp()
        os.chdir(tmpdir)
        data_R_path=tmpdir+"/data_R.mat"
        data_G_path=tmpdir+"/data_G.mat"
        data_B_path=tmpdir+"/data_B.mat"
        m_code=""
        m_code+="warning('off');"
        m_code+="data_R=SRM('"+path+"', 1);"
        m_code+="data_G=SRM('"+path+"', 2);"
        m_code+="data_B=SRM('"+path+"', 3);"
        m_code+="save('-mat7-binary', '"+data_R_path+"','data_R');"
        m_code+="save('-mat7-binary', '"+data_G_path+"','data_G');"
        m_code+="save('-mat7-binary', '"+data_B_path+"','data_B');"
        m_code+="exit"
        p=os.popen(M_BIN+" \""+m_code+"\"")
        p.read()
        R=numpy.array([])
        G=numpy.array([])
        B=numpy.array([])
        data_R=loadmat(data_R_path)
        data_G=loadmat(data_G_path)
        data_B=loadmat(data_B_path)
        for submodel in data_R["data_R"][0][0]:
            R = numpy.hstack((R,submodel.reshape((submodel.shape[1]))))
        for submodel in data_G["data_G"][0][0]:
            G = numpy.hstack((G,submodel.reshape((submodel.shape[1]))))
        for submodel in data_B["data_B"][0][0]:
            B = numpy.hstack((B,submodel.reshape((submodel.shape[1]))))
        X=numpy.hstack((R,G,B))
        os.chdir(cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print "Image mode not supported: ", im.mode
        sys.stdout.flush()

    im.close()

    return X
# }}}

