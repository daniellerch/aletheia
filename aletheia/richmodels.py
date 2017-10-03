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
    os.chdir(m_path)

    tmpdir=tempfile.mkdtemp()
    data_path=tmpdir+"/data.mat"

    im=Image.open(path)
    if im.mode=='L':
        m_code=""
        m_code+="warning('off');"
        m_code+="data=SRM('"+path+"', 0);"
        m_code+="save('-mat7-binary', '"+data_path+"','data');"
        m_code+="exit"
        p=os.popen(M_BIN+" \""+m_code+"\"")
        p.read()
        data=loadmat(data_path)
        shutil.rmtree(tmpdir)

        X=numpy.array([])
        for submodel in data["data"][0][0]:
            X = numpy.hstack((X,submodel.reshape((submodel.shape[1]))))

    elif im.mode in ['RGB', 'RGBA', 'RGBX']:
        m_code=""
        m_code+="warning('off');"
        m_code+="data_R=SRM('"+path+"', 1);"
        m_code+="data_G=SRM('"+path+"', 2);"
        m_code+="data_B=SRM('"+path+"', 3);"
        m_code+="save('-mat7-binary', '"+data_path+"','data');"
        m_code+="exit"
        p=os.popen(M_BIN+" \""+m_code+"\"")
        p.read()
        data=loadmat(data_path)
        shutil.rmtree(tmpdir)

        R=numpy.array([])
        G=numpy.array([])
        B=numpy.array([])
        for submodel in data["data_R"][0][0]:
            R = numpy.hstack((X,submodel.reshape((submodel.shape[1]))))
        for submodel in data["data_G"][0][0]:
            G = numpy.hstack((X,submodel.reshape((submodel.shape[1]))))
        for submodel in data["data_B"][0][0]:
            B = numpy.hstack((X,submodel.reshape((submodel.shape[1]))))
        X=numpy.hstack((R,G,B))

    else:
        print "Image mode not supported: ", im.mode
        sys.stdout.flush()


    os.chdir(cwd)




    im.close()


    return X
# }}}

