import os
import sys
import numpy
import logging
import tempfile
import shutil
import subprocess
from PIL import Image
from scipy.io import loadmat

M_BIN="octave -q --no-gui --eval"

FEAEXT_1CH = ["SRM", "SRMQ1", "HILL_MAXSRM", "HILL_sigma_spam_PSRM"]
FEAEXT_3CH = ["SCRMQ1", "GFR"]

# {{{ _extract()
def _extract(extractor_name, path, params={}):
    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    m_path=os.path.join(basedir, 'external', 'octave')

    X=numpy.array([])
    im=Image.open(path)
    if ((im.mode=='L' and extractor_name in FEAEXT_1CH) or 
        (im.mode in ['RGB', 'RGBA', 'RGBX'] and extractor_name in FEAEXT_3CH)):
        tmpdir=tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
        except Exception,e:
            print "chdir:", str(e)

        data_path=tmpdir+"/data.mat"
        m_code=""
        m_code+="cd "+tmpdir+";"
        m_code+="addpath('"+m_path+"');"
        m_code+="warning('off');"
        m_code+="pkg load image;"
        m_code+="pkg load signal;"
        if extractor_name=="GFR":
            m_code+="data="+extractor_name+"('"+path+"'," \
                    +str(params["rotations"])+", "+str(params["quality"])+");"
        else:
            m_code+="data="+extractor_name+"('"+path+"', 1);"
        m_code+="save('-mat7-binary', '"+data_path+"','data');"
        m_code+="exit"
        p=subprocess.Popen(M_BIN+" \""+m_code+"\"", stdout=subprocess.PIPE, shell=True)
        # output, err = p.communicate()
        status = p.wait()

        data=loadmat(data_path)
        shutil.rmtree(tmpdir)

        if extractor_name=="GFR":
            X = data["data"][0]
        else:
            for submodel in data["data"][0][0]:
                X = numpy.hstack((X,submodel.reshape((submodel.shape[1]))))

    else:
        print "Image mode/extractor not supported: ", im.mode, "/", extractor_name
        print
        sys.stdout.flush()

    im.close()

    return X
# }}}


def SRM_extract(path):
    return _extract('SRM', path)

def SRMQ1_extract(path):
    return _extract('SRMQ1', path)

def SCRMQ1_extract(path):
    return _extract('SCRMQ1', path)

def GFR_extract(path, quality="auto", rotations=40):
    
    if quality=="auto":
        try:
            p=subprocess.Popen("identify -format '%Q' "+path, \
                               stdout=subprocess.PIPE, shell=True)
            quality, err = p.communicate()
            status = p.wait()
        except:
            quality = 95


    # Only two qualities are supported. Use the closest.
    quality = int(quality)
    if abs(quality-75) > abs(quality-95):
        quality = 95
    else:
        quality = 75

    params = {
        "rotations": rotations,
        "quality": quality
    }
    print params

    return _extract('GFR', path, params)

def HILL_sigma_spam_PSRM_extract(path):
    return _extract('HILL_sigma_spam_PSRM', path)

def HILL_MAXSRM_extract(path):
    return _extract('HILL_MAXSRM', path)


def extractor_fn(name):
    if name == "srm": 
        return SRM_extract
    if name == "srmq1": 
        return SRMQ1_extract
    if name == "scrmq1": 
        return SCRMQ1_extract
    if name == "gfr": 
        return GFR_extract

    print "Unknown feature extractor:", name
    sys.exit(0)


