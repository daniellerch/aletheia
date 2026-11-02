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

from aletheialib import utils

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count

_q = utils.octave_str # quote strings for Octave code


FEAEXT_1CH = ["GFR", "DCTR", "SRM", "SRMQ1"]
FEAEXT_3CH = ["SCRMQ1", "GFR", "SRM", "DCTR"]

# {{{ check_octave()
def check_octave():
    if not shutil.which("octave"):
        print("ERROR: you need to install Octave to run this command!");
        sys.exit(0)
# }}}

# {{{ embed()
def embed(sim, path, payload, dst_path=None):

    check_octave()

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    m_path=os.path.join(basedir, 'aletheia-cache', 'octave')


    if sim in ["j_uniward", "j_uniward_color", 
               "j_mipod",
               "nsf5", "nsf5_color", 
               "ebs", "ebs_color", 
               "ued", "ued_color", 
               "jpeg_read_struct"]:
        if not os.path.isfile(os.path.join(m_path, 'jpeg_write.mex')):
            tb_path=os.path.join(basedir, 'aletheia-cache', 'jpeg_toolbox')
            os.chdir(tb_path)
            os.system("make")


    payload = str(float(payload))
    for i in range(3):
        try:
            X=numpy.array([])

            im=Image.open(path)
            if (im.mode!='L' and sim in ["wow", "hugo", "hill", "s_uniward"]):
                print("Error,", sim, "must be used with grayscale images")
                return X

            tmpdir=tempfile.mkdtemp()
            X_path=tmpdir+"/X.mat"

            m_code=""
            m_code+="cd("+_q(tmpdir)+");"
            m_code+="addpath("+_q(m_path)+");"
            m_code+="warning('off');"
            m_code+="pkg load image;"
            m_code+="pkg load signal;"
            m_code+="pkg load nan;"

            if sim=='wow':
                m_code+="X=WOW("+_q(path)+","+payload+");"
            elif sim=='hugo':
                m_code+="X=HUGO("+_q(path)+","+payload+");"
            elif sim=='s_uniward':
                m_code+="X=S_UNIWARD("+_q(path)+","+payload+");"
            elif sim=='s_uniward_color':
                m_code+="X=S_UNIWARD_COLOR("+_q(path)+","+payload+");"
            elif sim=='hill':
                m_code+="X=HILL("+_q(path)+","+payload+");"
            elif sim=='hill_color':
                m_code+="X=HILL_COLOR("+_q(path)+","+payload+");"
            elif sim=='j_uniward':
                m_code+="J_UNIWARD("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='j_uniward_color':
                m_code+="J_UNIWARD_COLOR("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='j_mipod':
                m_code+="J_MIPOD("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='j_mipod_color':
                m_code+="J_MIPOD_COLOR("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='nsf5':
                m_code+="NSF5("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='nsf5_color':
                m_code+="NSF5_COLOR("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='ebs':
                m_code+="EBS("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='ebs_color':
                m_code+="EBS_COLOR("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='ued':
                m_code+="UED("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='ued_color':
                m_code+="UED_COLOR("+_q(path)+","+payload+","+_q(dst_path)+");"
            elif sim=='experimental':
                m_code+="X=EXPERIMENTAL("+_q(path)+","+payload+");"
            elif sim=='jpeg_read_struct':
                m_code+="X=JPEG_READ_STRUCT("+_q(path)+");"

            if not dst_path:
                m_code+="save('-mat7-binary', "+_q(X_path)+",'X');"
            m_code+="exit"

            p=utils.run_octave(m_code)

            if not dst_path:
                data=loadmat(X_path)
                X=data['X']
                shutil.rmtree(tmpdir)
                return X
             
            shutil.rmtree(tmpdir)
            return

        except:
            print("Error, retry!", path)
            continue
    return
# }}}

# {{{ _extract()
def _extract(extractor_name, path, params={}):

    check_octave()

    fdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(fdir, os.pardir))
    m_path=os.path.join(basedir, 'aletheia-cache', 'octave')


    X=numpy.array([])
    im=Image.open(path)
    if ((im.mode=='L' and extractor_name in FEAEXT_1CH) or 
        (im.mode in ['RGB', 'RGBA', 'RGBX'] and extractor_name in FEAEXT_3CH)):
        tmpdir=tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
        except Exception as e:
            print("chdir:", str(e))

        channel = 1
        if "channel" in params:
            channel = params["channel"]

        data_path=tmpdir+"/data.mat"
        m_code=""
        m_code+="cd("+_q(tmpdir)+");"
        m_code+="addpath("+_q(m_path)+");"
        m_code+="warning('off');"
        m_code+="pkg load image;"
        m_code+="pkg load signal;"
        m_code+="pkg load nan;"
        if extractor_name=="GFR":
            m_code+="data="+extractor_name+"("+_q(path)+"," \
                    +str(int(params["rotations"]))+", "+str(int(params["quality"]))+", "+str(int(channel))+");"
        elif extractor_name=="DCTR":
            m_code+="data="+extractor_name+"("+_q(path)+"," \
                    +str(int(params["quality"]))+", "+str(int(channel))+");"
        else:
            m_code+="data="+extractor_name+"("+_q(path)+", "+str(int(channel))+");"
        m_code+="save('-mat7-binary', "+_q(data_path)+",'data');"
        m_code+="exit"
        p=utils.run_octave(m_code)

        data=loadmat(data_path)
        shutil.rmtree(tmpdir)

        if extractor_name in ["GFR", "DCTR"]:
            X = data["data"][0]
        else:
            for submodel in data["data"][0][0]:
                X = numpy.hstack((X,submodel.reshape((submodel.shape[1]))))

    else:
        print("Image mode/extractor not supported: ", im.mode, "/", extractor_name)
        print("")
        sys.stdout.flush()

    im.close()

    return X
# }}}

# {{{ _jpeg()
def _jpeg(fn_name, path):
 
    check_octave()

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    m_path=os.path.join(basedir, 'aletheia-cache', 'octave')

    if not os.path.isfile(os.path.join(m_path, 'jpeg_read.mex')):
        tb_path=os.path.join(basedir, 'aletheia-cache', 'jpeg_toolbox')
        os.chdir(tb_path)
        os.system("make")

    fn_names_with_return = ['jpeg_read_struct']

    for i in range(3):
        try:
            X=numpy.array([])
            im=Image.open(path)

            tmpdir=tempfile.mkdtemp()
            X_path=tmpdir+"/X.mat"

            m_code=""
            m_code+="cd("+_q(tmpdir)+");"
            m_code+="addpath("+_q(m_path)+");"
            m_code+="warning('off');"
            m_code+="pkg load image;"
            m_code+="pkg load signal;"
            m_code+="pkg load nan;"

            if fn_name=='jpeg_read_struct':
                m_code+="X=JPEG_READ_STRUCT("+_q(path)+");"

            if fn_name in fn_names_with_return:
                m_code+="save('-mat7-binary', "+_q(X_path)+",'X');"
            m_code+="exit"

            p=utils.run_octave(m_code)

            if fn_name in fn_names_with_return:
                data=loadmat(X_path)
                X=data['X']
                shutil.rmtree(tmpdir)
                return X
             
            shutil.rmtree(tmpdir)
            return

        except:
            print("Error, retry!", path)
            continue
    return
# }}}

# {{{ _attack()
def _attack(attack_name, path, params={}):

    check_octave()

    fdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(fdir, os.pardir))
    m_path=os.path.join(basedir, 'aletheia-cache', 'octave')

    X=numpy.array([])
    im=Image.open(path)
    tmpdir=tempfile.mkdtemp()
    try:
        os.chdir(tmpdir)
    except Exception as e:
        print("chdir:", str(e))

    channel = 1
    if "channel" in params:
        channel = params["channel"]

    data_path=tmpdir+"/data.mat"
    m_code=""
    m_code+="cd("+_q(tmpdir)+");"
    m_code+="addpath("+_q(m_path)+");"
    m_code+="warning('off');"
    m_code+="pkg load image;"
    m_code+="pkg load signal;"
    m_code+="pkg load nan;"
    m_code+="data="+attack_name+"("+_q(path)+", "+str(int(channel))+");"
    m_code+="save('-mat7-binary', "+_q(data_path)+",'data');"
    m_code+="exit"
    p=utils.run_octave(m_code)

    data=loadmat(data_path)
    shutil.rmtree(tmpdir)

    im.close()

    return data
# }}}




