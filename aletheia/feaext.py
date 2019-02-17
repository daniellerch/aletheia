import os
import sys
import numpy
import logging
import tempfile
import shutil
import subprocess
from PIL import Image
from scipy.io import loadmat

from aletheia import utils
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count

lock = multiprocessing.Lock()

M_BIN="octave -q --no-gui --eval"

FEAEXT_1CH = ["SRM", "SRMQ1", "HILL_MAXSRM", "HILL_sigma_spam_PSRM"]
FEAEXT_3CH = ["SCRMQ1", "GFR", "SRM"]


# {{{ extract_features()
def extract_features(extract_fn, image_path, ofile, params={}):

    cwd = os.getcwd()
    image_path=utils.absolute_path(image_path)

    # Read filenames
    files=[]
    if os.path.isdir(image_path):
        for dirpath,_,filenames in os.walk(image_path):
            for f in filenames:
                path=os.path.abspath(os.path.join(dirpath, f))
                if not utils.is_valid_image(path):
                    print("Warning, please provide a valid image: ", f)
                else:
                    files.append(path)
    else:
        files=[image_path]

    files.sort(key=utils.natural_sort_key)

    output_file=utils.absolute_path(ofile)
    
    if os.path.isdir(output_file):
        print("The provided file is a directory:", output_file)
        sys.exit(0)

    if os.path.exists(output_file):
        os.remove(output_file)

    def extract_and_save(path):
        try:
            X = extract_fn(path, **params)
        except Exception as e:
            print("Cannot extract feactures from", path)
            print(str(e))
            return

        X = X.reshape((1, X.shape[0]))
        lock.acquire()
        with open(output_file, 'a+') as f_handle:
            with open(output_file+".label", 'a+') as f_handle_label:
                numpy.savetxt(f_handle, X)
                f_handle_label.write(os.path.basename(path)+"\n")
        lock.release()

    pool = ThreadPool(cpu_count())
    results = pool.map(extract_and_save, files)
    pool.close()
    pool.terminate()
    pool.join()

    """
    for path in files:
        X = feaext.SRM_extract(path, **params)
        print X.shape
        X = X.reshape((1, X.shape[0]))
        with open(sys.argv[3], 'a+') as f_handle:
            numpy.savetxt(f_handle, X)
    """

    os.chdir(cwd)
# }}}

# {{{ _extract()
def _extract(extractor_name, path, params={}):
    fdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(fdir, os.pardir))
    m_path=os.path.join(basedir, 'external', 'octave')

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
        m_code+="cd "+tmpdir+";"
        m_code+="addpath('"+m_path+"');"
        m_code+="warning('off');"
        m_code+="pkg load image;"
        m_code+="pkg load signal;"
        if extractor_name=="GFR":
            m_code+="data="+extractor_name+"('"+path+"'," \
                    +str(params["rotations"])+", "+str(params["quality"])+", "+str(channel)+");"
        else:
            m_code+="data="+extractor_name+"('"+path+"', "+str(channel)+");"
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
        print("Image mode/extractor not supported: ", im.mode, "/", extractor_name)
        print("")
        sys.stdout.flush()

    im.close()

    return X
# }}}


def SRM_extract(path):
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        C1 = _extract('SRM', path, params={"channel":1})
        C2 = _extract('SRM', path, params={"channel":2})
        C3 = _extract('SRM', path, params={"channel":3})
        X = numpy.hstack((C1, C2, C3))
        return X
    else:
        return  _extract('SRM', path)

def SRMQ1_extract(path):
    return _extract('SRMQ1', path)

def SCRMQ1_extract(path):
    return _extract('SCRMQ1', path)

def GFR_extract(path, quality="auto", rotations=32):
    
    if quality=="auto":
        try:
            p=subprocess.Popen("identify -format '%Q' "+path, \
                               stdout=subprocess.PIPE, shell=True)
            quality, err = p.communicate()
            status = p.wait()
        except:
            quality = 95


    # suppoted qualities
    q = numpy.array([75, 85, 95])
    params = {
        "rotations": rotations,
        "quality": q[numpy.argmin(numpy.abs(q-int(quality)))]
    }
    #print params

    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        params["channel"] = 1
        C1 = _extract('GFR', path, params)
        params["channel"] = 2
        C2 = _extract('GFR', path, params)
        params["channel"] = 3
        C3 = _extract('GFR', path, params)
        X = numpy.hstack((C1, C2, C3))
        return X
    else:
        return  _extract('GFR', path, params)

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

    print("Unknown feature extractor:", name)
    sys.exit(0)


