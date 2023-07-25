import os
import sys
import numpy
import logging
import tempfile
import shutil
import subprocess
from PIL import Image
from scipy.io import loadmat

from aletheialib import utils
from aletheialib.octave_interface import _extract


import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count

lock = multiprocessing.Lock()




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
        print("Output file already exists! cotinue ...")
        with open(output_file+".label", 'r') as f:
            labels = [os.path.join(image_path, x) for x in f.read().splitlines()]

        pending_files = [x for x in files if x not in labels]
        files = pending_files
        print("Pending files:", len(files))

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
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        C1 = _extract('SRMQ1', path, params={"channel":1})
        C2 = _extract('SRMQ1', path, params={"channel":2})
        C3 = _extract('SRMQ1', path, params={"channel":3})
        X = numpy.hstack((C1, C2, C3))
        return X
    else:
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


def DCTR_extract(path, quality="auto"):
    
    if quality=="auto":
        try:
            p=subprocess.Popen("identify -format '%Q' "+path, \
                               stdout=subprocess.PIPE, shell=True)
            quality, err = p.communicate()
            status = p.wait()
        except:
            quality = 95


    # suppoted qualities
    #q = numpy.array([75, 85, 95])
    params = {
        #"quality": q[numpy.argmin(numpy.abs(q-int(quality)))]
        "quality": int(quality)
    }
    #print params

    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        params["channel"] = 1
        C1 = _extract('DCTR', path, params)
        params["channel"] = 2
        C2 = _extract('DCTR', path, params)
        params["channel"] = 3
        C3 = _extract('DCTR', path, params)
        X = numpy.hstack((C1, C2, C3))
        return X
    else:
        return  _extract('DCTR', path, params)





def extractor_fn(name):
    if name == "srm": 
        return SRM_extract
    if name == "srmq1": 
        return SRMQ1_extract
    if name == "scrmq1": 
        return SCRMQ1_extract
    if name == "gfr": 
        return GFR_extract
    if name == "dctr": 
        return dctr_extract

    print("Unknown feature extractor:", name)
    sys.exit(0)


