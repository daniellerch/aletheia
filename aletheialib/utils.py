#!/usr/bin/python

import os
import re
import sys
import zipfile
from PIL import Image
import urllib.request
import urllib.error


EXTERNAL_RESOURCES='https://raw.githubusercontent.com/daniellerch/aletheia-external-resources/main/'

def is_valid_image(path):
 
    try:
        image = Image.open(path)
    except Exception as e:
        print(str(e))
        return False

    try:
        image.verify()
    except Exception as e:
        print(str(e))
        return False

    return True

def absolute_path(path):

    if os.path.isabs(path):
        return path

    return os.path.join(os.getcwd(), path)



def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]    


def which(program):

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def check_bin(cmd):
    import shutil
    if not shutil.which(cmd):
        print("ERROR: you need to install "+cmd+" to run this command!");
        sys.exit(0)


# {{{ download_octave_code()
def download_octave_code(method):

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    cache_dir = os.path.join(basedir, 'aletheia-cache', 'octave')

    remote_octave_file = EXTERNAL_RESOURCES+'octave/code/'+method+'.m'
    remote_license_file = EXTERNAL_RESOURCES+'octave/code/'+method+'.LICENSE'

    local_octave_file = os.path.join(cache_dir, method+'.m')
    local_license_file = os.path.join(cache_dir, method+'.LICENSE')

    # Has the file already been downloaded?
    if os.path.isfile(local_octave_file): 
        return

    # Download the license if available
    os.makedirs(os.path.join(cache_dir, "octave"), exist_ok=True)
    try:
        urllib.request.urlretrieve(remote_license_file, local_license_file)
    except:
        print("Error,", method, "license cannot be downloaded")
        sys.exit(0)


    # Accept license
    print("\nCODE:", method)
    print("\nTo proceed, kindly confirm your acceptance of the license and your")
    print("agreement to download the code from 'aletheia-external-resources'\n")

    print("LICENSE:\n")
    with open(local_license_file, 'r') as f:
        print(f.read())

    r = ""
    while r not in ["y", "n"]:
        r = input("Do you agree? (y/n): ")
        if r == "n":
            print("The terms have not been accepted\n")
            sys.exit(0)
        elif r == "y":
            break

    # Download code
    try:
        urllib.request.urlretrieve(remote_octave_file, local_octave_file)
    except:
        print("Error,", method, "code cannot be downloaded")
        sys.exit(0)

# }}}

# {{{ download_octave_aux_file()
def download_octave_aux_file(fname):

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    cache_dir = os.path.join(basedir, 'aletheia-cache', 'octave')

    remote_octave_file = EXTERNAL_RESOURCES+'octave/code/'+fname
    local_octave_file = os.path.join(cache_dir, fname)

    # Has the file already been downloaded?
    if os.path.isfile(local_octave_file): 
        return

    # Download the file
    os.makedirs(os.path.join(cache_dir, "octave"), exist_ok=True)
    try:
        urllib.request.urlretrieve(remote_octave_file, local_octave_file)
    except:
        print("Error,", fname, "code cannot be downloaded")
        sys.exit(0)

# }}}

# {{{ download_octave_jpeg_toolbox()
def download_octave_jpeg_toolbox():

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    cache_dir = os.path.join(basedir, 'aletheia-cache')

    # Has the JPEG TOOLBOX already been downloaded?
    if os.path.isfile(os.path.join(cache_dir, 'jpeg_toolbox/Makefile')):
        return

    # Download the license if available
    os.makedirs(os.path.join(cache_dir, "jpeg_toolbox"), exist_ok=True)
    licpath = 'jpeg_toolbox/LICENSE'
    remote_license_file = EXTERNAL_RESOURCES+'octave/'+licpath
    local_license_file = os.path.join(cache_dir, licpath)
    try:
        urllib.request.urlretrieve(remote_license_file, local_license_file)
    except Exception as e:
        print("Error, JPEG TOOLBOX license cannot be downloaded")
        sys.exit(0)

    # Accept license
    print("\nTo use this command, you must accept the license of the JPEG TOOLBOX. In that")
    print("case, we will download the JPEG TOOLBOX from 'aletheia-external-resources'.\n")
    with open(local_license_file, 'r') as f:
        print(f.read())

    r = ""
    while r not in ["y", "n"]:
        r = input("Do you accept the license? (y/n): ")
        if r == "n":
            print("The license has not been accepted\n")
            sys.exit(0)
        elif r == "y":
            break


    # Download Mathlab file
    f = 'JPEG_READ_STRUCT.m'
    remote_file = EXTERNAL_RESOURCES+'octave/code/'+f
    local_file = os.path.join(cache_dir, 'octave', f)
    os.makedirs(os.path.join(cache_dir, "octave"), exist_ok=True)
    try:
        urllib.request.urlretrieve(remote_file, local_file)
    except Exception as e:
        print(remote_file)
        print(local_file)
        print("Error, JPEG_READ_STRUCT cannot be downloaded")
        sys.exit(0)


    # Download JPEG TOOLBOX
    for f in ['jpeg_read.c', 'jpeg_write.c', 'Makefile']:
        remote_file = EXTERNAL_RESOURCES+'octave/jpeg_toolbox/'+f
        local_file = os.path.join(cache_dir, 'jpeg_toolbox/'+f)
        try:
            urllib.request.urlretrieve(remote_file, local_file)
        except:
            print("Error,", remote_file, "cannot be downloaded")
            sys.exit(0)


    os.chdir(os.path.join(cache_dir, 'jpeg_toolbox/'))
    os.system("make")




    
# }}}

# {{{ download_F5()
def download_F5():

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    cache_dir = os.path.join(basedir, 'aletheia-cache')

    # Has the F5 already been downloaded and compiled?
    if os.path.isfile(os.path.join(cache_dir, 'F5/Extract.class')):
        return

    print("")
    r = ""
    while r not in ["y", "n"]:
        r = input("Do you agree to download the F5 software from 'aletheia-external-resources'? (y/n): ")
        if r == "n":
            sys.exit(0)
        elif r == "y":
            break

    # Download F5
    remote_file = EXTERNAL_RESOURCES+'steganography-tools/F5.zip'
    local_file = os.path.join(cache_dir, 'F5.zip')
    try:
        urllib.request.urlretrieve(remote_file, local_file)
    except:
        print("Error,", remote_file, "cannot be downloaded")
        sys.exit(0)

    # Unzip
    F5_path = os.path.join(cache_dir, "F5")
    with zipfile.ZipFile(local_file, 'r') as zip_file:
        zip_file.extractall(cache_dir)

    os.chdir(F5_path)
    os.system("make")

    
# }}}

# {{{ download_e4s()
def download_e4s():

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))
    cache_dir = os.path.join(basedir, 'aletheia-cache', 'octave')

    # Has the e4s already been downloaded?
    if os.path.isfile(os.path.join(cache_dir, 'ensemble_training.m')):
        return

    # Download the license if available
    licpath = 'e4s.LICENSE'
    remote_license_file = EXTERNAL_RESOURCES+'octave/code/'+licpath
    local_license_file = os.path.join(cache_dir, licpath)
    try:
        urllib.request.urlretrieve(remote_license_file, local_license_file)
    except Exception as e:
        print("Error, e4s license cannot be downloaded")
        sys.exit(0)

    # Accept license
    print("\nENSEMBLE CLASSIFERS FOR STEGANALYSIS (e4s)\n")
    print("\nTo proceed, kindly confirm your acceptance of the license and your")
    print("agreement to download the code from 'aletheia-external-resources'\n")
    with open(local_license_file, 'r') as f:
        print(f.read())

    r = ""
    while r not in ["y", "n"]:
        r = input("Do you accept the license? (y/n): ")
        if r == "n":
            print("The license has not been accepted\n")
            sys.exit(0)
        elif r == "y":
            break

    # Download
    for f in ['ensemble_fit.m', 'ensemble_predict.m', 'ensemble_testing.m', 'ensemble_training.m']:
        remote_file = EXTERNAL_RESOURCES+'octave/code/'+f
        local_file = os.path.join(cache_dir, f)
        try:
            urllib.request.urlretrieve(remote_file, local_file)
        except:
            print("Error,", remote_file, "cannot be downloaded")
            sys.exit(0)



    
# }}}


