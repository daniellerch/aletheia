import os
import sys
import scipy
import logging
import numpy
import string
import tempfile
import shutil
import numpy
import subprocess
import random

import numpy as np

from scipy.io import savemat, loadmat
from PIL import Image

from aletheialib import utils
from aletheialib import octave_interface 

import multiprocessing
#from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import Pool as ThreadPool 
from multiprocessing import cpu_count

from imageio import imread, imwrite


# {{{ embed_message()

def embed(params):
    i, path, output_dir, payload, dst_path, embed_fn_saving, embed_fn = params
    try:
        basename=os.path.basename(path)
        dst_path=os.path.join(output_dir, basename)
        numpy.random.seed(i)

        if embed_fn_saving:
            if "-" in payload:
                rng = payload.split('-')
                rini = float(rng[0])
                rend = float(rng[1])
                rnd_payload = numpy.random.uniform(rini, rend)
                #print("rnd_payload:", rnd_payload)
                embed_fn(path, rnd_payload, dst_path)
            else:
                embed_fn(path, payload, dst_path)
        else:
            if "-" in payload:
                rng = payload.split('-')
                rini = float(rng[0])
                rend = float(rng[1])
                rnd_payload = numpy.random.uniform(rini, rend)
                #print("rnd_payload:", rnd_payload)
                X=embed_fn(path, rnd_payload)
            else:
                X=embed_fn(path, payload)
            try:
                imwrite(dst_path, X.astype('uint8'))
            except Exception as e:
                print(str(e))

    except Exception as e:
        print(str(e))


def embed_message(embed_fn, path, payload, output_dir, 
                  embed_fn_saving=False, show_debug_info=True):

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
                    if show_debug_info:
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
        if os.path.exists(dst_path) and os.path.getsize(dst_path)>0:
            if show_debug_info:
                print("Warning! file already exists, ignored:", dst_path)
            continue
        filtered_files.append(f)
    files = filtered_files
    del filtered_files


    params = []
    i = 0
    for f in files:
        params.append( (i, f, output_dir, payload, dst_path, embed_fn_saving, embed_fn) )
        i += 1


    # Process thread pool in batches
    batch=1000
    for i in range(0, len(params), batch):
        params_batch = params[i:i+batch]
        n_core=cpu_count()
        if show_debug_info:
            print("Using", n_core, "threads")
        pool = ThreadPool(n_core)
        results = pool.map(embed, params_batch)
        pool.close()
        pool.terminate()
        pool.join()


# }}}




def wow(path, payload):
    return octave_interface.embed('wow', path, payload)

def s_uniward(path, payload):
    return octave_interface.embed('s_uniward', path, payload)

def s_uniward_color(path, payload):
    return octave_interface.embed('s_uniward_color', path, payload)

def j_uniward(path, payload, dst_path):
    return octave_interface.embed('j_uniward', path, payload, dst_path)

def j_uniward_color(path, payload, dst_path):
    return octave_interface.embed('j_uniward_color', path, payload, dst_path)

def hugo(path, payload):
    return octave_interface.embed('hugo', path, payload)

def hill(path, payload):
    return octave_interface.embed('hill', path, payload)

def hill_color(path, payload):
    return octave_interface.embed('hill_color', path, payload)

def ebs(path, payload, dst_path):
    return octave_interface.embed('ebs', path, payload, dst_path)

def ebs_color(path, payload, dst_path):
    return octave_interface.embed('ebs_color', path, payload, dst_path)

def ued(path, payload, dst_path):
    return octave_interface.embed('ued', path, payload, dst_path)

def ued_color(path, payload, dst_path):
    return octave_interface.embed('ued_color', path, payload, dst_path)

def nsf5(path, payload, dst_path):
    return octave_interface.embed('nsf5', path, payload, dst_path)

def nsf5_color(path, payload, dst_path):
    return octave_interface.embed('nsf5_color', path, payload, dst_path)

def experimental(path, payload):
    return octave_interface.embed('experimental', path, payload)

def custom(path, command, dst_path):
    bn = os.path.basename(path)
    shutil.copyfile(path, dst_path)
    cmd = command.replace("<IMAGE>", dst_path)
    FNUL = open(os.devnull, 'w')
    p=subprocess.Popen(cmd, stdout=FNUL, stderr=FNUL, shell=True)
    #output, err = p.communicate()
    status = p.wait()



# {{{ lsbm()
def lsbm(path, payload):
    s = int.from_bytes(os.urandom(4), 'big')
    numpy.random.seed(s)
    payload = float(payload)
    X = imread(path).astype('int16')
    Z = X.copy()
    prob = np.random.uniform(low=0., high=1, size=X.shape)
    msg = np.random.randint(0, 2, size=X.shape).astype('int16')
    sign = np.random.choice([-1, 1], size=X.shape).astype('int16')
    sign[X%2==msg] = 0
    sign[(X==0)&(sign==-1)] = 1
    sign[(X==255)&(sign==1)] = -1
    X[prob<payload] += sign[prob<payload]
    return X.astype('uint8')
# }}}

# {{{ lsbr()
def lsbr(path, payload):
    s = int.from_bytes(os.urandom(4), 'big')
    numpy.random.seed(s)
    payload = float(payload)
    X = imread(path)
    Z = X.copy()
    prob = np.random.uniform(low=0., high=1, size=X.shape)
    msg = np.random.randint(0, 2, size=X.shape).astype('uint8')
    X[prob<payload] = X[prob<payload] - X[prob<payload]%2 + msg[prob<payload]
    return X
# }}}

# {{{ steghide()
def steghide(path, payload, dst_path):

    p=subprocess.Popen("LANG=en_US && steghide info "+path, \
                       shell=True,  
                       stdout=subprocess.PIPE,   
                       stdin=subprocess.PIPE, 
                       stderr=subprocess.DEVNULL) 
    output, err = p.communicate(input=b'n')   
    status = p.wait() 
    output = output.decode()   
 
    found = False
    capacity = 0.0
    for line in output.splitlines():
        if "capacity" in line:  
            m = 1 
            line = line.replace("capacity: ", "") 
            if "KB" in line: 
                line = line.replace("KB", "") 
                m = 1024  
            elif "MB" in line:  
                line = line.replace("MB", "") 
                m = 1024*1024 
            elif "Byte" in line: 
                line = line.replace("Byte", "")  
     
            capacity = int(float(line)*m) 
            found = True
            break
    if not found:
        print("ERROR: can not get capacity for", path)
        sys.exit(0)

    nbytes = int(capacity*float(payload));

    password = ''.join(random.sample(string.ascii_letters+string.digits, 8))

    with open("/tmp/secret-"+password+".data", "wb") as secret:
        secret.write(os.urandom(nbytes))


    cmd = "steghide embed -cf "+path+" -ef /tmp/secret-"+password+".data -sf " \
          +dst_path+" -p "+password+" -Z -q"
    os.system(cmd)                                                           
    os.remove("/tmp/secret-"+password+".data")  

# }}}

# {{{ outguess()
def outguess(path, payload, dst_path):

    # XXX: We use steghide to estimate the capacity, this is not a good
    # option but Outguess does not provide this feature.
    p=subprocess.Popen("LANG=en_US && steghide info "+path, \
                       shell=True,  
                       stdout=subprocess.PIPE,   
                       stdin=subprocess.PIPE, 
                       stderr=subprocess.DEVNULL) 
    output, err = p.communicate(input=b'n')   
    status = p.wait() 
    output = output.decode()   
 
    found = False
    capacity = 0.0
    for line in output.splitlines():
        if "capacity" in line:  
            m = 1 
            line = line.replace("capacity: ", "") 
            if "KB" in line: 
                line = line.replace("KB", "") 
                m = 1024  
            elif "MB" in line:  
                line = line.replace("MB", "") 
                m = 1024*1024 
            elif "Byte" in line: 
                line = line.replace("Byte", "")  
     
            capacity = int(float(line)*m) 
            found = True
            break

    if not found:
        print("ERROR: can not get capacity for", path)
        sys.exit(0)


    password = ''.join(random.sample(string.ascii_letters+string.digits, 8))


    # Outguess fails frequently, so we make several attempts while reducing 
    # the payload
    for i in range(100): 
        nbytes = int(capacity*float(payload));
        payload *= 0.9
        with open("/tmp/secret-"+password+".data", "wb") as secret:
            secret.write(os.urandom(nbytes))
        cmd = "outguess -k "+password+" -d "+"/tmp/secret-"+password+".data " \
              +path+" "+dst_path+" 2>/dev/null"
        exit_status = os.WEXITSTATUS(os.system(cmd))
        os.remove("/tmp/secret-"+password+".data")  
        #if exit_status == 0:
        #    break
        if os.path.exists(dst_path) and os.path.getsize(dst_path)>0:
            return
    print("WARNING: Outguess embedding failed:", path)

# }}}

# {{{ steganogan()
def steganogan(path, payload, dst_path):

    I = imread(path)
    capacity = 1
    for m in I.shape:
        capacity *= m
    nbytes = int(capacity*float(payload)/8);
    #print("nbytes:", nbytes)

    msg = ''.join(random.choice(string.ascii_letters+string.digits) for i in range(nbytes))
    cmd = "steganogan encode --cpu "+path+" "+msg+" -o "+dst_path
    #cmd = "steganogan encode "+path+" "+msg+" -o "+dst_path
    os.system(cmd)                                                           

# }}}


# {{{ embedding_fn()
def embedding_fn(name):
    if name=="lsbr-sim":
        return lsbr
    if name=="lsbm-sim":
        return lsbm
    if name=="hugo-sim":
        return hugo
    if name=="wow-sim":
        return wow
    if name=="s-uniward-sim":
        return s_uniward
    if name=="s-uniward-color-sim":
        return s_uniward_color
    if name=="j-uniward-sim":
        return j_uniward
    if name=="j-uniward-color-sim":
        return j_uniward_color
    if name=="hill-sim":
        return hill
    if name=="hill-color-sim":
        return hill_color
    if name=="ebs-sim":
        return ebs
    if name=="ebs-color-sim":
        return ebs_color
    if name=="ued-sim":
        return ued
    if name=="ueb-color-sim":
        return ued_color
    if name=="nsf5-sim":
        return nsf5
    if name=="nsf5-color-sim":
        return nsf5_color
    if name=="steghide-sim":
        return steghide
    if name=="outguess-sim":
        return outguess
    if name=="steganogan-sim":
        return steganogan
    #print(f"|{name}|")
    print("Unknown simulator:", name)
    sys.exit(0)
# }}}






