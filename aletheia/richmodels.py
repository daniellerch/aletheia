import os
import sys
import numpy
import oct2py
import logging
from PIL import Image

# {{{ join_features()
def join_features(data):

    X=numpy.array([])
    for k in data.keys():
        x=numpy.array(data[k])
        x=numpy.reshape(x, x.shape[1])
        X = numpy.hstack((X,x))
    X=numpy.array(X)

    return X
# }}}

# {{{ SRM_extract()
def SRM_extract(path):

    logging.basicConfig(level=logging.INFO)
    #octave = oct2py.Oct2Py(logger=logging.getLogger())
    octave = oct2py.Oct2Py()

    if not os.path.isabs(path):
        path=os.path.join(os.getcwd(), path)

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))

    octave.cd(basedir)
    octave.cd('external')
    octave.cd('octave')

    X=numpy.array([])
    im=Image.open(path)
    if im.mode=='L':
        data=octave.SRM(path, 1)
        X=join_features(data)

    elif im.mode in ['RGB', 'RGBA', 'RGBX']:
        data=octave.SRM(path, 1)
        R=join_features(data)
        data=octave.SRM(path, 2)
        G=join_features(data)
        data=octave.SRM(path, 3)
        B=join_features(data)
        X=numpy.hstack((R,G,B))

    else:
        print "Image mode not supported: ", im.mode
        sys.stdout.flush()

    return X
# }}}

