import os
import numpy
import oct2py


def SRM_extract(path):

    octave = oct2py.Oct2Py()

    if not os.path.isabs(path):
        path=os.path.join(os.getcwd(), path)

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))

    octave.cd(basedir)
    octave.cd('external')
    octave.cd('octave')

    data=octave.SRM(path)
    X=numpy.array([])
    for k in data.keys():
        x=numpy.array(data[k])
        x=numpy.reshape(x, x.shape[1])
        X = numpy.hstack((X,x))
    X=numpy.array(X)

    return X
