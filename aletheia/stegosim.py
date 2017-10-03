import os
import sys
import oct2py
import PIL
import logging
import numpy

class StegoSim:

    def __init__(self):
        currdir=os.path.dirname(__file__)
        basedir=os.path.abspath(os.path.join(currdir, os.pardir))
        path=os.path.join(basedir, 'external', 'octave')

        #octave = oct2py.Oct2Py(logger=logging.getLogger())
        self.octave = oct2py.Oct2Py()
        self.octave.addpath(path)


    # {{{ embed()
    def embed(self, sim, path, payload):
        
        if not os.path.isabs(path):
            path=os.path.join(os.getcwd(), path)

        im=PIL.Image.open(path)
        if im.mode!='L':
            print "Image mode not supported: ", im.mode
            sys.stdout.flush()
        im.close()

        X=numpy.array([])
        #logging.basicConfig(level=logging.INFO)

        
        self.octave.eval('pkg load image')
     
        if sim=='wow':
            X,_=self.octave.WOW(path, payload)
        elif sim=='s_uniward':
            X=self.octave.S_UNIWARD(path, payload)
        elif sim=='hugo':
            X,_=self.octave.HUGO(path, payload)
        else:
            print "Unknown simulator: ", sim
            sys.stdout.flush()
        
        return X
    # }}}


def wow(path, payload):
    ss=StegoSim()
    return ss.embed('wow', path, payload)
       
def s_uniward(path, payload):
    ss=StegoSim()
    return ss.embed('s_uniward', path, payload)

def hugo(path, payload):
    ss=StegoSim()
    return ss.embed('hugo', path, payload)





