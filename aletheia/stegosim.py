import os
import oct2py
import PIL
import logging

# {{{ embed()
def embed(sim, path, payload):
    
    logging.basicConfig(level=logging.INFO)
    #octave = oct2py.Oct2Py(logger=logging.getLogger())
    octave = oct2py.Oct2Py()

    if not os.path.isabs(path):
        path=os.path.join(os.getcwd(), path)

    im=PIL.Image.open(path)
    if im.mode!='L':
        print "Image mode not supported: ", im.mode
        sys.stdout.flush()

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir))

    octave.cd(basedir)
    octave.cd('external')
    octave.cd('octave')

    octave.eval('pkg load image')
    
    if sim=='wow':
        X,_=octave.WOW(path, payload)
    elif sim=='s_uniward':
        X=octave.S_UNIWARD(path, payload)
    elif sim=='hugo':
        X,_=octave.HUGO(path, payload)
    else:
        print "Unknown simulator: ", sim
        sys.stdout.flush()

    return X
# }}}

def wow(path, payload):
    return embed('wow', path, payload)
       
def s_uniward(path, payload):
    return embed('s_uniward', path, payload)

def hugo(path, payload):
    return embed('hugo', path, payload)



