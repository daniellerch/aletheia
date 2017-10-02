import os
import oct2py
import PIL

def wow(path, payload):
    
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
    X, distortion=octave.WOW(path, payload)

    return X
