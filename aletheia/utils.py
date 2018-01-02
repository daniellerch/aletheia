#!/usr/bin/python

import os
import re
from PIL import Image

def is_valid_image(path):
 
    try:
        image = Image.open(path)
    except Exception, e:
        print str(e)
        return False

    try:
        image.verify()
    except Exception, e:
        print str(e)
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

