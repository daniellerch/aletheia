#!/usr/bin/python

import os
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



