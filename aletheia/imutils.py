#!/usr/bin/python

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

