import os
import sys
import scipy
import logging
import numpy
import tempfile
import shutil
import numpy
import subprocess
import random

from aletheialib import utils
from aletheialib.octave_interface import _jpeg



class JPEG():
    def __init__(self, path):
        utils.download_octave_jpeg_toolbox()
        self.data = _jpeg('jpeg_read_struct', path)
        #print(self.data)
        #numpy.set_printoptions(threshold=sys.maxsize)
        #print(self.data[0][0][7][0].shape)
        #print(self.data[0][0][7][0])
        #print(self.data[0][0][7][0][0])
        #print(self.data[0][0][7][0][1])
        #sys.exit(0)

    def size(self):
        return (self.data[0][0][0][0][0], self.data[0][0][1][0][0])

    def coeffs(self, channel):
        return self.data[0][0][7][0][channel]

    def components(self):
        return int(self.data[0][0][4][0][0])


    #    "image_width": struct[0][0][0][0][0],
    #    "image_height": struct[0][0][1][0][0],
    #    "image_components": struct[0][0][2][0][0],
    #    "image_color_space": struct[0][0][3][0][0],
    #    "jpeg_components": struct[0][0][4][0][0],
    #    "jpeg_color_space": struct[0][0][5][0][0],
    #    "comments": struct[0][0][6],
    #    "coff_arrays": struct[0][0][7][0][0],
    #    "quant_tables": struct[0][0][8][0][0],
    #    "ac_huff_tables": struct[0][0][9][0][0],
    #    "dc_huff_tables": struct[0][0][10][0][0],
    #    "optimize_coding": struct[0][0][11][0][0],
    #    "comp_info": struct[0][0][12][0][0],
    #    "progressive_mode": struct[0][0][13][0][0]





