#!/usr/bin/python

import sys
import numpy
import ntpath
import tempfile
from scipy import ndimage, misc
from cmath import sqrt

from PIL import Image
from PIL.ExifTags import TAGS

# -- APPENDED FILES --

def extra_size(filename):
    print "WARNING! not implemented"

    name=ntpath.basename(filename)
    I = misc.imread(filename)

    misc.imsave(tempfile.gettempdir()+'/'+name, I)
    return 0


# -- EXIF --

# {{{ exif()
def exif(filename):
    image = Image.open(filename)
    try:
        exif = { TAGS[k]: v for k, v in image._getexif().items() if k in TAGS }
        return exif

    except AttributeError:
        return {}
# }}}


# -- SAMPLE PAIR ATTACK --

# {{{ spa()
"""
    Sample Pair Analysis attack. 
    Return Beta, the detected embedding rate.
"""
def spa(filename, channel=0): 

    I3d = misc.imread(filename)
    width, height, channels = I3d.shape

    I = I3d[:,:,channel]

    x=0; y=0; k=0
    for j in range(height):
        for i in range(width-1):
            r = I[i, j]
            s = I[i+1, j]
            if (s%2==0 and r<s) or (s%2==1 and r>s):
                x+=1
            if (s%2==0 and r>s) or (s%2==1 and r<s):
                y+=1
            if round(s/2)==round(r/2):
                k+=1

    if k==0:
        print "ERROR"
        sys.exit(0)

    a=2*k
    b=2*(2*x-width*(height-1))
    c=y-x

    bp=(-b+sqrt(b**2-4*a*c))/(2*a)
    bm=(-b-sqrt(b**2-4*a*c))/(2*a)

    beta=min(bp.real, bm.real)
    return beta
# }}}


# -- RS ATTACK --

# {{{ solve()
def solve(a, b, c):
    sq = numpy.sqrt(b**2 - 4*a*c)
    return ( -b + sq ) / ( 2*a ), ( -b - sq ) / ( 2*a )
# }}}

# {{{ smoothness()
def smoothness(I):
    return ( numpy.sum(numpy.abs( I[:-1,:] - I[1:,:] )) + 
             numpy.sum(numpy.abs( I[:,:-1] - I[:,1:] )) )
# }}}

# {{{ groups()
def groups(I, mask):
    grp=[]
    m, n = I.shape 
    x, y = numpy.abs(mask).shape
    for i in xrange(m-x):
        for j in xrange(n-y):
            grp.append(I[i:(i+x), j:(j+y)])
    return grp
# }}}

# {{{ difference()
def difference(I, mask):
    cmask = - mask
    cmask[(mask > 0)] = 0
    L = []
    for g in groups(I, mask):
        flip = (g + cmask) ^ numpy.abs(mask) - cmask
        L.append(numpy.sign(smoothness(flip) - smoothness(g)))
    N = len(L)
    R = float(L.count(1))/N
    S = float(L.count(-1))/N
    return R-S
# }}}

# {{{ rs()
def rs(filename, channel=0):
    I = misc.imread(filename)
    I = I[:,:,channel]
    I = I.astype(int)

    mask = numpy.array( [[1,0],[0,1]] )
    d0 = difference(I, mask)
    d1 = difference(I^1, mask)

    mask = -mask
    n_d0 = difference(I, mask)
    n_d1 = difference(I^1, mask)

    p0, p1 = solve(2*(d1+d0), (n_d0-n_d1-d1-3*d0), (d0-n_d0)) 
    if numpy.abs(p0) < numpy.abs(p1): 
        z = p0
    else: 
        z = p1

    return z / (z-0.5)
# }}}






