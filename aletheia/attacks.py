#!/usr/bin/python

import sys
from scipy import ndimage, misc
from cmath import sqrt

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


