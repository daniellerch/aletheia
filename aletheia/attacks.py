#!/usr/bin/python

import os
import sys
import ntpath
import tempfile
import subprocess

from aletheia import stegosim

import numpy as np
from scipy import ndimage, misc
from cmath import sqrt

from PIL import Image
from PIL.ExifTags import TAGS

from aletheia.jpeg import JPEG

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count

found = multiprocessing.Value('i', 0)

# -- APPENDED FILES --

def extra_size(filename):
    print("WARNING! not implemented")

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

    if channel!=None:
        I3d = misc.imread(filename)
        width, height, channels = I3d.shape
        I = I3d[:,:,channel]
    else:
        I = misc.imread(filename)
        width, height = I.shape

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
        print("ERROR")
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
    sq = np.sqrt(b**2 - 4*a*c)
    return ( -b + sq ) / ( 2*a ), ( -b - sq ) / ( 2*a )
# }}}

# {{{ smoothness()
def smoothness(I):
    return ( np.sum(np.abs( I[:-1,:] - I[1:,:] )) + 
             np.sum(np.abs( I[:,:-1] - I[:,1:] )) )
# }}}

# {{{ groups()
def groups(I, mask):
    grp=[]
    m, n = I.shape 
    x, y = np.abs(mask).shape
    for i in range(m-x):
        for j in range(n-y):
            grp.append(I[i:(i+x), j:(j+y)])
    return grp
# }}}

# {{{ difference()
def difference(I, mask):
    cmask = - mask
    cmask[(mask > 0)] = 0
    L = []
    for g in groups(I, mask):
        flip = (g + cmask) ^ np.abs(mask) - cmask
        L.append(np.sign(smoothness(flip) - smoothness(g)))
    N = len(L)
    R = float(L.count(1))/N
    S = float(L.count(-1))/N
    return R-S
# }}}

# {{{ rs()
def rs(filename, channel=0):
    I = misc.imread(filename)
    if channel!=None:
        I = I[:,:,channel]
    I = I.astype(int)

    mask = np.array( [[1,0],[0,1]] )
    d0 = difference(I, mask)
    d1 = difference(I^1, mask)

    mask = -mask
    n_d0 = difference(I, mask)
    n_d1 = difference(I^1, mask)

    p0, p1 = solve(2*(d1+d0), (n_d0-n_d1-d1-3*d0), (d0-n_d0)) 
    if np.abs(p0) < np.abs(p1): 
        z = p0
    else: 
        z = p1

    return z / (z-0.5)
# }}}


# -- NAIVE ATTACKS

# {{{ high_pass_filter()
def high_pass_filter(input_image, output_image): 

    I = misc.imread(input_image)
    if len(I.shape)==3:
        kernel = np.array([[[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]],
                           [[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]],
                           [[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]]])
    else:
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])


    If = ndimage.convolve(I, kernel)
    misc.imsave(output_image, If)
# }}}

# {{{ low_pass_filter()
def low_pass_filter(input_image, output_image): 

    I = misc.imread(input_image)
    if len(I.shape)==3:
        kernel = np.array([[[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]])
    else:
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

    kernel = kernel.astype('float32')/9
    If = ndimage.convolve(I, kernel)
    misc.imsave(output_image, If)
# }}}

# {{{ imgdiff()
def imgdiff(image1, image2): 

    I1 = misc.imread(image1).astype('int16')
    I2 = misc.imread(image2).astype('int16')
    np.set_printoptions(threshold=sys.maxsize)

    if len(I1.shape) != len(I2.shape):
        print("Error, both images must have the same dimensionality")
        sys.exit(0)

    if len(I1.shape) == 2:
        D = I1 - I2
        print(D)
    
    elif len(I1.shape) == 3:
        D1 = I1[:,:,0] - I2[:,:,0]
        D2 = I1[:,:,1] - I2[:,:,1]
        D3 = I1[:,:,2] - I2[:,:,2]
        print("Channel 1:")
        print(D1)
        print("Channel 2:")
        print(D2)
        print("Channel 3:")
        print(D3)
    else:
        print("Error, too many dimensions:", I1.shape)



# }}}

# {{{ imgdiff_pixels()
def imgdiff_pixels(image1, image2): 

    I1 = misc.imread(image1).astype('int16')
    I2 = misc.imread(image2).astype('int16')
    np.set_printoptions(threshold=sys.maxsize)

    if len(I1.shape) != len(I2.shape):
        print("Error, both images must have the same dimensionality")
        sys.exit(0)

    if len(I1.shape) == 2:
        D = I1 - I2
        pairs = list(zip(I1.ravel(), D.ravel()))
        print(np.array(pairs, dtype=('i4,i4')).reshape(I1.shape))


    elif len(I1.shape) == 3:
        D1 = I1[:,:,0] - I2[:,:,0]
        D2 = I1[:,:,1] - I2[:,:,1]
        D3 = I1[:,:,2] - I2[:,:,2]
        print("Channel 1:")
        pairs = list(zip(I1[:,:,0].ravel(), D1.ravel()))
        pairs_diff = [p for p in pairs if p[1]!=0]
        #print(np.array(pairs, dtype=('i4,i4')).reshape(I1[:,:,0].shape))
        print(pairs_diff)
        print("Channel 2:")
        pairs = list(zip(I1[:,:,1].ravel(), D2.ravel()))
        pairs_diff = [p for p in pairs if p[1]!=0]
        #print(np.array(pairs, dtype=('i4,i4')).reshape(I1[:,:,1].shape))
        print(pairs_diff)
        print("Channel 3:")
        pairs = list(zip(I1[:,:,2].ravel(), D3.ravel()))
        pairs_diff = [p for p in pairs if p[1]!=0]
        #print(np.array(pairs, dtype=('i4,i4')).reshape(I1[:,:,2].shape))
        print(pairs_diff)
    else:
        print("Error, too many dimensions:", I1.shape)



# }}}

# {{{ print_diffs()
def print_diffs(cover, stego): 

    def print_list(l, ln):
        for i in range(0, len(l), ln):
            print(l[i:i+ln])


    C = misc.imread(cover).astype('int16')
    S = misc.imread(stego).astype('int16')
    np.set_printoptions(threshold=sys.maxsize)

    if len(C.shape) != len(S.shape):
        print("Error, both images must have the same dimensionality")
        sys.exit(0)

    if len(C.shape) == 2:
        D = S - C
        pairs = list(zip(C.ravel(), S.ravel(), D.ravel()))
        pairs_diff = [p for p in pairs if p[2]!=0]
        print_list(pairs_diff, 5)


    elif len(C.shape) == 3:
        D1 = S[:,:,0] - C[:,:,0]
        D2 = S[:,:,1] - C[:,:,1]
        D3 = S[:,:,2] - C[:,:,2]
        print("\nChannel 1:")
        pairs = list(zip(C[:,:,0].ravel(), S[:,:,0].ravel(), D1.ravel()))
        pairs_diff = [p for p in pairs if p[2]!=0]
        print_list(pairs_diff, 5)
        print("\nChannel 2:")
        pairs = list(zip(C[:,:,0].ravel(), S[:,:,0].ravel(), D1.ravel()))
        pairs_diff = [p for p in pairs if p[2]!=0]
        print_list(pairs_diff, 5)
        print("\nChannel 3:")
        pairs = list(zip(C[:,:,0].ravel(), S[:,:,0].ravel(), D1.ravel()))
        pairs_diff = [p for p in pairs if p[2]!=0]
        print_list(pairs_diff, 5)
    else:
        print("Error, too many dimensions:", C.shape)



# }}}

# {{{ print_dct_diffs()
def print_dct_diffs(cover, stego): 

    def print_list(l, ln):
        for i in range(0, len(l), ln):
            print(l[i:i+ln])

    C_jpeg = JPEG(cover)
    S_jpeg = JPEG(stego)
    for i in range(C_jpeg.components()):
        C = C_jpeg.coeffs(i)
        S = S_jpeg.coeffs(i)
        D = S-C
        print("\nChannel "+str(i)+":")
        pairs = list(zip(C.ravel(), S.ravel(), D.ravel()))
        pairs_diff = [p for p in pairs if p[2]!=0]
        print_list(pairs_diff, 5)


# }}}

# {{{ remove_alpha_channel()
def remove_alpha_channel(input_image, output_image): 

    I = misc.imread(input_image)
    I[:,:,3] = 255;
    misc.imsave(output_image, I)
# }}}

# {{{ brute_force()
def brute_force(command, password_file):
    
    with open(password_file, "rU") as f:
        passwords = f.readlines()

    class PasswordFound(Exception): 
        pass

    n_proc = cpu_count()
    print("Using", n_proc, "processes")
    pool = ThreadPool(n_proc)

    def crack(passw):
        if found.value == 1:
            return False

        FNUL = open(os.devnull, 'w')
        cmd = command.replace("<PASSWORD>", passw.replace("\n", ""))
        #p=subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        p=subprocess.Popen(cmd, stdout=FNUL, stderr=FNUL, shell=True)
        #output, err = p.communicate()
        status = p.wait()
        if p.returncode==0:
            print("\nPassword found:", passw)
            with found.get_lock():
                found.value = 1
                return True



    # Process thread pool in batches
    batch=1000
    for i in range(0, len(passwords), batch):
        perc = round(100*float(i)/len(passwords),2)
        sys.stdout.write("Completed: "+str(perc)+'%    \r')
        passw_batch = passwords[i:i+batch]
        pool = ThreadPool(n_proc)
        results = pool.map(crack, passw_batch)
        pool.close()
        pool.terminate()
        pool.join()
        if any(results):
            break

# }}}



