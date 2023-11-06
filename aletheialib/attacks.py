#!/usr/bin/python

import os
import sys
import shutil
import magic
import ntpath
import tempfile
import subprocess

from io import BytesIO
from aletheialib import stegosim, utils

import numpy as np
import scipy.stats
from scipy import ndimage
from cmath import sqrt
from imageio import imread, imsave

from PIL import Image
from PIL.ExifTags import TAGS

from aletheialib.jpeg import JPEG

import multiprocessing
#from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ThreadPool 
from multiprocessing import cpu_count
from multiprocessing import Pool



# -- EXIF --

# {{{ exif()
def exif(filename):
    image = Image.open(filename)

    exifdata = image.getexif()
    for tag_id in exifdata:
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        if isinstance(data, bytes):
            data = data.decode()
        print(f"{tag:25}: {data}")


# }}}$


# -- SAMPLE PAIR ATTACK --

# {{{ spa()
"""
    Sample Pair Analysis attack. 
    Return Beta, the detected embedding rate.
"""
def spa(filename, channel=0): 
    return spa_image(imread(filename), channel)

def spa_image(image, channel=0):
    if channel!=None:
        width, height, channels = image.shape
        I = image[:,:,channel]
    else:
        I = image
        width, height = I.shape

    r = I[:-1,:]
    s = I[1:,:]

    # we only care about the lsb of the next pixel
    lsb_is_zero = np.equal(np.bitwise_and(s, 1), 0)
    lsb_non_zero = np.bitwise_and(s, 1)
    msb = np.bitwise_and(I, 0xFE)

    r_less_than_s = np.less(r, s)
    r_greater_than_s = np.greater(r, s)

    x = np.sum(np.logical_or(np.logical_and(lsb_is_zero, r_less_than_s),
                             np.logical_and(lsb_non_zero, r_greater_than_s)).astype(int))

    y = np.sum(np.logical_or(np.logical_and(lsb_is_zero, r_greater_than_s),
                             np.logical_and(lsb_non_zero, r_less_than_s)).astype(int))

    k = np.sum(np.equal(msb[:-1,:], msb[1:,:]).astype(int))

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
            yield I[i:(i+x), j:(j+y)]
# }}}

class RSAnalysis(object):
    def __init__(self, mask):
        self.mask = mask
        self.cmask = - mask
        self.cmask[(mask > 0)] = 0
        self.abs_mask = np.abs(mask)

    def call(self, group):
        flip = (group + self.cmask) ^ self.abs_mask - self.cmask
        return  np.sign(smoothness(flip) - smoothness(group))


# {{{ difference()
def difference(I, mask):
    pool = Pool(multiprocessing.cpu_count())
    analysis = pool.map(RSAnalysis(mask).call, groups(I, mask))
    pool.close()
    pool.join()

    counts = [0, 0, 0]
    for v in analysis:
        counts[v] += 1

    N = sum(counts)
    R = float(counts[1])/N
    S = float(counts[-1])/N
    return R-S
# }}}

# {{{ rs()
def rs(filename, channel=0):
    return rs_image(np.asarray(imread(filename), channel))

def rs_image(image, channel=0):
    if channel!=None:
        I = image[:,:,channel]
    else:
        I = image
    I = I.astype(int)

    mask = np.array( [[0, 0, 0], [0, 1, 0], [0, 0, 0]] )
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

# -- CALIBRATION --

# {{{ calibration()
def H_i(dct, k, l, i):
    dct_kl = dct[k::8,l::8].flatten()
    return sum(np.abs(dct_kl) == i)

def H_i_all(dct, i):
    dct_kl = dct.flatten()
    return sum(np.abs(dct_kl) == i)


def beta_kl(dct_0, dct_b, k, l):
    h00 = H_i(dct_0, k, l, 0)
    h01 = H_i(dct_0, k, l, 1)
    h02 = H_i(dct_0, k, l, 2)
    hb0 = H_i(dct_b, k, l, 0)
    hb1 = H_i(dct_b, k, l, 1)
    return (h01*(hb0-h00) + (hb1-h01)*(h02-h01)) / (h01**2 + (h02-h01)**2)


def calibration_f5(path):
    """ it uses jpeg_toolbox """
    import jpeg_toolbox as jt

    tmpdir = tempfile.mkdtemp()
    predfile = os.path.join(tmpdir, 'img.jpg')
    os.system("convert -chop 4x4 "+path+" "+predfile)
    im_jpeg = jt.load(path)
    impred_jpeg = jt.load(predfile)
    shutil.rmtree(tmpdir)

    beta_list = []
    for i in range(im_jpeg["jpeg_components"]):
        dct_b = im_jpeg["coef_arrays"][i]
        dct_0 = impred_jpeg["coef_arrays"][i]
        b01 = beta_kl(dct_0, dct_b, 0, 1)   
        b10 = beta_kl(dct_0, dct_b, 1, 0)   
        b11 = beta_kl(dct_0, dct_b, 1, 1)
        beta = (b01+b10+b11)/3
        if beta > 0.05:
            print("Hidden data found in channel "+str(i)+":", beta)
        else:
            print("No hidden data found in channel "+str(i))



def calibration_chisquare_mode(path):
    """ it uses jpeg_toolbox """
    import jpeg_toolbox as jt

    tmpdir = tempfile.mkdtemp()
    predfile = os.path.join(tmpdir, 'img.jpg')
    os.system("convert -chop 4x4 "+path+" "+predfile)
    im_jpeg = jt.load(path)
    impred_jpeg = jt.load(predfile)
    shutil.rmtree(tmpdir)

    beta_list = []
    for i in range(im_jpeg["jpeg_components"]):
        dct = im_jpeg["coef_arrays"][i]
        dct_estim = impred_jpeg["coef_arrays"][i]
        
        p_list = []
        for k in range(4):
            for l in range(4):
                if (k, l) == (0, 0):
                    continue

                f_exp, f_obs = [], []
                for j in range(5):
                    h  = H_i(dct, k, l, j)
                    h_estim = H_i(dct_estim, k, l, j)
                    if h<5 or h_estim<5:
                        break
                    f_exp.append(h_estim)
                    f_obs.append(h)
                #print(f_exp, f_obs)

                chi, p = scipy.stats.chisquare(f_obs, f_exp)
                p_list.append(p)

        p = np.mean(p_list)
        if p < 0.05:
            print("Hidden data found in channel "+str(i)+". p-value:", np.round(p, 6))
        else:
            print("No hidden data found in channel "+str(i))




def calibration_f5_octave_jpeg(filename, return_result=False):
    """ It uses JPEG from octave """
    tmpdir = tempfile.mkdtemp()
    predfile = os.path.join(tmpdir, 'img.jpg')
    os.system("convert -chop 4x4 "+filename+" "+predfile)

    im_jpeg = JPEG(filename)
    impred_jpeg = JPEG(predfile)
    beta_avg = 0.0
    for i in range(im_jpeg.components()):
        dct_b = im_jpeg.coeffs(i)
        dct_0 = impred_jpeg.coeffs(i)
        b01 = beta_kl(dct_0, dct_b, 0, 1)   
        b10 = beta_kl(dct_0, dct_b, 1, 0)   
        b11 = beta_kl(dct_0, dct_b, 1, 1)
        beta = (b01+b10+b11)/3
        if beta > 0.05:
            beta_avg += beta
            if not return_result:
                print("Hidden data found in channel "+str(i)+":", beta)
        else:
            if not return_result:
                print("No hidden data found in channel "+str(i))
    beta_avg /= im_jpeg.components()
    if return_result:
        return beta_avg

    shutil.rmtree(tmpdir)

# }}}


# -- NAIVE ATTACKS

# {{{ high_pass_filter()
def high_pass_filter(input_image, output_image): 

    I = imread(input_image)
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
    imsave(output_image, If)
# }}}

# {{{ low_pass_filter()
def low_pass_filter(input_image, output_image): 

    I = imread(input_image)
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
    imsave(output_image, If)
# }}}

# {{{ imgdiff()
def imgdiff(image1, image2): 

    I1 = imread(image1).astype('int16')
    I2 = imread(image2).astype('int16')
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

    I1 = imread(image1).astype('int16')
    I2 = imread(image2).astype('int16')
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


    C = imread(cover).astype('int16')
    S = imread(stego).astype('int16')
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
        pairs = list(zip(C[:,:,1].ravel(), S[:,:,1].ravel(), D2.ravel()))
        pairs_diff = [p for p in pairs if p[2]!=0]
        print_list(pairs_diff, 5)
        print("\nChannel 3:")
        pairs = list(zip(C[:,:,2].ravel(), S[:,:,2].ravel(), D3.ravel()))
        pairs_diff = [p for p in pairs if p[2]!=0]
        print_list(pairs_diff, 5)
    else:
        print("Error, too many dimensions:", C.shape)



# }}}

# {{{ print_dct_diffs()
def print_dct_diffs(cover, stego): 
    #import jpeg_toolbox as jt

    def print_list(l, ln):
        mooc = 0
        for i in range(0, len(l), ln):
            print(l[i:i+ln])
            v = l[i:i+ln][0][2]
            if np.abs(v) > 1:
                mooc += 1

    C_jpeg = JPEG(cover)
    #C_jpeg = jt.load(cover)
    S_jpeg = JPEG(stego)
    #S_jpeg = jt.load(stego)
    for i in range(C_jpeg.components()):
    #for i in range(C_jpeg["jpeg_components"]):
        C = C_jpeg.coeffs(i)
        S = S_jpeg.coeffs(i)
        #C = C_jpeg["coef_arrays"][i]
        #S = S_jpeg["coef_arrays"][i]
        if C.shape!=S.shape:
            print("WARNING! channels with different size. Channel: ", i)
            continue
        D = S-C
        print("\nChannel "+str(i)+":")
        pairs = list(zip(C.ravel(), S.ravel(), D.ravel()))
        pairs_diff = [p for p in pairs if p[2]!=0]
        print_list(pairs_diff, 5)


    print("\nCommon DCT coefficients frequency variation:")
    for i in range(C_jpeg.components()):
        print("\nChannel "+str(i)+":")
        nz_coeffs = np.count_nonzero(C_jpeg.coeffs(i))
        changes = np.sum(np.abs(C_jpeg.coeffs(i)-S_jpeg.coeffs(i)))
        rate = round(changes/nz_coeffs, 4)
        print(f"non zero coeffs: {nz_coeffs}, changes: {changes}, rate: {rate}")
        print("H BAR    COVER      STEGO       DIFF")
        print("------------------------------------")
        for v in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            cover = np.sum(C_jpeg.coeffs(i)==v)
            stego = np.sum(S_jpeg.coeffs(i)==v)
            var = stego-cover
            print(f"{v:+}: {cover:10d} {stego:10d} {var:10d}")


# }}}

# {{{ remove_alpha_channel()
def remove_alpha_channel(input_image, output_image): 

    I = imread(input_image)
    I[:,:,3] = 255;
    imsave(output_image, I)
# }}}

# {{{ eof_extract()
def eof_extract(input_image, output_data): 

    name, ext = os.path.splitext(input_image)

    eof = None
    if ext.lower() in [".jpeg", ".jpg"]:
        eof = b"\xFF\xD9"
    elif ext.lower() in [".gif"]:
        eof = b"\x00\x3B"
    elif ext.lower() in [".png"]:
        eof = b"\x49\x45\x4E\x44\xAE\x42\x60\x82"
    else:
        print("Please provide a JPG, GIF or PNG file")
        sys.exit(0)

    raw = open(input_image, 'rb').read()
    buff = BytesIO()
    buff.write(raw)
    buff.seek(0)
    bytesarray = buff.read()
    data = bytesarray.rsplit(eof, 1) # last occurrence

    # data[0] contains the original image
    if len(data[1])==0:
        print("No data found")
        sys.exit(0)
    with open(output_data, 'wb') as outf:
        outf.write(data[1])

    ft = magic.Magic(mime=True).from_file(output_data)
    print("\nData extracted from", input_image, "to", output_data, "("+ft+")\n")

# }}}



