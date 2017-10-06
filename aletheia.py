#!/usr/bin/python

import sys
import json
import os
import scipy
import numpy
import multiprocessing

from aletheia import stegosim, richmodels
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count

from aletheia import attacks, utils
#from cnn import net as cnn

lock = multiprocessing.Lock()


# {{{ embed_message()
def embed_message(embed_fn, path, payload, output_dir):

    # Read filenames
    files=[]
    if os.path.isdir(path):
        for dirpath,_,filenames in os.walk(path):
            for f in filenames:
                path=os.path.abspath(os.path.join(dirpath, f))
                if not utils.is_valid_image(path):
                    print "Warning, prease provide a valid image: ", f
                else:
                    files.append(path)
    else:
        files=[path]
    
    def embed(path):
        X=embed_fn(path, payload)
        basename=os.path.basename(path)
        dst_path=os.path.join(output_dir, basename)
        try:
            scipy.misc.toimage(X, cmin=0, cmax=255).save(dst_path)
        except Exception, e:
            print str(e)

    # Process thread pool in batches
    batch=1000
    for i in xrange(0, len(files), batch):
        files_batch = files[i:i+batch]
        pool = ThreadPool(cpu_count())
        results = pool.map(embed, files_batch)
        pool.close()
        pool.terminate()
        pool.join()

    """
    for path in files:
        I=scipy.misc.imread(path)
        X=embed_fn(path, payload)
        basename=os.path.basename(path)
        dst_path=os.path.join(output_dir, basename)
        try:
            scipy.misc.toimage(X, cmin=0, cmax=255).save(dst_path)
        except Exception, e:
            print str(e)
    """
   
# }}}

# {{{ extract_features()
def extract_features(extract_fn, image_path, ofile):

    image_path=utils.absolute_path(image_path)

    # Read filenames
    files=[]
    if os.path.isdir(image_path):
        for dirpath,_,filenames in os.walk(image_path):
            for f in filenames:
                path=os.path.abspath(os.path.join(dirpath, f))
                if not utils.is_valid_image(path):
                    print "Warning, prease provide a valid image: ", f
                else:
                    files.append(path)
    else:
        files=[image_path]

    output_file=utils.absolute_path(ofile)
    
    if os.path.isdir(output_file):
        print "The provided file is a directory:", output_file
        sys.exit(0)

    if os.path.exists(output_file):
        os.remove(output_file)

    def extract_and_save(path):
        X = extract_fn(path)
        X = X.reshape((1, X.shape[0]))

        lock.acquire()
        with open(output_file, 'a+') as f_handle:
            numpy.savetxt(f_handle, X)
        lock.release()

    #pool = ThreadPool(cpu_count())
    pool = ThreadPool(8)
    results = pool.map(extract_and_save, files)
    pool.close()
    pool.terminate()
    pool.join()

    """
    for path in files:
        X = richmodels.SRM_extract(path)
        print X.shape
        X = X.reshape((1, X.shape[0]))
        with open(sys.argv[3], 'a+') as f_handle:
            numpy.savetxt(f_handle, X)
    """
# }}}

# {{{ train_models()
def train_models():

    print "-- TRAINING HUGO 0.40 --"
    tr_cover='../WORKDIR/DL_TR_RK_HUGO_0.40_db_boss5000_50/A_cover'
    tr_stego='../WORKDIR/DL_TR_RK_HUGO_0.40_db_boss5000_50/A_stego'
    ts_cover='../WORKDIR/DL_TS_RK_HUGO_0.40_db_boss250_50/SUP/cover'
    ts_stego='../WORKDIR/DL_TS_RK_HUGO_0.40_db_boss250_50/SUP/stego'
    tr_cover=ts_cover
    tr_stego=ts_stego
    nn = cnn.GrayScale(tr_cover, tr_stego, ts_cover, ts_stego)
    nn.train('models/hugo-0.40.h5')
# }}}

def main():

    if len(sys.argv)<2:
        print sys.argv[0], "<command>\n"
        print "COMMANDS:"
        print ""
        print "  Attacks to LSB replacement:"
        print "  - spa:   Sample Pairs Analysis."
        print "  - rs:    RS attack."
        print ""
        print "  Feature extractors:"
        print "  - srm:    Full Spatial Rich Models."
        print "  - srmq1:  Spatial Rich Models with fixed quantization q=1c."
        print ""
        print "  Embedding simulators:"
        print "  - hugo-sim:       Embedding using HUGO simulator."
        print "  - wow-sim:        Embedding using WOW simulator."
        print "  - s-uniward-sim:  Embedding using S-UNIWARD simulator."
        print ""
        print "  Model training::"
        print "  - rf:   Random Forest."
        print ""
        print "\n"
        sys.exit(0)


    if False: pass


    # -- ATTACKS --

    # {{{ spa
    elif sys.argv[1]=="spa":
   
        if len(sys.argv)!=3:
            print sys.argv[0], "spa <image>\n"
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print "Please, provide a valid image"
            sys.exit(0)

        threshold=0.05
        bitrate_R=attacks.spa(sys.argv[2], 0)
        bitrate_G=attacks.spa(sys.argv[2], 1)
        bitrate_B=attacks.spa(sys.argv[2], 2)

        if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
            print "No hiden data found"
            sys.exit(0)

        if bitrate_R>=threshold:
            print "Hiden data found in channel R", bitrate_R
        if bitrate_G>=threshold:
            print "Hiden data found in channel G", bitrate_G
        if bitrate_B>=threshold:
            print "Hiden data found in channel B", bitrate_B
        sys.exit(0)
    # }}}

    # {{{ rs
    elif sys.argv[1]=="rs":

        if len(sys.argv)!=3:
            print sys.argv[0], "spa <image>\n"
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print "Please, provide a valid image"
            sys.exit(0)

        threshold=0.05
        bitrate_R=attacks.rs(sys.argv[2], 0)
        bitrate_G=attacks.rs(sys.argv[2], 1)
        bitrate_B=attacks.rs(sys.argv[2], 2)

        if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
            print "No hiden data found"
            sys.exit(0)

        if bitrate_R>=threshold:
            print "Hiden data found in channel R", bitrate_R
        if bitrate_G>=threshold:
            print "Hiden data found in channel G", bitrate_G
        if bitrate_B>=threshold:
            print "Hiden data found in channel B", bitrate_B
        sys.exit(0)
    # }}}



    # -- FEATURE EXTRACTORS --

    # {{{ srm
    elif sys.argv[1]=="srm":

        if len(sys.argv)!=4:
            print sys.argv[0], "srm <image/dir> <output-file>\n"
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        extract_features(richmodels.SRM_extract, image_path, ofile)
    # }}}

    # {{{ srmq1
    elif sys.argv[1]=="srmq1":

        if len(sys.argv)!=4:
            print sys.argv[0], "srm <image/dir> <output-file>\n"
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        extract_features(richmodels.SRMQ1_extract, image_path, ofile)
    # }}}



    # -- EMBEDDING SIMULATORS --

    # {{{ hugo-sim
    if sys.argv[1]=="hugo-sim":

        if len(sys.argv)!=5:
            print sys.argv[0], "hugo-sim <image/dir> <payload> <output-dir>\n"
            sys.exit(0)

        embed_message(stegosim.hugo, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ wow-sim
    elif sys.argv[1]=="wow-sim":

        if len(sys.argv)!=5:
            print sys.argv[0], "wow-sim <image/dir> <payload> <output-dir>\n"
            sys.exit(0)

        embed_message(stegosim.wow, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ s-uniward-sim
    elif sys.argv[1]=="s-uniward-sim":

        if len(sys.argv)!=5:
            print sys.argv[0], "s-uniward-sim <image/dir> <payload> <output-dir>\n"
            sys.exit(0)

        embed_message(stegosim.s_uniward, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}



    # -- MODEL TRAINING --

    # {{{ rf
    if sys.argv[1]=="rf":

        if len(sys.argv)!=4:
            print sys.argv[0], "rf <cover-fea> <stego-fea>\n"
            sys.exit(0)

        cover_fea=sys.argv[2]
        stego_fea=sys.argv[3]

        Xc=numpy.loadtxt(cover_fea)
        print Xc.shape
        Xs=numpy.loadtxt(stego_fea)
        print Xs.shape


    # }}}




    else:
        print "Wrong command!"

    if sys.argv[1]=="train-models":
        train_models()


if __name__ == "__main__":
    main()



