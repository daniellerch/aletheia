#!/usr/bin/python -W ignore

import sys
import json
import os
import scipy
import numpy
import pandas
import pickle
import multiprocessing

from aletheia import stegosim, richmodels, models
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count

from aletheia import attacks, utils
#from cnn import net as cnn

lock = multiprocessing.Lock()


# {{{ embed_message()
def embed_message(embed_fn, path, payload, output_dir):

    path=utils.absolute_path(path)

    # Read filenames
    files=[]
    if os.path.isdir(path):
        for dirpath,_,filenames in os.walk(path):
            for f in filenames:
                path=os.path.abspath(os.path.join(dirpath, f))
                if not utils.is_valid_image(path):
                    print "Warning, please provide a valid image: ", f
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
                    print "Warning, please provide a valid image: ", f
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
        print "  ML-based detectors:"
        print "  - esvm-predict:  Predict using eSVM."
        print "  - e4s-predict:   Predict using EC."
        print ""
        print "  Feature extractors:"
        print "  - srm:    Full Spatial Rich Models."
        print "  - srmq1:  Spatial Rich Models with fixed quantization q=1c."
        print ""
        print "  Embedding simulators:"
        print "  - hugo-sim:       Embedding using HUGO simulator."
        print "  - wow-sim:        Embedding using WOW simulator."
        print "  - s-uniward-sim:  Embedding using S-UNIWARD simulator."
        print "  - hill-sim:       Embedding using HILL simulator."
        print ""
        print "  Model training:"
        print "  - esvm:  Ensemble of Support Vector Machines."
        print "  - e4s:   Ensemble Classifiers for Steganalysis."
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


    # -- ML-BASED DETECTORS --

    # {{{ esvm
    elif sys.argv[1]=="esvm-predict":

        if len(sys.argv)!=5:
            print sys.argv[0], "esvm-predict <model-file> <feature-extractor> <image/dir>\n"
            print "Feature extractors:"
            print "  - srm:    Full Spatial Rich Models."
            print "  - srmq1:  Spatial Rich Models with fixed quantization q=1c."
            print ""
            sys.exit(0)

        model_file=sys.argv[2]
        extractor=sys.argv[3]
        path=utils.absolute_path(sys.argv[4])

        files=[]
        if os.path.isdir(path):
            for dirpath,_,filenames in os.walk(path):
                for f in filenames:
                    path=os.path.abspath(os.path.join(dirpath, f))
                    if not utils.is_valid_image(path):
                        print "Warning, please provide a valid image: ", f
                    else:
                        files.append(path)
        else:
            files=[path]


        clf=pickle.load(open(model_file, "r"))
        for f in files:
            
            if extractor=="srm": X = richmodels.SRM_extract(f)
            if extractor=="srmq1": X = richmodels.SRMQ1_extract(f)

            X = X.reshape((1, X.shape[0]))
            p = clf.predict_proba(X)
            print p
            if p[0][0] > 0.5:
                print os.path.basename(f), "Cover, probability:", p[0][0]
            else:
                print os.path.basename(f), "Stego, probability:", p[0][1]
    # }}}

    # {{{ e4s
    elif sys.argv[1]=="e4s-predict":

        if len(sys.argv)!=5:
            print sys.argv[0], "e4s-predict <model-file> <feature-extractor> <image/dir>\n"
            print "Feature extractors:"
            print "  - srm:    Full Spatial Rich Models."
            print "  - srmq1:  Spatial Rich Models with fixed quantization q=1c."
            print ""
            sys.exit(0)

        model_file=sys.argv[2]
        extractor=sys.argv[3]
        path=utils.absolute_path(sys.argv[4])

        files=[]
        if os.path.isdir(path):
            for dirpath,_,filenames in os.walk(path):
                for f in filenames:
                    path=os.path.abspath(os.path.join(dirpath, f))
                    if not utils.is_valid_image(path):
                        print "Warning, please provide a valid image: ", f
                    else:
                        files.append(path)
        else:
            files=[path]


        clf=models.Ensemble4Stego()
        clf.load(model_file)
        for f in files:
            
            if extractor=="srm": X = richmodels.SRM_extract(f)
            if extractor=="srmq1": X = richmodels.SRMQ1_extract(f)

            X = X.reshape((1, X.shape[0]))
            p = clf.predict_proba(X)
            print p
            if p[0][0] > 0.5:
                print os.path.basename(f), "Cover, probability:", p[0][0]
            else:
                print os.path.basename(f), "Stego, probability:", p[0][1]
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
    elif sys.argv[1]=="hugo-sim":

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

    # {{{ hill-sim
    elif sys.argv[1]=="hill-sim":

        if len(sys.argv)!=5:
            print sys.argv[0], "s-uniward-sim <image/dir> <payload> <output-dir>\n"
            sys.exit(0)

        embed_message(stegosim.hill, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}


    # -- MODEL TRAINING --

    # {{{ esvm
    elif sys.argv[1]=="esvm":

        if len(sys.argv)!=5:
            print sys.argv[0], "esvm <cover-fea> <stego-fea> <model-file>\n"
            sys.exit(0)

        from sklearn.model_selection import train_test_split

        cover_fea=sys.argv[2]
        stego_fea=sys.argv[3]
        model_file=sys.argv[4]

        X_cover = pandas.read_csv(cover_fea, delimiter = " ").values
        X_stego = pandas.read_csv(stego_fea, delimiter = " ").values
        #X_cover=numpy.loadtxt(cover_fea)
        #X_stego=numpy.loadtxt(stego_fea)

        X=numpy.vstack((X_cover, X_stego))
        y=numpy.hstack(([0]*len(X_cover), [1]*len(X_stego)))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10)

        clf=models.EnsembleSVM()
        clf.fit(X_train, y_train)
        val_score=clf.score(X_val, y_val)

        pickle.dump(clf, open(model_file, "wb"))
        print "Validation score:", val_score
    # }}}

    # {{{ e4s
    elif sys.argv[1]=="e4s":

        if len(sys.argv)!=5:
            print sys.argv[0], "e4s <cover-fea> <stego-fea> <model-file>\n"
            sys.exit(0)

        from sklearn.model_selection import train_test_split

        cover_fea=sys.argv[2]
        stego_fea=sys.argv[3]
        model_file=sys.argv[4]

        X_cover = pandas.read_csv(cover_fea, delimiter = " ").values
        X_stego = pandas.read_csv(stego_fea, delimiter = " ").values
        #X_cover=numpy.loadtxt(cover_fea)
        #X_stego=numpy.loadtxt(stego_fea)

        X=numpy.vstack((X_cover, X_stego))
        y=numpy.hstack(([0]*len(X_cover), [1]*len(X_stego)))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10)

        clf=models.Ensemble4Stego()
        clf.fit(X_train, y_train)
        val_score=clf.score(X_val, y_val)

        clf.save(model_file)
        print "Validation score:", val_score
    # }}}


    else:
        print "Wrong command!"

    if sys.argv[1]=="train-models":
        train_models()


if __name__ == "__main__":
    main()



