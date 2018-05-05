#!/usr/bin/python -W ignore

import sys
import json
import os
import scipy
import numpy
import pandas
import pickle
import multiprocessing
import shutil

from aletheia import stegosim, richmodels, models
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count
from scipy import misc

from aletheia import attacks, utils
#from cnn import net as cnn

lock = multiprocessing.Lock()


# {{{ embed_message()
def embed_message(embed_fn, path, payload, output_dir):

    path=utils.absolute_path(path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        n_core=cpu_count()
        print "Using", n_core, "threads"
        pool = ThreadPool(n_core)
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

    files.sort(key=utils.natural_sort_key)

    output_file=utils.absolute_path(ofile)
    
    if os.path.isdir(output_file):
        print "The provided file is a directory:", output_file
        sys.exit(0)

    if os.path.exists(output_file):
        os.remove(output_file)

    def extract_and_save(path):
        try:
            X = extract_fn(path)
        except Exception,e:
            print "Cannot extract feactures from", path
            print str(e)
            return

        X = X.reshape((1, X.shape[0]))
        lock.acquire()
        with open(output_file, 'a+') as f_handle:
            with open(output_file+".label", 'a+') as f_handle_label:
                numpy.savetxt(f_handle, X)
                f_handle_label.write(os.path.basename(path)+"\n")
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

    attacks_doc="\n" \
    "  Attacks to LSB replacement:\n" \
    "  - spa:   Sample Pairs Analysis.\n" \
    "  - rs:    RS attack."

    embsim_doc="\n" \
    "  Embedding simulators:\n" \
    "  - lsbr-sim:       Embedding using LSB replacement simulator.\n" \
    "  - lsbm-sim:       Embedding using LSB matching simulator.\n" \
    "  - hugo-sim:       Embedding using HUGO simulator.\n" \
    "  - wow-sim:        Embedding using WOW simulator.\n" \
    "  - s-uniward-sim:  Embedding using S-UNIWARD simulator.\n" \
    "  - hill-sim:       Embedding using HILL simulator."

    model_doc="\n" \
    "  Model training:\n" \
    "  - esvm:     Ensemble of Support Vector Machines.\n" \
    "  - e4s:      Ensemble Classifiers for Steganalysis.\n" \
    "  - xu-net:   Convolutional Neural Network for Steganalysis."

    mldetect_doc="\n" \
    "  ML-based detectors:\n" \
    "  - esvm-predict:  Predict using eSVM.\n" \
    "  - e4s-predict:   Predict using EC."

    feaextract_doc="\n" \
    "  Feature extractors:\n" \
    "  - srm:    Full Spatial Rich Models.\n" \
    "  - srmq1:  Spatial Rich Models with fixed quantization q=1c.\n" \
    "  - scrmq1: Spatial Color Rich Models with fixed quantization q=1c."

    auto_doc="\n" \
    "  Automated attacks:\n" \
    "  - ats:      Artificial Training Sets."

    if len(sys.argv)<2:
        print sys.argv[0], "<command>\n"
        print "COMMANDS:"
        print attacks_doc
        print mldetect_doc
        print feaextract_doc
        print embsim_doc
        print model_doc
        print auto_doc
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

        I = misc.imread(sys.argv[2])
        if len(I.shape)==2:
            bitrate=attacks.spa(sys.argv[2], None)
            if bitrate<threshold:
                print "No hiden data found"
            else:
                print "Hiden data found", bitrate
        else:
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


        I = misc.imread(sys.argv[2])
        if len(I.shape)==2:
            bitrate=attacks.rs(sys.argv[2], None)
            if bitrate<threshold:
                print "No hiden data found"
            else:
                print "Hiden data found", bitrate
        else:
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

    # {{{ esvm-predict
    elif sys.argv[1]=="esvm-predict":

        if len(sys.argv)!=5:
            print sys.argv[0], "esvm-predict <model-file> <feature-extractor> <image/dir>"
            print feaextract_doc
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
            if extractor=="scrmq1": X = richmodels.SCRMQ1_extract(f)

            X = X.reshape((1, X.shape[0]))
            p = clf.predict_proba(X)
            print p
            if p[0][0] > 0.5:
                print os.path.basename(f), "Cover, probability:", p[0][0]
            else:
                print os.path.basename(f), "Stego, probability:", p[0][1]
    # }}}

    # {{{ e4s-predict
    elif sys.argv[1]=="e4s-predict":

        if len(sys.argv)!=5:
            print sys.argv[0], "e4s-predict <model-file> <feature-extractor> <image/dir>\n"
            print ""
            print feaextract_doc
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
            elif extractor=="srmq1": X = richmodels.SRMQ1_extract(f)
            elif extractor=="scrmq1": X = richmodels.SCRMQ1_extract(f)
            else:
                print "Unknown extractor:", extractor
                sys.exit(0)

            X = X.reshape((1, X.shape[0]))
            p = clf.predict(X)
            if p[0] == 0:
                print os.path.basename(f), "Cover"
            else:
                print os.path.basename(f), "Stego"
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
            print sys.argv[0], "srmq1 <image/dir> <output-file>\n"
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        extract_features(richmodels.SRMQ1_extract, image_path, ofile)
    # }}}

    # {{{ scrmq1
    elif sys.argv[1]=="scrmq1":

        if len(sys.argv)!=4:
            print sys.argv[0], "scrmq1 <image/dir> <output-file>\n"
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        extract_features(richmodels.SCRMQ1_extract, image_path, ofile)
    # }}}



    # -- EMBEDDING SIMULATORS --

    # {{{ lsbr-sim
    elif sys.argv[1]=="lsbr-sim":

        if len(sys.argv)!=5:
            print sys.argv[0], "lsbr-sim <image/dir> <payload> <output-dir>\n"
            sys.exit(0)

        embed_message(stegosim.lsbr, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ lsbm-sim
    elif sys.argv[1]=="lsbm-sim":

        if len(sys.argv)!=5:
            print sys.argv[0], "lsbm-sim <image/dir> <payload> <output-dir>\n"
            sys.exit(0)

        embed_message(stegosim.lsbm, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

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
            print sys.argv[0], "hill-sim <image/dir> <payload> <output-dir>\n"
            sys.exit(0)

        embed_message(stegosim.hill, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ experimental-sim
    elif sys.argv[1]=="experimental-sim":

        if len(sys.argv)!=5:
            print sys.argv[0], "experimental-sim <image/dir> <payload> <output-dir>"
            print "NOTE: Please, put your EXPERIMENTAL.m file into external/octave\n"
            sys.exit(0)

        embed_message(stegosim.experimental, sys.argv[2], sys.argv[3], sys.argv[4])
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
        model_file=utils.absolute_path(sys.argv[4])

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
        model_file=utils.absolute_path(sys.argv[4])

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

    # {{{ xu-net
    elif sys.argv[1]=="xu-net":

        if len(sys.argv)!=5:
            print sys.argv[0], "xu-net <cover-dir> <stego-dir> <model-name>\n"
            sys.exit(0)

        print "WARNING! xu-net module is not finished yet!"

        cover_dir=sys.argv[2]
        stego_dir=sys.argv[3]
        model_name=sys.argv[4]

        net = models.XuNet()
        net.train(cover_dir, stego_dir, val_size=0.10, name=model_name)
        
        #print "Validation score:", val_score
    # }}}


    # -- AUTOMATED ATTACKS --

    # {{{ ats
    elif sys.argv[1]=="ats":

        if len(sys.argv)!=6:
            print sys.argv[0], "ats <embed-sim> <payload> <fea-extract> <images>\n"
            print "  Embedding simulators:"
            print "  - lsbr-sim:       Embedding using LSB replacement simulator."
            print "  - lsbm-sim:       Embedding using LSB matching simulator."
            print "  - hugo-sim:       Embedding using HUGO simulator."
            print "  - wow-sim:        Embedding using WOW simulator."
            print "  - s-uniward-sim:  Embedding using S-UNIWARD simulator."
            print "  - hill-sim:       Embedding using HILL simulator."
            print ""
            print feaextract_doc
            print ""
            sys.exit(0)

        emb_sim=sys.argv[2]
        payload=sys.argv[3]
        feaextract=sys.argv[4]
        A_dir=sys.argv[5]

        fn_sim=""
        if emb_sim=="lsbm-sim": fn_sim=stegosim.lsbm
        elif emb_sim=="lsbr-sim": fn_sim=stegosim.lsbr
        elif emb_sim=="hugo-sim": fn_sim=stegosim.hugo
        elif emb_sim=="wow-sim": fn_sim=stegosim.wow
        elif emb_sim=="s-uniward-sim": fn_sim=stegosim.s_uniward
        elif emb_sim=="hill-sim": fn_sim=stegosim.hill
        else: 
            print "Unknown simulator:", emb_sim
            sys.exit(0)

        fn_feaextract=""
        if feaextract=="srm": fn_feaextract=richmodels.SRM_extract
        elif feaextract=="srmq1": fn_feaextract=richmodels.SRMQ1_extract
        elif feaextract=="scrmq1": fn_feaextract=richmodels.SCRMQ1_extract
        else: 
            print "Unknown feature extractor:", feaextract
            sys.exit(0)

        import tempfile
        B_dir=tempfile.mkdtemp()
        C_dir=tempfile.mkdtemp()
        embed_message(fn_sim, A_dir, payload, B_dir)
        embed_message(fn_sim, B_dir, payload, C_dir)
 
        fea_dir=tempfile.mkdtemp()
        A_fea=os.path.join(fea_dir, "A.fea")
        C_fea=os.path.join(fea_dir, "C.fea")
        extract_features(fn_feaextract, A_dir, A_fea)
        extract_features(fn_feaextract, C_dir, C_fea)

        A = pandas.read_csv(A_fea, delimiter = " ").values
        C = pandas.read_csv(C_fea, delimiter = " ").values

        X=numpy.vstack((A, C))
        y=numpy.hstack(([0]*len(A), [1]*len(C)))

        clf=models.Ensemble4Stego()
        clf.fit(X, y)


        files=[]
        for dirpath,_,filenames in os.walk(B_dir):
            for f in filenames:
                path=os.path.abspath(os.path.join(dirpath, f))
                if not utils.is_valid_image(path):
                    print "Warning, this is not a valid image: ", f
                else:
                    files.append(path)

        for f in files:
            B = fn_feaextract(f)
            B = B.reshape((1, B.shape[0]))
            p = clf.predict(B)
            if p[0] == 0:
                print os.path.basename(f), "Cover"
            else:
                print os.path.basename(f), "Stego"

        shutil.rmtree(B_dir)
        shutil.rmtree(C_dir)
        shutil.rmtree(fea_dir)

    # }}}


    else:
        print "Wrong command!"

    if sys.argv[1]=="train-models":
        train_models()


if __name__ == "__main__":
    main()



