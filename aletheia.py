#!/usr/bin/env python3 

import os
import sys
import glob
import json
import time
import scipy
import numpy
import pandas
import pickle
import shutil
import random
import subprocess

from scipy import misc

from aletheialib import attacks, utils
from aletheialib import stegosim, feaext, models
#from cnn import net as cnn





def main():

    attacks_doc="\n" \
    "  Statistical attacks:\n" \
    "  - spa:           Sample Pairs Analysis.\n" \
    "  - rs:            RS attack.\n" \
    "  - calibration:   Calibration attack to JPEG images."

    embsim_doc="\n" \
    "  Embedding simulators:\n" \
    "  - lsbr-sim:             Embedding using LSB replacement simulator.\n" \
    "  - lsbm-sim:             Embedding using LSB matching simulator.\n" \
    "  - hugo-sim:             Embedding using HUGO simulator.\n" \
    "  - wow-sim:              Embedding using WOW simulator.\n" \
    "  - s-uniward-sim:        Embedding using S-UNIWARD simulator.\n" \
    "  - j-uniward-sim:        Embedding using J-UNIWARD simulator.\n" \
    "  - j-uniward-color-sim:  Embedding using J-UNIWARD color simulator.\n" \
    "  - hill-sim:             Embedding using HILL simulator.\n" \
    "  - ebs-sim:              Embedding using EBS simulator.\n" \
    "  - ebs-color-sim:        Embedding using EBS color simulator.\n" \
    "  - ued-sim:              Embedding using UED simulator.\n" \
    "  - ued-color-sim:        Embedding using UED color simulator.\n" \
    "  - nsf5-sim:             Embedding using nsF5 simulator.\n" \
    "  - nsf5-color-sim:       Embedding using nsF5 color simulator."

    model_doc="\n" \
    "  Model training:\n" \
    "  - esvm:     Ensemble of Support Vector Machines.\n" \
    "  - e4s:      Ensemble Classifiers for Steganalysis.\n" \
    "  - srnet:    Steganalysis Residual Network."

    mldetect_doc="\n" \
    "  ML-based detectors:\n" \
    "  - esvm-predict:   Predict using eSVM.\n" \
    "  - e4s-predict:    Predict using EC.\n" \
    "  - srnet-predict:  Predict using SRNet."

    feaextract_doc="\n" \
    "  Feature extractors:\n" \
    "  - srm:           Full Spatial Rich Models.\n" \
    "  - hill-maxsrm:   Selection-Channel-Aware Spatial Rich Models for HILL.\n" \
    "  - srmq1:         Spatial Rich Models with fixed quantization q=1c.\n" \
    "  - scrmq1:        Spatial Color Rich Models with fixed quantization q=1c.\n" \
    "  - gfr:           JPEG steganalysis with 2D Gabor Filters."

    auto_doc="\n" \
    "  Unsupervised attacks:\n" \
    "  - ats:      Artificial Training Sets."

    tools_doc="\n" \
    "  Tools:\n" \
    "  - brute-force:       Brute force attack using a list of passwords.\n" \
    "  - hpf:               High-pass filter.\n" \
    "  - print-diffs:       Differences between two images.\n" \
    "  - print-dct-diffs:   Differences between the DCT coefficients of two JPEG images.\n" \
    "  - rm-alpha:          Opacity of the alpha channel to 255.\n" \
    "  - prep-ml-exp:       Prepare an experiment for testing ML tools."



    if len(sys.argv)<2:
        print(sys.argv[0], "<command>\n")
        print("COMMANDS:")
        print(attacks_doc)
        print(mldetect_doc)
        print(feaextract_doc)
        print(embsim_doc)
        print(model_doc)
        print(auto_doc)
        print(tools_doc)
        print("\n")
        sys.exit(0)


    if False: pass


    # -- ATTACKS --

    # {{{ spa
    elif sys.argv[1]=="spa":
   
        if len(sys.argv)!=3:
            print(sys.argv[0], "spa <image>\n")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        threshold=0.05

        I = misc.imread(sys.argv[2])
        if len(I.shape)==2:
            bitrate=attacks.spa(sys.argv[2], None)
            if bitrate<threshold:
                print("No hidden data found")
            else:
                print("Hiden data found"), bitrate
        else:
            bitrate_R=attacks.spa(sys.argv[2], 0)
            bitrate_G=attacks.spa(sys.argv[2], 1)
            bitrate_B=attacks.spa(sys.argv[2], 2)

            if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
                print("No hidden data found")
                sys.exit(0)

            if bitrate_R>=threshold:
                print("Hiden data found in channel R", bitrate_R)
            if bitrate_G>=threshold:
                print("Hiden data found in channel G", bitrate_G)
            if bitrate_B>=threshold:
                print("Hiden data found in channel B", bitrate_B)

        sys.exit(0)
    # }}}

    # {{{ rs
    elif sys.argv[1]=="rs":

        if len(sys.argv)!=3:
            print(sys.argv[0], "spa <image>\n")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        threshold=0.05


        I = misc.imread(sys.argv[2])
        if len(I.shape)==2:
            bitrate=attacks.rs(sys.argv[2], None)
            if bitrate<threshold:
                print("No hidden data found")
            else:
                print("Hiden data found", bitrate)
        else:
            bitrate_R=attacks.rs(sys.argv[2], 0)
            bitrate_G=attacks.rs(sys.argv[2], 1)
            bitrate_B=attacks.rs(sys.argv[2], 2)

            if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
                print("No hidden data found")
                sys.exit(0)

            if bitrate_R>=threshold:
                print("Hiden data found in channel R", bitrate_R)
            if bitrate_G>=threshold:
                print("Hiden data found in channel G", bitrate_G)
            if bitrate_B>=threshold:
                print("Hiden data found in channel B", bitrate_B)
            sys.exit(0)
    # }}}

    # {{{ calibration
    elif sys.argv[1]=="calibration":

        if len(sys.argv)!=3:
            print(sys.argv[0], "calibration <image>\n")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        fn = utils.absolute_path(sys.argv[2])
        attacks.calibration(fn)
    # }}}



    # -- ML-BASED DETECTORS --

    # {{{ esvm-predict
    elif sys.argv[1]=="esvm-predict":

        if len(sys.argv)!=5:
            print(sys.argv[0], "esvm-predict <model-file> <feature-extractor> <image/dir>")
            print(feaextract_doc)
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
                        print("Warning, please provide a valid image: ", f)
                    else:
                        files.append(path)
        else:
            files=[path]


        clf=pickle.load(open(model_file, "r"))
        for f in files:
            
            X = feaext.extractor_fn(extractor)(f)
            X = X.reshape((1, X.shape[0]))
            p = clf.predict_proba(X)
            print(p)
            if p[0][0] > 0.5:
                print(os.path.basename(f), "Cover, probability:", p[0][0])
            else:
                print(os.path.basename(f), "Stego, probability:", p[0][1])
    # }}}

    # {{{ e4s-predict
    elif sys.argv[1]=="e4s-predict":

        if len(sys.argv)!=5:
            print(sys.argv[0], "e4s-predict <model-file> <feature-extractor> <image/dir>\n")
            print("")
            print(feaextract_doc)
            print("")
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
                        print("Warning, please provide a valid image: ", f)
                    else:
                        files.append(path)
        else:
            files=[path]


        clf=models.Ensemble4Stego()
        clf.load(model_file)
        for f in files:
           
            X = feaext.extractor_fn(extractor)(f)
            X = X.reshape((1, X.shape[0]))
            p = clf.predict(X)
            if p[0] == 0:
                print(os.path.basename(f), "Cover")
            else:
                print(os.path.basename(f), "Stego")
    # }}}

    # {{{ srnet-predict
    elif sys.argv[1]=="srnet-predict":

        if len(sys.argv)<4:
            print(sys.argv[0], "srnet-predict <model dir> <image/dir> [dev]\n")
            print("      dev:  Device: GPU Id or 'CPU' (default='CPU')")
            print("")
            sys.exit(0)

        model_dir=sys.argv[2]
        path=utils.absolute_path(sys.argv[3])

        if len(sys.argv)<5:
            dev_id = "CPU"
            print("'dev' not provided, using:", dev_id)
        else:
            dev_id = sys.argv[4]


        files=[]
        if os.path.isdir(path):
            for dirpath,_,filenames in os.walk(path):
                for f in filenames:
                    path=os.path.abspath(os.path.join(dirpath, f))
                    if not utils.is_valid_image(path):
                        print("Warning, please provide a valid image: ", f)
                    else:
                        files.append(path)
        else:
            files=[path]

        models.nn_configure_device(dev_id)
        pred = models.nn_predict(models.SRNet, files, model_dir, batch_size=20)
        #print(pred)

        for i in range(len(files)):
            if pred[i] == 0:
                print(os.path.basename(files[i]), "Cover")
            else:
                print(os.path.basename(files[i]), "Stego")
    # }}}


    # -- FEATURE EXTRACTORS --

    # {{{ srm
    elif sys.argv[1]=="srm":

        if len(sys.argv)!=4:
            print(sys.argv[0], "srm <image/dir> <output-file>\n")
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        feaext.extract_features(feaext.SRM_extract, image_path, ofile)
    # }}}

    # {{{ srmq1
    elif sys.argv[1]=="srmq1":

        if len(sys.argv)!=4:
            print(sys.argv[0], "srmq1 <image/dir> <output-file>\n")
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        feaext.extract_features(feaext.SRMQ1_extract, image_path, ofile)
    # }}}

    # {{{ scrmq1
    elif sys.argv[1]=="scrmq1":

        if len(sys.argv)!=4:
            print(sys.argv[0], "scrmq1 <image/dir> <output-file>\n")
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        feaext.extract_features(feaext.SCRMQ1_extract, image_path, ofile)
    # }}}

    # {{{ gfr
    elif sys.argv[1]=="gfr":

        if len(sys.argv)<4:
            print(sys.argv[0], "gfr <image/dir> <output-file> [quality] [rotations]\n")
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        if len(sys.argv)<5:
            quality = "auto"
            print("JPEG quality not provided, using detection via 'identify'")
        else:
            quality = sys.argv[4]


        if len(sys.argv)<6:
            rotations = 32
            print("Number of rotations for Gabor kernel no provided, using:", \
                  rotations)
        else:
            rotations = sys.argv[6]


        params = {
            "quality": quality,
            "rotations": rotations
        }
            
        feaext.extract_features(feaext.GFR_extract, image_path, ofile, params)
    # }}}

    # {{{ hill-sigma-spam-psrm
    elif sys.argv[1]=="hill-sigma-spam-psrm":

        if len(sys.argv)!=4:
            print(sys.argv[0], "hill-sigma-spam-psrm <image/dir> <output-file>\n")
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        feaext.extract_features(feaext.HILL_sigma_spam_PSRM_extract, image_path, ofile)
    # }}}

    # {{{ hill-maxsrm
    elif sys.argv[1]=="hill-maxsrm":

        if len(sys.argv)!=4:
            print(sys.argv[0], "hill-maxsrm <image/dir> <output-file>\n")
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        feaext.extract_features(feaext.HILL_MAXSRM_extract, image_path, ofile)
    # }}}


    # -- EMBEDDING SIMULATORS --

    # {{{ lsbr-sim
    elif sys.argv[1]=="lsbr-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "lsbr-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.lsbr, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ lsbm-sim
    elif sys.argv[1]=="lsbm-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "lsbm-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.lsbm, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ hugo-sim
    elif sys.argv[1]=="hugo-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "hugo-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.hugo, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ wow-sim
    elif sys.argv[1]=="wow-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "wow-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.wow, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ s-uniward-sim
    elif sys.argv[1]=="s-uniward-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "s-uniward-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.s_uniward, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ hill-sim
    elif sys.argv[1]=="hill-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "hill-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.hill, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}

    # {{{ j-uniward-sim
    elif sys.argv[1]=="j-uniward-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "j-uniward-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.j_uniward, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}

    # {{{ j-uniward-color-sim
    elif sys.argv[1]=="j-uniward-color-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "j-uniward-color-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.j_uniward_color, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}

    # {{{ ebs-sim
    elif sys.argv[1]=="ebs-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "ebs-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.ebs, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}

    # {{{ ebs-color-sim
    elif sys.argv[1]=="ebs-color-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "ebs-color-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.ebs_color, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}

    # {{{ ued-sim
    elif sys.argv[1]=="ued-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "ued-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.ued, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}

    # {{{ ued-color-sim
    elif sys.argv[1]=="ued-color-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "ued-color-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.ued_color, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}

    # {{{ nsf5-sim
    elif sys.argv[1]=="nsf5-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "nsf5-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.nsf5, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}

    # {{{ nsf5-color-sim
    elif sys.argv[1]=="nsf5-color-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "nsf5-color-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.nsf5_color, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}

    # {{{ experimental-sim
    elif sys.argv[1]=="experimental-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "experimental-sim <image/dir> <payload> <output-dir>")
            print("NOTE: Please, put your EXPERIMENTAL.m file into external/octave\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.experimental, sys.argv[2], sys.argv[3], sys.argv[4])
    # }}}


    # -- MODEL TRAINING --

    # {{{ esvm
    elif sys.argv[1]=="esvm":

        if len(sys.argv)!=5:
            print(sys.argv[0], "esvm <cover-fea> <stego-fea> <model-file>\n")
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
        print("Validation score:", val_score)
    # }}}

    # {{{ e4s
    elif sys.argv[1]=="e4s":

        if len(sys.argv)!=5:
            print(sys.argv[0], "e4s <cover-fea> <stego-fea> <model-file>\n")
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
        print("Validation score:", val_score)
    # }}}

    # {{{ srnet
    elif sys.argv[1]=="srnet":

        if len(sys.argv)<5:
            print(sys.argv[0], "srnet <cover-dir> <stego-dir> <model-name> [dev] [ES] [valsz] [logdir]\n")
            print("     dev:     Device: GPU Id or 'CPU' (default='CPU')")
            print("     ES:      early stopping iterations (default=100)")
            print("     valsz:   Size of validation set. (default=0.1%)")
            print("     logdir:  Log directory. (default=log)")
            print("")
            sys.exit(0)

        cover_dir=sys.argv[2]
        stego_dir=sys.argv[3]
        model_name=sys.argv[4]

        if len(sys.argv)<6:
            dev_id = "CPU"
            print("'dev' not provided, using:", dev_id)
        else:
            dev_id = sys.argv[5]

        if len(sys.argv)<7:
            early_stopping = 100
            print("'ES' not provided, using:", early_stopping)
        else:
            early_stopping = int(sys.argv[6])

        if len(sys.argv)<8:
            val_size = 0.1
            print("'valsz' not provided, using:", val_size)
        else:
            val_size = int(sys.argv[7])


        if len(sys.argv)<9:
            log_dir = 'log'
            print("'logdir' not provided, using:", log_dir)
        else:
            log_dir = sys.argv[8]

        if dev_id == "CPU":
            print("Running with CPU. It could be very slow!")


        models.nn_configure_device(dev_id)

        from sklearn.model_selection import train_test_split
        cover_files = sorted(glob.glob(os.path.join(cover_dir, '*')))
        stego_files = sorted(glob.glob(os.path.join(stego_dir, '*')))
        train_cover_files, valid_cover_files, train_stego_files, valid_stego_files = \
            train_test_split(cover_files, stego_files, test_size=val_size, random_state=0)
        print("Using", len(train_cover_files)*2, "samples for training and", 
                       len(valid_cover_files)*2, "for validation.")

        output_dir = os.path.join(log_dir, 'output')
        checkpoint_dir = os.path.join(log_dir, 'checkpoint')
        for d in [output_dir, checkpoint_dir]:
            try:
                os.makedirs(d)
            except:
                pass

        data = (train_cover_files, train_stego_files,
                valid_cover_files, valid_stego_files)
        models.nn_fit(models.SRNet, data, model_name, log_path=output_dir,
                      load_checkpoint=model_name, checkpoint_path=checkpoint_dir,
                      batch_size=20, optimizer=models.AdamaxOptimizer(0.001), 
                      early_stopping=early_stopping, valid_interval=1000)
    # }}}


    # -- AUTOMATED ATTACKS --

    # {{{ ats
    elif sys.argv[1]=="ats":

        if len(sys.argv)!=6:
            print(sys.argv[0], "ats <embed-sim> <payload> <fea-extract> <images>\n")
            print(embsim_doc)
            print("")
            print(feaextract_doc)
            print("")
            sys.exit(0)

        emb_sim=sys.argv[2]
        payload=sys.argv[3]
        feaextract=sys.argv[4]
        A_dir=sys.argv[5]

        fn_sim=stegosim.embedding_fn(emb_sim)
        fn_feaextract=feaext.extractor_fn(feaextract)

        import tempfile
        B_dir=tempfile.mkdtemp()
        C_dir=tempfile.mkdtemp()
        stegosim.embed_message(fn_sim, A_dir, payload, B_dir)
        stegosim.embed_message(fn_sim, B_dir, payload, C_dir)
 
        fea_dir=tempfile.mkdtemp()
        A_fea=os.path.join(fea_dir, "A.fea")
        C_fea=os.path.join(fea_dir, "C.fea")
        feaext.extract_features(fn_feaextract, A_dir, A_fea)
        feaext.extract_features(fn_feaextract, C_dir, C_fea)

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
                    print("Warning, this is not a valid image: ", f)
                else:
                    files.append(path)

        for f in files:
            B = fn_feaextract(f)
            B = B.reshape((1, B.shape[0]))
            p = clf.predict(B)
            if p[0] == 0:
                print(os.path.basename(f), "Cover")
            else:
                print(os.path.basename(f), "Stego")

        shutil.rmtree(B_dir)
        shutil.rmtree(C_dir)
        shutil.rmtree(fea_dir)

    # }}}



    # -- NAIVE ATTACKS --

    # {{{ brute-force
    elif sys.argv[1]=="brute-force":

        if len(sys.argv)!=4:
            print(sys.argv[0], "brute-force <unhide command> <passw file>\n")
            print("Example:")
            print(sys.argv[0], "brute-force 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt\n")
            print("")
            sys.exit(0)

        attacks.brute_force(sys.argv[2], sys.argv[3])
    # }}}

    # {{{ hpf
    elif sys.argv[1]=="hpf":

        if len(sys.argv)!=4:
            print(sys.argv[0], "hpf <input-image> <output-image>\n")
            print("")
            sys.exit(0)

        attacks.high_pass_filter(sys.argv[2], sys.argv[3])
    # }}}

    # {{{ print-diffs
    elif sys.argv[1]=="print-diffs":

        if len(sys.argv)!=4:
            print(sys.argv[0], "print-diffs <cover image> <stego image>\n")
            print("")
            sys.exit(0)

        cover = utils.absolute_path(sys.argv[2])
        stego = utils.absolute_path(sys.argv[3])
        if not os.path.isfile(cover):
            print("Cover file not found:", cover)
            sys.exit(0)
        if not os.path.isfile(stego):
            print("Stego file not found:", stego)
            sys.exit(0)

        attacks.print_diffs(cover, stego)
    # }}}

    # {{{ print-dct-diffs
    elif sys.argv[1]=="print-dct-diffs":

        if len(sys.argv)!=4:
            print(sys.argv[0], "print-dtc-diffs <cover image> <stego image>\n")
            print("")
            sys.exit(0)

        cover = utils.absolute_path(sys.argv[2])
        stego = utils.absolute_path(sys.argv[3])
        if not os.path.isfile(cover):
            print("Cover file not found:", cover)
            sys.exit(0)
        if not os.path.isfile(stego):
            print("Stego file not found:", stego)
            sys.exit(0)


        attacks.print_dct_diffs(cover, stego)
    # }}}

    # {{{ rm-alpha
    elif sys.argv[1]=="rm-alpha":

        if len(sys.argv)!=4:
            print(sys.argv[0], "rm-alpha <input-image> <output-image>\n")
            print("")
            sys.exit(0)

        attacks.remove_alpha_channel(sys.argv[2], sys.argv[3])
    # }}}


    # -- TOOLS --

    # {{{ prepare-ml-experiment
    elif sys.argv[1]=="prep-ml-exp":

        if len(sys.argv)<6:
            #print(sys.argv[0], "prep-ml-exp <cover dir> <output dir> <test size> <sim> <payload> [transf]\n")
            print(sys.argv[0], "prep-ml-exp <cover dir> <output dir> <test size> <sim> <payload>\n")
            #print("   transf: list of transformations separated by '|' to apply before hidding data.")
            #print("         - cropNxN: Crop a centered NxN patch")
            #print("         - resizeNxN: Resize the image to NxN")
            #print("")
            print("Example:")
            #print("", sys.argv[0], " prep-ml-exp cover/ out/ 0.1 hill-sim 0.4 crop512x512|BLUR|SHARP")
            print("", sys.argv[0], " prep-ml-exp cover/ out/ 0.1 hill-sim 0.4")
            print("")
            sys.exit(0)

        cover_dir = sys.argv[2]
        output_dir = sys.argv[3]
        test_size = float(sys.argv[4])
        emb_sim = sys.argv[5]
        payload = float(sys.argv[6])

        trn_cover = os.path.join(output_dir, 'trnset', 'cover')
        trn_stego = os.path.join(output_dir, 'trnset', 'stego')
        tst_cover = os.path.join(output_dir, 'tstset', 'cover')
        tst_stego = os.path.join(output_dir, 'tstset', 'stego')
        tst_cover_to_stego = os.path.join(output_dir, 'tstset', '_cover_to_stego')
        fn_sim=stegosim.embedding_fn(emb_sim)

        files = sorted(glob.glob(os.path.join(cover_dir, '*')))
        random.seed(0)
        random.shuffle(files)

        test_files = files[:int(len(files)*test_size)]
        train_files = files[int(len(files)*test_size):]
        print("Using", len(train_files)*2, "files for training and", len(test_files), "for testing.")

        for d in [trn_cover, trn_stego, tst_cover, tst_stego, tst_cover_to_stego]:
            try:
                os.makedirs(d)
            except:
                pass

        for f in train_files:
            shutil.copy(f, trn_cover)
        stegosim.embed_message(fn_sim, trn_cover, payload, trn_stego)


        # to avoid leaks we do not use the same images as cover and stego
        cover_test_files = test_files[:int(len(test_files)*0.5)]
        cover_to_stego_test_files = test_files[int(len(test_files)*0.5):]
        for f in cover_test_files:
            shutil.copy(f, tst_cover)
        for f in cover_to_stego_test_files:
            shutil.copy(f, tst_cover_to_stego)
        stegosim.embed_message(fn_sim, tst_cover_to_stego, payload, tst_stego)
        shutil.rmtree(tst_cover_to_stego)

    # }}}



    else:
        print("Wrong command!")



if __name__ == "__main__":
    main()





