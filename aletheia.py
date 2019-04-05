#!/usr/bin/env python3 

import os
import sys
import json
import time
import scipy
import numpy
import pandas
import pickle
import shutil
import subprocess

from scipy import misc

from aletheia import attacks, utils
from aletheia import stegosim, feaext, models
#from cnn import net as cnn





# {{{ train_models()
def train_models():

    print("-- TRAINING HUGO 0.40 --")
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
    "  - xu-net:   Convolutional Neural Network for Steganalysis."

    mldetect_doc="\n" \
    "  ML-based detectors:\n" \
    "  - esvm-predict:  Predict using eSVM.\n" \
    "  - e4s-predict:   Predict using EC."

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

    naive_doc="\n" \
    "  Naive attacks:\n" \
    "  - brute-force:       Brute force attack using a list of passwords.\n" \
    "  - hpf:               High-pass filter.\n" \
    "  - imgdiff:           Differences between two images.\n" \
    "  - imgdiff-pixels:    Differences between two images (show pixel values).\n" \
    "  - rm-alpha:          Opacity of the alpha channel to 255."


    if len(sys.argv)<2:
        print(sys.argv[0], "<command>\n")
        print("COMMANDS:")
        print(attacks_doc)
        print(mldetect_doc)
        print(feaextract_doc)
        print(embsim_doc)
        print(model_doc)
        print(auto_doc)
        print(naive_doc)
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
                print("No hiden data found")
            else:
                print("Hiden data found"), bitrate
        else:
            bitrate_R=attacks.spa(sys.argv[2], 0)
            bitrate_G=attacks.spa(sys.argv[2], 1)
            bitrate_B=attacks.spa(sys.argv[2], 2)

            if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
                print("No hiden data found")
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
                print("No hiden data found")
            else:
                print("Hiden data found", bitrate)
        else:
            bitrate_R=attacks.rs(sys.argv[2], 0)
            bitrate_G=attacks.rs(sys.argv[2], 1)
            bitrate_B=attacks.rs(sys.argv[2], 2)

            if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
                print("No hiden data found")
                sys.exit(0)

            if bitrate_R>=threshold:
                print("Hiden data found in channel R", bitrate_R)
            if bitrate_G>=threshold:
                print("Hiden data found in channel G", bitrate_G)
            if bitrate_B>=threshold:
                print("Hiden data found in channel B", bitrate_B)
            sys.exit(0)
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

    # {{{ xu-net
    elif sys.argv[1]=="xu-net":

        if len(sys.argv)!=5:
            print(sys.argv[0], "xu-net <cover-dir> <stego-dir> <model-name>\n")
            sys.exit(0)

        print("WARNING! xu-net module is not finished yet!")

        cover_dir=sys.argv[2]
        stego_dir=sys.argv[3]
        model_name=sys.argv[4]

        net = models.XuNet()
        net.train(cover_dir, stego_dir, val_size=0.10, name=model_name)
        
        #print("Validation score:", val_score)
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

    # {{{ imgdiff
    elif sys.argv[1]=="imgdiff":

        if len(sys.argv)!=4:
            print(sys.argv[0], "imgdiff <image1> <image2>\n")
            print("")
            sys.exit(0)

        attacks.imgdiff(sys.argv[2], sys.argv[3])
    # }}}

    # {{{ imgdiff-pixels
    elif sys.argv[1]=="imgdiff-pixels":

        if len(sys.argv)!=4:
            print(sys.argv[0], "imgdiff-pixels <image1> <image2>\n")
            print("")
            sys.exit(0)

        attacks.imgdiff_pixels(sys.argv[2], sys.argv[3])
    # }}}

    # {{{ rm-alpha
    elif sys.argv[1]=="rm-alpha":

        if len(sys.argv)!=4:
            print(sys.argv[0], "rm-alpha <input-image> <output-image>\n")
            print("")
            sys.exit(0)

        attacks.remove_alpha_channel(sys.argv[2], sys.argv[3])
    # }}}




    else:
        print("Wrong command!")

    if sys.argv[1]=="train-models":
        train_models()


if __name__ == "__main__":
    main()



