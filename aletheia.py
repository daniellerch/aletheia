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
import imageio
import tempfile
import subprocess

import numpy as np

from PIL import Image
from scipy import misc
from matplotlib import pyplot as plt
from imageio import imread

from aletheialib import jpeg
from aletheialib import attacks, utils
from aletheialib import stegosim, feaext
from aletheialib.octave_interface import _attack


def check_bin(cmd):
    if not shutil.which(cmd):
        print("ERROR: you need to install "+cmd+" to run this command!");
        sys.exit(0)


def main():

    auto_doc="\n" \
    "  Automatic steganalysis:\n" \
    "  - auto:      Try different steganalysis methods."
    #"  - auto-dci:      Try different steganalysis methods with DCI."

    #"  - aump:          Adaptive Steganalysis Attack.\n" \
    attacks_doc="\n" \
    "  Statistical attacks:\n" \
    "  - sp:            Sample Pairs Analysis (Octave vesion).\n" \
    "  - ws:            Weighted Stego Attack.\n" \
    "  - triples:       Triples Attack.\n" \
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
    "  - s-uniward-color-sim:  Embedding using S-UNIWARD color simulator.\n" \
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
    "  ML-based steganalysis:\n" \
    "  - split-sets:            Prepare sets for training and testing.\n" \
    "  - split-sets-dci:        Prepare sets for training and testing (DCI).\n" \
    "  - effnetb0:              Train a model with EfficientNet B0.\n" \
    "  - effnetb0-score:        Score with EfficientNet B0.\n" \
    "  - effnetb0-predict:      Predict with EfficientNet B0.\n" \
    "  - effnetb0-dci-score:    DCI Score with EfficientNet B0.\n" \
    "  - effnetb0-dci-predict:  DCI Predict with EfficientNet B0.\n" \
    "  - esvm:                  Train an ensemble of Support Vector Machines.\n" \
    "  - e4s:                   Train Ensemble Classifiers for Steganalysis.\n" \
    "  - esvm-predict:          Predict using eSVM.\n" \
    "  - e4s-predict:           Predict using EC."

    feaextract_doc="\n" \
    "  Feature extractors:\n" \
    "  - srm:           Full Spatial Rich Models.\n" \
    "  - hill-maxsrm:   Selection-Channel-Aware Spatial Rich Models for HILL.\n" \
    "  - srmq1:         Spatial Rich Models with fixed quantization q=1c.\n" \
    "  - scrmq1:        Spatial Color Rich Models with fixed quantization q=1c.\n" \
    "  - gfr:           JPEG steganalysis with 2D Gabor Filters."

    ats_doc="\n" \
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
        print(auto_doc)
        print(attacks_doc)
        print(feaextract_doc)
        print(embsim_doc)
        print(model_doc)
        print(ats_doc)
        print(tools_doc)
        print("\n")
        sys.exit(0)


    if False: pass



    # -- AUTO --

    # {{{ auto
    elif sys.argv[1]=="auto":
   
        if len(sys.argv)!=3:
            print(sys.argv[0], "auto <image|dir>\n")
            sys.exit(0)


        from aletheialib import models
        dir_path = os.path.dirname(os.path.realpath(__file__))
        threshold=0.05
        path = utils.absolute_path(sys.argv[2])

        os.environ["CUDA_VISIBLE_DEVICES"] = "CPU" # XXX: read from input
        #os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, '*.*'))
        else:
            files = [path]

        nn = models.NN("effnetb0")
        files = nn.filter_images(files)
        if len(files)==0:
            print("ERROR: please provice valid files")
            sys.exit(0)


        jpg_files = []
        bitmap_files = []
        for f in files:
            _, ext = os.path.splitext(f)
            if ext.lower() in ['.jpg', '.jpeg']:
                jpg_files.append(f)
            else:
                bitmap_files.append(f)
 
        # JPG
        if len(jpg_files)>0:

            model_path = os.path.join(dir_path, "models/effnetb0-A-alaska2-outguess.h5")
            nn.load_model(model_path, quiet=True)
            outguess_pred = nn.predict(jpg_files, 10)

            model_path = os.path.join(dir_path, "models/effnetb0-A-alaska2-steghide.h5")
            nn.load_model(model_path, quiet=True)
            steghide_pred = nn.predict(jpg_files, 10)

            model_path = os.path.join(dir_path, "models/effnetb0-A-alaska2-nsf5.h5")
            nn.load_model(model_path, quiet=True)
            nsf5_pred = nn.predict(jpg_files, 10)


            mx = 20
            print("")
            print(' '*mx + "    Outguess  Steghide  nsF5 *")
            print('-'*mx + "------------------------------")
            for i in range(len(jpg_files)):
                name = os.path.basename(jpg_files[i])
                if len(name)>mx:
                    name = name[:mx-3]+"..."
                else:
                    name = name.ljust(mx, ' ')
                
                print(name, 
                      "  ", round(outguess_pred[i],1), 
                      "     ", round(steghide_pred[i],1), 
                      "     ", round(nsf5_pred[i],1), 
                      )

        # BITMAP
        if len(bitmap_files)>0:

            model_path = os.path.join(dir_path, "models/effnetb0-A-alaska2-lsbr.h5")
            nn.load_model(model_path, quiet=True)
            lsbr_pred = nn.predict(bitmap_files, 10)

            model_path = os.path.join(dir_path, "models/effnetb0-A-alaska2-lsbm.h5")
            nn.load_model(model_path, quiet=True)
            lsbm_pred = nn.predict(bitmap_files, 10)

            model_path = os.path.join(dir_path, "models/effnetb0-A-alaska2-hill.h5")
            nn.load_model(model_path, quiet=True)
            hill_pred = nn.predict(bitmap_files, 10)


            mx = 20
            print("")
            print(' '*mx + "    LSBR      LSBM      HILL *")
            print('-'*mx + "------------------------------")
            for i in range(len(bitmap_files)):
                name = os.path.basename(bitmap_files[i])
                if len(name)>mx:
                    name = name[:mx-3]+"..."
                else:
                    name = name.ljust(mx, ' ')
                
                print(name, 
                      "  ", round(lsbr_pred[i],1), 
                      "     ", round(lsbm_pred[i],1), 
                      "     ", round(hill_pred[i],1), 
                      )


        print("")
        print("* Probability of being stego using the indicated steganographic method.\n")

 
        sys.exit(0)
    # }}}




    # -- ATTACKS --

    # {{{ sp
    elif sys.argv[1]=="sp":
   
        if len(sys.argv)!=3:
            print(sys.argv[0], "spa <image>\n")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        threshold=0.05
        path = utils.absolute_path(sys.argv[2])
        im=Image.open(path)
        if im.mode in ['RGB', 'RGBA', 'RGBX']:
            alpha_R = _attack('SP', path, params={"channel":1})["data"][0][0]
            alpha_G = _attack('SP', path, params={"channel":2})["data"][0][0]
            alpha_B = _attack('SP', path, params={"channel":3})["data"][0][0]
  

            if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
                print("No hidden data found")

            if alpha_R>=threshold:
                print("Hidden data found in channel R", alpha_R)
            if alpha_G>=threshold:
                print("Hidden data found in channel G", alpha_G)
            if alpha_B>=threshold:
                print("Hidden data found in channel B", alpha_B)

        else:
            alpha = _attack('SP', path, params={"channel":1})["data"][0][0]
            if alpha>=threshold:
                print("Hidden data found", alpha)
            else:
                print("No hidden data found")
 
        sys.exit(0)
    # }}}

    # {{{ ws
    elif sys.argv[1]=="ws":
   
        if len(sys.argv)!=3:
            print(sys.argv[0], "ws <image>\n")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        threshold=0.05
        path = utils.absolute_path(sys.argv[2])
        im=Image.open(path)
        if im.mode in ['RGB', 'RGBA', 'RGBX']:
            alpha_R = _attack('WS', path, params={"channel":1})["data"][0][0]
            alpha_G = _attack('WS', path, params={"channel":2})["data"][0][0]
            alpha_B = _attack('WS', path, params={"channel":3})["data"][0][0]

            if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
                print("No hidden data found")

            if alpha_R>=threshold:
                print("Hidden data found in channel R", alpha_R)
            if alpha_G>=threshold:
                print("Hidden data found in channel G", alpha_G)
            if alpha_B>=threshold:
                print("Hidden data found in channel B", alpha_B)

        else:
            alpha = _attack('WS', path, params={"channel":1})["data"][0][0]
            if alpha>=threshold:
                print("Hidden data found", alpha)
            else:
                print("No hidden data found")
 
        sys.exit(0)
    # }}}

    # {{{ triples
    elif sys.argv[1]=="triples":
   
        if len(sys.argv)!=3:
            print(sys.argv[0], "triples <image>\n")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        threshold=0.05
        path = utils.absolute_path(sys.argv[2])
        im=Image.open(path)
        if im.mode in ['RGB', 'RGBA', 'RGBX']:
            alpha_R = _attack('TRIPLES', path, params={"channel":1})["data"][0][0]
            alpha_G = _attack('TRIPLES', path, params={"channel":2})["data"][0][0]
            alpha_B = _attack('TRIPLES', path, params={"channel":3})["data"][0][0]
  

            if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
                print("No hidden data found")

            if alpha_R>=threshold:
                print("Hidden data found in channel R", alpha_R)
            if alpha_G>=threshold:
                print("Hidden data found in channel G", alpha_G)
            if alpha_B>=threshold:
                print("Hidden data found in channel B", alpha_B)

        else:
            alpha = _attack('TRIPLES', path, params={"channel":1})["data"][0][0]
            if alpha>=threshold:
                print("Hidden data found", alpha)
            else:
                print("No hidden data found")
 
        sys.exit(0)
    # }}}

    # {{{ aump
    elif sys.argv[1]=="aump":
   
        if len(sys.argv)!=3:
            print(sys.argv[0], "aump <image>\n")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        path = utils.absolute_path(sys.argv[2])
        im=Image.open(path)
        if im.mode in ['RGB', 'RGBA', 'RGBX']:
            beta_R = _attack('AUMP', path, params={"channel":1})["data"][0][0]
            beta_G = _attack('AUMP', path, params={"channel":2})["data"][0][0]
            beta_B = _attack('AUMP', path, params={"channel":3})["data"][0][0]
            # XXX: How to use beta?

        else:
            beta = _attack('AUMP', path, params={"channel":1})["data"][0][0]
            # XXX: How to use beta?
 
        sys.exit(0)
    # }}}

    # {{{ spa
    elif sys.argv[1]=="spa":
   
        if len(sys.argv)!=3:
            print(sys.argv[0], "spa <image>\n")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        threshold=0.05

        I = imread(sys.argv[2])
        if len(I.shape)==2:
            bitrate=attacks.spa_image(I, None)
            if bitrate<threshold:
                print("No hidden data found")
            else:
                print("Hidden data found"), bitrate
        else:
            bitrate_R=attacks.spa_image(I, 0)
            bitrate_G=attacks.spa_image(I, 1)
            bitrate_B=attacks.spa_image(I, 2)

            if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
                print("No hidden data found")
                sys.exit(0)

            if bitrate_R>=threshold:
                print("Hidden data found in channel R", bitrate_R)
            if bitrate_G>=threshold:
                print("Hidden data found in channel G", bitrate_G)
            if bitrate_B>=threshold:
                print("Hidden data found in channel B", bitrate_B)
        sys.exit(0)
    # }}}

    # {{{ rs
    elif sys.argv[1]=="rs":

        if len(sys.argv)!=3:
            print(sys.argv[0], "rs <image>\n")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        threshold=0.05


        I = np.asarray(imread(sys.argv[2]))
        if len(I.shape)==2:
            bitrate=attacks.rs_image(I)
            if bitrate<threshold:
                print("No hidden data found")
            else:
                print("Hidden data found", bitrate)
        else:
            bitrate_R=attacks.rs_image(I, 0)
            bitrate_G=attacks.rs_image(I, 1)
            bitrate_B=attacks.rs_image(I, 2)

            if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
                print("No hidden data found")
                sys.exit(0)

            if bitrate_R>=threshold:
                print("Hidden data found in channel R", bitrate_R)
            if bitrate_G>=threshold:
                print("Hidden data found in channel G", bitrate_G)
            if bitrate_B>=threshold:
                print("Hidden data found in channel B", bitrate_B)
            sys.exit(0)
    # }}}

    # {{{ calibration
    elif sys.argv[1]=="calibration":

        if len(sys.argv)!=4:
            print(sys.argv[0], "calibration <f5|chisquare_mode> <image>\n")
            sys.exit(0)

        if sys.argv[2] not in ["f5", "chisquare_mode"]:
            print("Please, provide a valid method")
            sys.exit(0)

        if not utils.is_valid_image(sys.argv[3]):
            print("Please, provide a valid image")
            sys.exit(0)
    
        if not sys.argv[3].lower().endswith(('.jpg', '.jpeg')):
            print("Please, provide a JPEG file")
            sys.exit(0)

        fn = utils.absolute_path(sys.argv[3])
        if "f5" in sys.argv[2]:
            attacks.calibration_f5(fn)
        elif "chisquare_mode" in sys.argv[2]:
            attacks.calibration_chisquare_mode(fn)
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

    # {{{ dctr
    elif sys.argv[1]=="dctr":

        if len(sys.argv)<4:
            print(sys.argv[0], "dctr <image/dir> <output-file> [quality]\n")
            sys.exit(0)

        image_path=sys.argv[2]
        ofile=sys.argv[3]

        if len(sys.argv)<5:
            quality = "auto"
            print("JPEG quality not provided, using detection via 'identify'")
        else:
            quality = sys.argv[4]



        params = {
            "quality": quality,
        }
            
        feaext.extract_features(feaext.DCTR_extract, image_path, ofile, params)
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

    # {{{ s-uniward-color-sim
    elif sys.argv[1]=="s-uniward-color-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "s-uniward-color-sim <image/dir> <payload> <output-dir>\n")
            sys.exit(0)

        stegosim.embed_message(stegosim.s_uniward_color, sys.argv[2], sys.argv[3], sys.argv[4])
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

    # {{{ steghide-sim
    elif sys.argv[1]=="steghide-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "steghide-sim <image/dir> <payload> <output-dir>")
            sys.exit(0)

        check_bin("steghide")

        stegosim.embed_message(stegosim.steghide, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}

    # {{{ steganogan-sim
    elif sys.argv[1]=="steganogan-sim":

        if len(sys.argv)!=5:
            print(sys.argv[0], "steganogan-sim <image/dir> <payload> <output-dir>")
            sys.exit(0)

        check_bin("steghide")

        stegosim.embed_message(stegosim.steganogan, sys.argv[2], sys.argv[3], sys.argv[4],
                      embed_fn_saving=True)
    # }}}




    # -- MODEL TRAINING --

    # {{{ split-sets
    elif sys.argv[1]=="split-sets":

        if len(sys.argv)<8:
            print(sys.argv[0], "split-sets <cover-dir> <stego-dir> <output-dir> <#valid> <#test>\n")
            print("     cover-dir:    Directory containing cover images")
            print("     stego-dir:    Directory containing stego images")
            print("     output-dir:   Output directory. Three sets will be created")
            print("     #valid:       Number of images for the validation set")
            print("     #test:        Number of images for the testing set")
            print("     seed:         Seed for reproducible results")
            print("")
            sys.exit(0)

        cover_dir=sys.argv[2]
        stego_dir=sys.argv[3]
        output_dir=sys.argv[4]
        n_valid=int(sys.argv[5])
        n_test=int(sys.argv[6])
        seed=int(sys.argv[7])


        cover_files = sorted(glob.glob(os.path.join(cover_dir, '*')))
        stego_files = sorted(glob.glob(os.path.join(stego_dir, '*')))

        if len(cover_files)!=len(stego_files):
            print("ERROR: we expect the same number of cover and stego files");
            sys.exit(0)

        from sklearn.model_selection import train_test_split
        trn_C_files, tv_C_files, trn_S_files, tv_S_files = \
            train_test_split(cover_files, stego_files, 
                             test_size=n_valid+n_test*2, random_state=seed)

        train_C_files = trn_C_files
        train_S_files = trn_S_files
        valid_C_files = tv_C_files[:n_valid//2]
        valid_S_files = tv_S_files[:n_valid//2]
        test_C_files = tv_C_files[n_valid//2:n_valid//2+n_test//2]
        test_S_files = tv_S_files[n_valid//2+n_test//2:n_valid//2+n_test]

        train_C_dir = os.path.join(output_dir, "train", "cover")
        train_S_dir = os.path.join(output_dir, "train", "stego")
        valid_C_dir = os.path.join(output_dir, "valid", "cover")
        valid_S_dir = os.path.join(output_dir, "valid", "stego")
        test_C_dir = os.path.join(output_dir, "test", "cover")
        test_S_dir = os.path.join(output_dir, "test", "stego")

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        os.makedirs(train_C_dir, exist_ok=True)
        os.makedirs(train_S_dir, exist_ok=True)
        os.makedirs(valid_C_dir, exist_ok=True)
        os.makedirs(valid_S_dir, exist_ok=True)
        os.makedirs(test_C_dir, exist_ok=True)
        os.makedirs(test_S_dir, exist_ok=True)
        
        for f in train_C_files:
            shutil.copy(f, train_C_dir)

        for f in train_S_files:
            shutil.copy(f, train_S_dir)

        for f in valid_C_files:
            shutil.copy(f, valid_C_dir)

        for f in valid_S_files:
            shutil.copy(f, valid_S_dir)

        for f in test_C_files:
            shutil.copy(f, test_C_dir)

        for f in test_S_files:
            shutil.copy(f, test_S_dir)
    # }}}

    # {{{ split-sets-dci
    elif sys.argv[1]=="split-sets-dci":

        if len(sys.argv)<9:
            print(sys.argv[0], "split-sets <cover-dir> <stego-dir> <double-dir> <output-dir> <#valid> <#test> <seed>\n")
            print("     cover-dir:    Directory containing cover images")
            print("     stego-dir:    Directory containing stego images")
            print("     double-dir:   Directory containing double stego images")
            print("     output-dir:   Output directory. Three sets will be created")
            print("     #valid:       Number of images for the validation set")
            print("     #test:        Number of images for the testing set")
            print("     seed:         Seed for reproducible results")
            print("")
            sys.exit(0)

        cover_dir=sys.argv[2]
        stego_dir=sys.argv[3]
        double_dir=sys.argv[4]
        output_dir=sys.argv[5]
        n_valid=int(sys.argv[6])
        n_test=int(sys.argv[7])
        seed=int(sys.argv[8])


        cover_files = np.array(sorted(glob.glob(os.path.join(cover_dir, '*'))))
        stego_files = np.array(sorted(glob.glob(os.path.join(stego_dir, '*'))))
        double_files = np.array(sorted(glob.glob(os.path.join(double_dir, '*'))))

        if len(cover_files)!=len(stego_files) or len(stego_files)!=len(double_files):
            print("split-sets-dci error: we expect sets with the same number of images")
            sys.exit(0)

        indices = list(range(len(cover_files)))
        random.seed(seed)
        random.shuffle(indices)

        valid_indices = indices[:n_valid//2]
        test_C_indices = indices[n_valid//2:n_valid//2+n_test//2]
        test_S_indices = indices[n_valid//2+n_test//2:n_valid//2+n_test]
        train_indices = indices[n_valid//2+n_test:]


        A_train_C_dir = os.path.join(output_dir, "A_train", "cover")
        A_train_S_dir = os.path.join(output_dir, "A_train", "stego")
        A_valid_C_dir = os.path.join(output_dir, "A_valid", "cover")
        A_valid_S_dir = os.path.join(output_dir, "A_valid", "stego")
        A_test_C_dir = os.path.join(output_dir, "A_test", "cover")
        A_test_S_dir = os.path.join(output_dir, "A_test", "stego")
        B_train_S_dir = os.path.join(output_dir, "B_train", "stego")
        B_train_D_dir = os.path.join(output_dir, "B_train", "double")
        B_valid_S_dir = os.path.join(output_dir, "B_valid", "stego")
        B_valid_D_dir = os.path.join(output_dir, "B_valid", "double")
        B_test_S_dir = os.path.join(output_dir, "B_test", "stego")
        B_test_D_dir = os.path.join(output_dir, "B_test", "double")


        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        os.makedirs(A_train_C_dir, exist_ok=True)
        os.makedirs(A_train_S_dir, exist_ok=True)
        os.makedirs(A_valid_C_dir, exist_ok=True)
        os.makedirs(A_valid_S_dir, exist_ok=True)
        os.makedirs(A_test_C_dir, exist_ok=True)
        os.makedirs(A_test_S_dir, exist_ok=True)
        os.makedirs(B_train_S_dir, exist_ok=True)
        os.makedirs(B_train_D_dir, exist_ok=True)
        os.makedirs(B_valid_S_dir, exist_ok=True)
        os.makedirs(B_valid_D_dir, exist_ok=True)
        os.makedirs(B_test_S_dir, exist_ok=True)
        os.makedirs(B_test_D_dir, exist_ok=True)

        for f in cover_files[train_indices]:
            shutil.copy(f, A_train_C_dir)

        for f in stego_files[train_indices]:
            shutil.copy(f, A_train_S_dir)
            shutil.copy(f, B_train_S_dir)

        for f in double_files[train_indices]:
            shutil.copy(f, B_train_D_dir)


        for f in cover_files[valid_indices]:
            shutil.copy(f, A_valid_C_dir)

        for f in stego_files[valid_indices]:
            shutil.copy(f, A_valid_S_dir)
            shutil.copy(f, B_valid_S_dir)

        for f in double_files[valid_indices]:
            shutil.copy(f, B_valid_D_dir)


        for f in cover_files[test_C_indices]:
            shutil.copy(f, A_test_C_dir)

        for f in stego_files[test_S_indices]:
            shutil.copy(f, A_test_S_dir)

        for f in stego_files[test_C_indices]:
            shutil.copy(f, B_test_S_dir)

        for f in double_files[test_S_indices]:
            shutil.copy(f, B_test_D_dir)



    # }}}


    # {{{ effnetb0
    elif sys.argv[1]=="effnetb0":
        from aletheialib import models

        if len(sys.argv)<7:
            print(sys.argv[0], "effnetb0 <trn-cover-dir> <trn-stego-dir> <val-cover-dir> <val-stego-dir> <model-file> [dev] [ES]\n")
            print("     trn-cover-dir:    Directory containing training cover images")
            print("     trn-stego-dir:    Directory containing training stego images")
            print("     val-cover-dir:    Directory containing validation cover images")
            print("     val-stego-dir:    Directory containing validation stego images")
            print("     model-name:       A name for the model")
            print("     dev:        Device: GPU Id or 'CPU' (default='CPU')")
            print("     ES:         early stopping iterations x1000 (default=100)")
            print("")
            sys.exit(0)

        trn_cover_dir=sys.argv[2]
        trn_stego_dir=sys.argv[3]
        val_cover_dir=sys.argv[4]
        val_stego_dir=sys.argv[5]
        model_name=sys.argv[6]

        if len(sys.argv)<8:
            dev_id = "CPU"
            print("'dev' not provided, using:", dev_id)
        else:
            dev_id = sys.argv[7]

        if len(sys.argv)<9:
            early_stopping = 10
            print("'ES' not provided, using:", early_stopping)
        else:
            early_stopping = int(sys.argv[8])

        if dev_id == "CPU":
            print("Running with CPU. It could be very slow!")


        os.environ["CUDA_VISIBLE_DEVICES"]=dev_id
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


        trn_cover_files = sorted(glob.glob(os.path.join(trn_cover_dir, '*')))
        trn_stego_files = sorted(glob.glob(os.path.join(trn_stego_dir, '*')))
        val_cover_files = sorted(glob.glob(os.path.join(val_cover_dir, '*')))
        val_stego_files = sorted(glob.glob(os.path.join(val_stego_dir, '*')))

        print("train:", len(trn_cover_files),"+",len(trn_stego_files))
        print("valid:", len(val_cover_files),"+",len(val_stego_files))

        if (not len(trn_cover_files) or not len(trn_stego_files) or
            not len(val_cover_files) or not len(val_stego_files)):
            print("ERROR: directory without files found")
            sys.exit(0)


        nn = models.NN("effnetb0", model_name=model_name, shape=(512,512,3))
        nn.train(trn_cover_files, trn_stego_files, 36, # 36|40
        #nn = models.NN("effnetb0", model_name=model_name, shape=(32,32,3))
        #nn.train(trn_cover_files, trn_stego_files, 500, # 36|40
                 val_cover_files, val_stego_files, 10,
                 1000000, early_stopping)


    # }}}

    # {{{ effnetb0-score
    elif sys.argv[1]=="effnetb0-score":
        from aletheialib import models

        if len(sys.argv)<5:
            print(sys.argv[0], "effnetb0-score <test-cover-dir> <test-stego-dir> <model-file> [dev]\n")
            print("     test-cover-dir:    Directory containing cover images")
            print("     test-stego-dir:    Directory containing stego images")
            print("     model-file:        Path of the model")
            print("     dev:        Device: GPU Id or 'CPU' (default='CPU')")
            print("")
            sys.exit(0)

        cover_dir=sys.argv[2]
        stego_dir=sys.argv[3]
        model_file=sys.argv[4]

        if len(sys.argv)<6:
            dev_id = "CPU"
            print("'dev' not provided, using:", dev_id)
        else:
            dev_id = sys.argv[5]

        if dev_id == "CPU":
            print("Running with CPU. It could be very slow!")

        os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        cover_files = sorted(glob.glob(os.path.join(cover_dir, '*')))
        stego_files = sorted(glob.glob(os.path.join(stego_dir, '*')))

        nn = models.NN("effnetb0")
        nn.load_model(model_file)

        pred_cover = nn.predict(cover_files, 10)
        pred_stego = nn.predict(stego_files, 10)

        ok = np.sum(np.round(pred_cover)==0)+np.sum(np.round(pred_stego)==1)
        score = ok/(len(pred_cover)+len(pred_stego))
        print("score:", score)

    # }}}

    # {{{ effnetb0-predict
    elif sys.argv[1]=="effnetb0-predict":
        from aletheialib import models

        if len(sys.argv)<4:
            print(sys.argv[0], "effnetb0-predict <test-dir/image> <model-file> [dev]\n")
            print("     test-dir:    Directory containing test images")
            print("     model-file:        Path of the model")
            print("     dev:        Device: GPU Id or 'CPU' (default='CPU')")
            print("")
            sys.exit(0)

        test_dir=sys.argv[2]
        model_file=sys.argv[3]

        if len(sys.argv)<5:
            dev_id = "CPU"
            print("'dev' not provided, using:", dev_id)
        else:
            dev_id = sys.argv[4]

        if dev_id == "CPU":
            print("Running with CPU. It could be very slow!")

        os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        nn = models.NN("effnetb0")
        nn.load_model(model_file)

        if os.path.isdir(test_dir):
            test_files = sorted(glob.glob(os.path.join(test_dir, '*')))
        else:
            test_files = [test_dir]

        test_files = nn.filter_images(test_files)
        if len(test_files)==0:
            print("ERROR: please provice valid files")
            sys.exit(0)


        pred = nn.predict(test_files, 10)

        for i in range(len(pred)):
            print(test_files[i], round(pred[i],3))

    # }}}

    # {{{ effnetb0-dci-score
    elif sys.argv[1]=="effnetb0-dci-score":
        from aletheialib import models
        from sklearn.metrics import accuracy_score

        if len(sys.argv)<8:
            print(sys.argv[0], "effnetb0-dci-score <A-test-cover-dir> <A-test-stego-dir> <B-test-stego-dir> <B-test-double-dir> <A-model-file> <B-model-file> [dev]\n")
            print("     A-test-cover-dir:    Directory containing A-cover images")
            print("     A-test-stego-dir:    Directory containing A-stego images")
            print("     B-test-stego-dir:    Directory containing B-stego images")
            print("     B-test-double-dir:   Directory containing B-double images")
            print("     A-model-file:        Path of the A-model")
            print("     B-model-file:        Path of the B-model")
            print("     dev:                 Device: GPU Id or 'CPU' (default='CPU')")
            print("")
            sys.exit(0)

        A_cover_dir=sys.argv[2]
        A_stego_dir=sys.argv[3]
        B_stego_dir=sys.argv[4]
        B_double_dir=sys.argv[5]
        A_model_file=sys.argv[6]
        B_model_file=sys.argv[7]

        if len(sys.argv)<9:
            dev_id = "CPU"
            print("'dev' not provided, using:", dev_id)
        else:
            dev_id = sys.argv[8]

        if dev_id == "CPU":
            print("Running with CPU. It could be very slow!")

        os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        A_cover_files = sorted(glob.glob(os.path.join(A_cover_dir, '*')))
        A_stego_files = sorted(glob.glob(os.path.join(A_stego_dir, '*')))
        B_stego_files = sorted(glob.glob(os.path.join(B_stego_dir, '*')))
        B_double_files = sorted(glob.glob(os.path.join(B_double_dir, '*')))

        A_nn = models.NN("effnetb0")
        A_nn.load_model(A_model_file)
        B_nn = models.NN("effnetb0")
        B_nn.load_model(B_model_file)


        A_files = A_cover_files+A_stego_files
        B_files = B_stego_files+B_double_files


        p_aa = A_nn.predict(A_files, 10)
        p_ab = A_nn.predict(B_files, 10)
        p_bb = B_nn.predict(B_files, 10)
        p_ba = B_nn.predict(A_files, 10)

        p_aa = np.round(p_aa).astype('uint8')
        p_ab = np.round(p_ab).astype('uint8')
        p_ba = np.round(p_ba).astype('uint8')
        p_bb = np.round(p_bb).astype('uint8')

        y_true = np.array([0]*len(A_cover_files) + [1]*len(A_stego_files))
        inc = ( (p_aa!=p_bb) | (p_ba!=0) | (p_ab!=1) ).astype('uint8')
        inc1 = (p_aa!=p_bb).astype('uint8')
        inc2 = ( (p_ba!=0) | (p_ab!=1) ).astype('uint8')
        inc2c = (p_ab!=1).astype('uint8')
        inc2s = (p_ba!=0).astype('uint8')
        C_ok = ( (p_aa==0) & (p_aa==y_true) & (inc==0) ).astype('uint8')
        S_ok = ( (p_aa==1) & (p_aa==y_true) & (inc==0) ).astype('uint8')

        print("#inc:", np.sum(inc==1), "#incF1:", np.sum(inc1==1), "#incF2:", np.sum(inc2==1),
               "#incF2C", np.sum(inc2c), "#incF2S:", np.sum(inc2s))
        print("#no_inc:", len(A_files)-np.sum(inc==1))
        print("#C-ok:", np.sum(C_ok==1))
        print("#S-ok:", np.sum(S_ok==1))
        print("aa-score:", accuracy_score(y_true, p_aa))
        print("bb-score:", accuracy_score(y_true, p_bb))
        print("dci-score:", float(np.sum(C_ok==1)+np.sum(S_ok==1))/(len(A_files)-np.sum(inc==1)))
        print("--")
        print("dci-prediction-score:", 1-float(np.sum(inc==1))/(2*len(p_aa)))

    # }}}

    # {{{ effnetb0-dci-predict
    elif sys.argv[1]=="effnetb0-dci-predict":
        from aletheialib import models
        from sklearn.metrics import accuracy_score

        if len(sys.argv)<6:
            print(sys.argv[0], "effnetb0-dci-predict <A-test-dir> <B-test-dir> <A-model-file> <B-model-file> [dev]\n")
            print("     A-test-dir:          Directory containing A test images")
            print("     B-test-dir:          Directory containing B test images")
            print("     A-model-file:        Path of the A-model")
            print("     B-model-file:        Path of the B-model")
            print("     dev:                 Device: GPU Id or 'CPU' (default='CPU')")
            print("")
            sys.exit(0)

        A_dir=sys.argv[2]
        B_dir=sys.argv[3]
        A_model_file=sys.argv[4]
        B_model_file=sys.argv[5]

        if len(sys.argv)<7:
            dev_id = "CPU"
            print("'dev' not provided, using:", dev_id)
        else:
            dev_id = sys.argv[6]

        if dev_id == "CPU":
            print("Running with CPU. It could be very slow!")

        os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        A_files = sorted(glob.glob(os.path.join(A_dir, '*')))
        B_files = sorted(glob.glob(os.path.join(B_dir, '*')))

        A_nn = models.NN("effnetb0")
        A_nn.load_model(A_model_file)
        B_nn = models.NN("effnetb0")
        B_nn.load_model(B_model_file)



        p_aa = A_nn.predict(A_files, 10)
        p_ab = A_nn.predict(B_files, 10)
        p_bb = B_nn.predict(B_files, 10)
        p_ba = B_nn.predict(A_files, 10)


        p_aa = np.round(p_aa).astype('uint8')
        p_ab = np.round(p_ab).astype('uint8')
        p_ba = np.round(p_ba).astype('uint8')
        p_bb = np.round(p_bb).astype('uint8')

        inc = ( (p_aa!=p_bb) | (p_ba!=0) | (p_ab!=1) ).astype('uint8')
        inc1 = (p_aa!=p_bb).astype('uint8')
        inc2 = ( (p_ba!=0) | (p_ab!=1) ).astype('uint8')
        inc2c = (p_ab!=1).astype('uint8')
        inc2s = (p_ba!=0).astype('uint8')


        for i in range(len(p_aa)):
            r = ""
            if inc[i]:
                r = "INC"
            else:
                r = round(p_aa[i],3)
            print(A_files[i], r)

        print("#inc:", np.sum(inc==1), "#incF1:", np.sum(inc1==1), "#incF2:", np.sum(inc2==1),
               "#incF2C", np.sum(inc2c), "#incF2S:", np.sum(inc2s))
        print("#no_inc:", len(A_files)-np.sum(inc==1))
        print("--")
        print("dci-prediction-score:", 1-float(np.sum(inc==1))/(2*len(p_aa)))

    # }}}




    # {{{ esvm
    elif sys.argv[1]=="esvm":
        from aletheialib import models

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

        from aletheialib import models
        clf=models.EnsembleSVM()
        clf.fit(X_train, y_train)
        val_score=clf.score(X_val, y_val)

        pickle.dump(clf, open(model_file, "wb"))
        print("Validation score:", val_score)
    # }}}

    # {{{ e4s
    elif sys.argv[1]=="e4s":
        from aletheialib import models

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
        from aletheialib import models

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


        from aletheialib import models
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



    # -- AUTOMATED ATTACKS --

    # {{{ ats
    elif sys.argv[1]=="ats":
        from aletheialib import models

        if len(sys.argv) not in [5, 6]:
            print(sys.argv[0], "ats <embed-sim> <payload> <fea-extract> <images>")
            print(sys.argv[0], "ats <custom command> <fea-extract> <images>\n")
            print(embsim_doc)
            print("")
            print(feaextract_doc)
            print("")
            print("Examples:")
            print(sys.argv[0], "ats hill-sim 0.40 srm image_dir/")
            print(sys.argv[0], "ats 'steghide embed -cf <IMAGE> -ef secret.txt -p mypass' srm image_dir/\n")
            sys.exit(0)

        embed_fn_saving=False

        if len(sys.argv) == 6:
            emb_sim=sys.argv[2]
            payload=sys.argv[3]
            feaextract=sys.argv[4]
            A_dir=sys.argv[5]
            fn_sim=stegosim.embedding_fn(emb_sim)
            fn_feaextract=feaext.extractor_fn(feaextract)
            if emb_sim in ["j-uniward-sim", "j-uniward-color-sim", 
                           "ued-sim", "ued-color-sim", "ebs-sim", "ebs-color-sim",
                           "nsf5-sim", "nsf5-color-sim"]:
                embed_fn_saving = True
        else:
            print("custom command")
            payload=sys.argv[2] # uggly hack
            feaextract=sys.argv[3]
            A_dir=sys.argv[4]
            fn_sim=stegosim.custom
            embed_fn_saving = True
            fn_feaextract=feaext.extractor_fn(feaextract)


        B_dir=tempfile.mkdtemp()
        C_dir=tempfile.mkdtemp()
        stegosim.embed_message(fn_sim, A_dir, payload, B_dir, embed_fn_saving=embed_fn_saving)
        stegosim.embed_message(fn_sim, B_dir, payload, C_dir, embed_fn_saving=embed_fn_saving)
 
        fea_dir=tempfile.mkdtemp()
        A_fea=os.path.join(fea_dir, "A.fea")
        C_fea=os.path.join(fea_dir, "C.fea")
        feaext.extract_features(fn_feaextract, A_dir, A_fea)
        feaext.extract_features(fn_feaextract, C_dir, C_fea)

        A = pandas.read_csv(A_fea, delimiter = " ").values
        C = pandas.read_csv(C_fea, delimiter = " ").values

        X=numpy.vstack((A, C))
        y=numpy.hstack(([0]*len(A), [1]*len(C)))

        from aletheialib import models
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


    # -- PLOT --

    # {{{ plot-histogram
    elif sys.argv[1]=="plot-histogram":

        if len(sys.argv)<2:
            print(sys.argv[0], "plot-histogram <image>\n")
            print("")
            sys.exit(0)

        fn = utils.absolute_path(sys.argv[2])
        I = imageio.imread(fn)
        data = []
        if len(I.shape) == 1:
            data.append(I.flatten())
        else:
            for i in range(I.shape[2]):
                data.append(I[:,:,i].flatten())

        plt.hist(data, range(0, 255), color=["r", "g", "b"])
        plt.show()

    # }}}

    # {{{ plot-histogram-diff
    elif sys.argv[1]=="plot-histogram-diff":

        if len(sys.argv)<4:
            print(sys.argv[0], "plot-histogram <image> <L|R|U|D>\n")
            print("")
            sys.exit(0)

        fn = utils.absolute_path(sys.argv[2])
        direction = sys.argv[3]
        if direction not in ["L", "R", "U", "D"]:
            print("Please provide the substract direction: L, R, U or D\n")
            sys.exit(0)

        I = imageio.imread(fn)
        data = []
        if len(I.shape) == 1:
            if direction == "L":
                D = I[:,1:]-I[:,:-1]
            if direction == "R":
                D = I[:,:-1]-I[:,1:]
            if direction == "U":
                D = I[:-1,:]-I[1:,:]
            if direction == "D":
                D = I[1:,:]-I[:-1,:]
            
            data.append(D.flatten())
        else:
            for i in range(I.shape[2]):
                if direction == "L":
                    D = I[:,1:,i]-I[:,:-1,i]
                if direction == "R":
                    D = I[:,:-1,i]-I[:,1:,i]
                if direction == "U":
                    D = I[:-1,:,i]-I[1:,:,i]
                if direction == "D":
                    D = I[1:,:,i]-I[:-1,:,i]

                data.append(D.flatten())

        plt.hist(data, range(0, 255), color=["r", "g", "b"])
        plt.show()

    # }}}

    # {{{ plot-dct-histogram
    elif sys.argv[1]=="plot-dct-histogram":

        if len(sys.argv)<2:
            print(sys.argv[0], "plot-dct-histogram <image>\n")
            print("")
            sys.exit(0)

        fn = utils.absolute_path(sys.argv[2])
        name, ext = os.path.splitext(fn)
        if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(fn):
            print("Please, provide a a JPEG image!\n")
            sys.exit(0)
        I = jpeg.JPEG(fn)
        channels = ["r", "g", "b"]
        dct_list = []
        for i in range(I.components()):
            dct = I.coeffs(i).flatten()
            dct_list.append(dct)
            #counts, bins = np.histogram(dct, range(-5, 5))
            #plt.plot(bins[:-1], counts, channels[i])
        plt.hist(dct_list, range(-5, 5), rwidth=1, color=["r", "g", "b"])

        plt.show()

    # }}}




    else:
        print("Wrong command!")



if __name__ == "__main__":
    main()





