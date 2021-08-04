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

from scipy import misc
from imageio import imread

import aletheialib.options as options



def main():

    ats_doc="\n" \
    "  Unsupervised attacks:\n" \
    "  - ats:      Artificial Training Sets."


    if len(sys.argv)<2:
        print(sys.argv[0], "<command>\n")
        print("COMMANDS:")
        print(options.auto.doc)
        print(options.structural.doc)
        print(options.calibration.doc)
        print(options.feaext.doc)
        print(options.embsim.doc)
        print(options.ml.doc)
        print(options.brute_force.doc)
        print(options.tools.doc)
        print("\n")
        sys.exit(0)


    # Automatic steganalysis
    if sys.argv[1] == "auto":
        options.auto.auto()

    elif sys.argv[1] == "dci":
        options.auto.dci()

    # Structural LSB detectors
    elif sys.argv[1] == "sp":
        options.structural.sp()

    elif sys.argv[1] == "ws":
        options.structural.ws()

    elif sys.argv[1] == "triples":
        options.structural.triples()

    elif sys.argv[1] == "aump":
        options.structural.aump()
  
    elif sys.argv[1] == "spa":
        options.structural.spa()

    elif sys.argv[1] == "rs":
        options.structural.rs()


    # Calibration attacks
    elif sys.argv[1] == "calibration":
        options.calibration.launch()


    # Feature extraction
    elif sys.argv[1] == "srm":
        options.feaext.srm()

    elif sys.argv[1] == "srmq1":
        options.feaext.srmq1()

    elif sys.argv[1] == "scrmq1":
        options.feaext.scrmq1()

    elif sys.argv[1] == "gfr":
        options.feaext.gfr()

    elif sys.argv[1] == "dctr":
        options.feaext.dctr()

    elif sys.argv[1] == "hill-sigma-spam-psrm":
        options.feaext.hill_sigma_spam_psrm()

    elif sys.argv[1] == "hill-maxsrm":
        options.feaext.hill_maxsrm()


    # Embedding simulators
    elif sys.argv[1] == "lsbr-sim":
        options.embsim.lsbr()

    elif sys.argv[1] == "lsbm-sim":
        options.embsim.lsbm()

    elif sys.argv[1] == "hugo-sim":
        options.embsim.hugo()

    elif sys.argv[1] == "wow-sim":
        options.embsim.wow()

    elif sys.argv[1] == "s-uniward-sim":
        options.embsim.s_uniward()

    elif sys.argv[1] == "s-uniward-color-sim":
        options.embsim.s_uniward_color()

    elif sys.argv[1] == "hill-sim":
        options.embsim.hill()

    elif sys.argv[1] == "j-uniward-sim":
        options.embsim.j_uniward()

    elif sys.argv[1] == "j-uniward-color-sim":
        options.embsim.j_uniward_color()

    elif sys.argv[1] == "ebs-sim":
        options.embsim.ebs()

    elif sys.argv[1] == "ebs-color-sim":
        options.embsim.ebs_color()

    elif sys.argv[1] == "ued-sim":
        options.embsim.ued()

    elif sys.argv[1] == "ued-color-sim":
        options.embsim.ued_color()

    elif sys.argv[1] == "nsf5-sim":
        options.embsim.nsf5()

    elif sys.argv[1] == "nsf5-color-sim":
        options.embsim.nsf5_color()

    elif sys.argv[1] == "experimental-sim":
        options.embsim.experimental()

    elif sys.argv[1] == "steghide-sim":
        options.embsim.steghide()

    elif sys.argv[1] == "steganogan-sim":
        options.embsim.steganogan()

    elif sys.argv[1] == "adv-lsbm-sim":
        options.embsim.adv_lsbm()

    elif sys.argv[1] == "adv-pm2-fix":
        options.embsim.adv_pm2_fix()

    # ML base steganaysis
    elif sys.argv[1] == "split-sets":
        options.ml.split_sets()

    elif sys.argv[1] == "split-sets-dci":
        options.ml.split_sets_dci()

    elif sys.argv[1] == "effnetb0":
        options.ml.effnetb0()

    elif sys.argv[1] == "effnetb0-score":
        options.ml.effnetb0_score()

    elif sys.argv[1] == "effnetb0-predict":
        options.ml.effnetb0_predict()

    elif sys.argv[1] == "effnetb0-dci-score":
        options.ml.effnetb0_dci_score()

    elif sys.argv[1] == "effnetb0-dci-predict":
        options.ml.effnetb0_dci_predict()

    elif sys.argv[1] == "esvm":
        options.ml.esvm()

    elif sys.argv[1] == "esvm-predict":
        options.ml.esvm_predict()

    elif sys.argv[1] == "e4s":
        options.ml.e4s()

    elif sys.argv[1] == "e4s-predict":
        options.ml.e4s_predict()

    elif sys.argv[1] == "ats":
        options.ml.ats()


    # Brute force passwords
    elif sys.argv[1] == "brute-force-f5":
        options.brute_force.f5()

    elif sys.argv[1] == "brute-force-stegosuite":
        options.brute_force.stegosuite()

    elif sys.argv[1] == "brute-force-steghide":
        options.brute_force.steghide()

    elif sys.argv[1] == "brute-force-outguess":
        options.brute_force.outguess()

    elif sys.argv[1] == "brute-force-openstego":
        options.brute_force.openstego()

    elif sys.argv[1] == "brute-force-generic":
        options.brute_force.generic()


    # Tools
    elif sys.argv[1] == "hpf":
        options.tools.hpf()

    elif sys.argv[1] == "print-diffs":
        options.tools.print_diffs()

    elif sys.argv[1] == "print-dct-diffs":
        options.tools.print_dct_diffs()

    elif sys.argv[1] == "print-pixels":
        options.tools.print_pixels()

    elif sys.argv[1] == "print-coeffs":
        options.tools.print_coeffs()

    elif sys.argv[1] == "rm-alpha":
        options.tools.rm_alpha()

    elif sys.argv[1] == "plot-histogram":
        options.tools.plot_histogram()

    elif sys.argv[1] == "plot-dct-histogram":
        options.tools.plot_dct_histogram()

    else:
        print("Wrong command!")



if __name__ == "__main__":
    main()





