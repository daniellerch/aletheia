#!/usr/bin/python


import sys
import json
from aletheia import attacks, imutils
#from cnn import net as cnn

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

    if len(sys.argv)!=3:
        print sys.argv[0], "<command> <image>\n"
        print "Commands: "
        print "  srm-extract:   Extract features using Spatial Rich Models."
        print "\n"
        sys.exit(0)


    # {{{ srm
    if sys.argv[1]=="srm":

        if not imutils.is_valid_image(sys.argv[2]):
            print "Please, provide a valid image"
            sys.exit(0)

        import numpy
        from aletheia import richmodels
        X = richmodels.SRM_extract(sys.argv[2])
        numpy.savetxt('srm.txt', X.reshape((-1,1)), delimiter=',') 

    # }}}


    if sys.argv[1]=="train-models":
        train_models()


if __name__ == "__main__":
    main()



