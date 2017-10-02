#!/usr/bin/python


import sys
import json
import os
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

    if len(sys.argv)<2:
        print sys.argv[0], "<command>\n"
        print "Commands: "
        print "  srm-extract:   Extract features using Spatial Rich Models."
        print "\n"
        sys.exit(0)


    # {{{ srm-extract
    if sys.argv[1]=="srm-extract":

        if len(sys.argv)!=4:
            print sys.argv[0], "srm-extract <image/dir> <output-file>\n"
            sys.exit(0)

        # Read filenames
        files=[]
        if os.path.isdir(sys.argv[2]):
            for dirpath,_,filenames in os.walk(sys.argv[2]):
                for f in filenames:
                    path=os.path.abspath(os.path.join(dirpath, f))
                    if not imutils.is_valid_image(path):
                        print "Warning, prease provide a valid image: ", f
                    else:
                        files.append(path)
        else:
            files=[sys.argv[2]]


        import numpy
        from aletheia import richmodels
        from multiprocessing.dummy import Pool as ThreadPool 

        if os.path.exists(sys.argv[3]):
            os.remove(sys.argv[3])

        def extract_and_save(path):
            X = richmodels.SRM_extract(path)
            X = X.reshape((1, X.shape[0]))
            with open(sys.argv[3], 'a+') as f_handle:
                numpy.savetxt(f_handle, X)
 
        pool = ThreadPool(len(files))
        results = pool.map(extract_and_save, files)
        pool.close()
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


    if sys.argv[1]=="train-models":
        train_models()


if __name__ == "__main__":
    main()



