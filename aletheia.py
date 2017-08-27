#!/usr/bin/python


import sys
from nn import cnn

if len(sys.argv)!=2:
    print(sys.argv[0], "<params>\n")
    sys.exit(0)

tr_cover='../WORKDIR/DL_TR_RK_HUGO_0.40_db_boss5000_50/A_cover'
tr_stego='../WORKDIR/DL_TR_RK_HUGO_0.40_db_boss5000_50/A_stego'
ts_cover='../WORKDIR/DL_TS_RK_HUGO_0.40_db_boss250_50/SUP/cover'
ts_stego='../WORKDIR/DL_TS_RK_HUGO_0.40_db_boss250_50/SUP/stego'

cnn.train(tr_cover, tr_stego, ts_cover, ts_stego, 'model.x')



