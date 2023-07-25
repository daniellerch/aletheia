#!/bin/bash


#  - srm:           Full Spatial Rich Models.
#  - srmq1:         Spatial Rich Models with fixed quantization q=1c.
#  - scrmq1:        Spatial Color Rich Models with fixed quantization q=1c.
#  - gfr:           JPEG steganalysis with 2D Gabor Filters.




test(){
   echo "Testing $1 / $2 ..."
   ./aletheia.py $1 $2 fea.csv
   rm -f fea.csv
}

test srm sample_images/alaska2 



