#!/bin/bash




test(){
   echo "Testing $1 / $2 ..."
   ./aletheia.py $1 $2 fea.txt
   wc -l fea.txt
   rm -f fea.txt
}

#test srm sample_images/alaska2 
test srmq1 sample_images/lena_gs.png
#test scrmq1 sample_images/alaska2 
#test gfr sample_images/alaska2jpg
#test dctr sample_images/alaska2jpg



