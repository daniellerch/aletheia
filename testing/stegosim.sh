#!/bin/bash


test(){
   echo "Testing $1 / $2 ..."
   ./aletheia.py $1 $2 0.10 out
   rm -fr out
}


test lsbr-sim sample_images/lena.png
test lsbm-sim sample_images/lena.png
#test hugo-sim sample_images/lena_gs.png
test wow-sim sample_images/lena_gs.png
test s-uniward-sim sample_images/lena_gs.png
test s-uniward-color-sim sample_images/lena.png
test j-uniward-sim sample_images/lena_gs.jpg
test j-uniward-color-sim sample_images/lena.jpg
test hill-sim sample_images/lena_gs.png
test hill-color-sim sample_images/lena.png
test ebs-sim sample_images/lena_gs.jpg
test ebs-color-sim sample_images/lena.jpg
test ued-sim sample_images/lena_gs.jpg
test ued-color-sim sample_images/lena.jpg
test nsf5-sim sample_images/lena_gs.jpg
test nsf5-color-sim sample_images/lena.jpg
test steghide-sim sample_images/lena.jpg
test outguess-sim sample_images/lena.jpg
test steganogan-sim sample_images/lena.png




