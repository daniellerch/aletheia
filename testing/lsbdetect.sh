#!/bin/bash


test(){
   echo "Testing $1 / $2 ..."
   ./aletheia.py $1 $2 
}


test spa sample_images/lena_lsbr.png
test rs sample_images/lena_lsbr.png
test ws sample_images/lena_lsbr.png
test triples sample_images/lena_lsbr.png




