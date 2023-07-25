#!/bin/bash


#  - sp:            Sample Pairs Analysis (Octave vesion).
#  - ws:            Weighted Stego Attack.
#  - triples:       Triples Attack.
#  - spa:           Sample Pairs Analysis.
#  - rs:            RS attack.



test(){
   echo "Testing $1 / $2 ..."
   ./aletheia.py $1 $2 
}


test spa sample_images/lena_lsbr.png
test rs sample_images/lena_lsbr.png




