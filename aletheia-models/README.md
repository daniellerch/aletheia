
## Color models

The following models were trained using a randomly selected subset containing 
1000 images for validations, 1000 images for testing and the remaining images 
for training.

The Alaska2 database contains 75000 images and can be downloaded from 
[here](https://www.kaggle.com/c/alaska2-image-steganalysis).



| Model file (JPG images)               | val score | test score |
|---------------------------------------|-----------|------------|
| models/effnetb0-A-alaska2-steghide.h5 |   0.944   |   0.945    |
| models/effnetb0-B-alaska2-steghide.h5 |   0.762   |   0.697    |
| models/effnetb0-A-alaska2-outguess.h5 |   0.999   |   0.999    |
| models/effnetb0-B-alaska2-outguess.h5 |   0.851   |   0.856    |
| models/effnetb0-A-alaska2-nsf5.h5     |   0.796   |   0.781    |
| models/effnetb0-B-alaska2-nsf5.h5     |   0.679   |   0.680    |
| models/effnetb0-A-alaska2-juniw.h5    |   0.769   |   0.759    |
| models/effnetb0-B-alaska2-juniw.h5    |   0.701   |   0.698    |





| Model file (Spatial images)             | val score | test score |
|-----------------------------------------|-----------|------------|
| models/effnetb0-A-alaska2-lsbr.h5       |   0.925   |   0.913    |
| models/effnetb0-A-alaska2-lsbm.h5       |   0.913   |   0.918    |
| models/effnetb0-B-alaska2-lsbm.h5       |   0.838   |   0.849    |
| models/effnetb0-A-alaska2-hill.h5       |   0.942   |   0.923    |
| models/effnetb0-B-alaska2-hill.h5       |   0.910   |   0.873    |
| models/effnetb0-A-alaska2-steganogan.h5 |   1.000   |   0.994    |
| models/effnetb0-B-alaska2-steganogan.h5 |   0.998   |   0.989    |


The A models split into cover and stego and the B models split into stego and
double stego. The later are used for DCI methods.



## DCI models

The following table contains the score using the Alaska2 testing set as a 
reference and the score with a different testing set. A mismatch between 
the images used for training and for testing (CSM, Cover Source Mimsatch) 
is a usual situation in real world cases that decreases significantly the
accuracy of the model. The table also contains the DCI predictions. These
predictions (made without tags) let us know if the model is reliable to be 
applied to the analyzed images.


| Models (JPG images)           | Testing set        | Alaska2 | Score | DCI pred (+)|
|-------------------------------|--------------------|---------|-------|-------------|
| effnetb0-A/B-alaska2-steghide | bossbase-q80-color |  0.945  | 0.985 |   0.830     |
| effnetb0-A/B-alaska2-steghide | bossbase-q95-color |  0.945  | 0.978 |   0.804     |
| effnetb0-A/B-alaska2-steghide | imagenet-mini      |  0.945  | 0.813 |   0.695     |
| effnetb0-A/B-alaska2-steghide | lfw-faces          |  0.945  | 0.665 |   0.639     |
| effnetb0-A/B-alaska2-nsf5     | bossbase-q80-color |  0.781  | 0.802 |   0.686     |
| effnetb0-A/B-alaska2-nsf5     | bossbase-q95-color |  0.781  | 0.716 |   0.696     |
| effnetb0-A/B-alaska2-nsf5     | imagenet-mini      |  0.781  | 0.686 |   0.638     |
| effnetb0-A/B-alaska2-nsf5     | lfw-faces          |  0.781  | 0.667 |   0.687     |
| effnetb0-A/B-alaska2-juniw    | bossbase-q80-color |  0.759  | 0.566 |   0.595     |
| effnetb0-A/B-alaska2-juniw    | bossbase-q95-color |  0.759  | 0.700 |   0.685     |
| effnetb0-A/B-alaska2-juniw    | imagenet-mini      |  0.759  | 0.561 |   0.559     |
| effnetb0-A/B-alaska2-juniw    | lfw-faces          |  0.759  | 0.558 |   0.515     |





| Models  (Bitmap images)     | Testing        | Alaska2 |  Score   | DCI pred (+) |
|-----------------------------|----------------|---------|----------|--------------|
| effnetb0-A/B-alaska2-lsbm   | bossbase-color |  0.918  |  0.606   |   0.596      |
| effnetb0-A/B-alaska2-hill   | bossbase-color |  0.923  |  0.503   |   0.583      |




(+) DCI predictions tell us about the reliability of the model for these images



## Secure payloads

The following table contains experiments with low payloads. The objective of 
these experiments is to find for which payloads the stegosystem is undetectable.


| Models (JPG images)        | Testing             | Payload | Score |
|----------------------------|---------------------|---------|-------|
| effnetb0-A-alaska2-juniw   | alaska2-test        |  0.05   | 0.537 |
| effnetb0-A-alaska2-juniw   | alaska2-test        |  0.10   | 0.603 |
| effnetb0-A-alaska2-juniw   | bossbase-q95-color  |  0.05   | 0.526 |
| effnetb0-A-alaska2-juniw   | mini-imagenet-test  |  0.05   | 0.509 |
| effnetb0-A-alaska2-juniw   | mini-imagenet-test  |  0.10   | 0.522 |
| effnetb0-A-alaska2-juniw   | mini-imagenet-test  |  0.20   | 0.553 |


| Models (Bitmap images)     | Testing          | Payload | Score |
|----------------------------|------------------|---------|-------|
| effnetb0-A-alaska2-hill    | alaska2-test     |  0.05   | 0.574 |
| effnetb0-A-alaska2-hill    | alaska2-test     |  0.10   | 0.768 |
| effnetb0-A-alaska2-hill    | bossbase-test    |  0.05   | 0.500 |
| effnetb0-A-alaska2-hill    | bossbase-test    |  0.10   | 0.502 |
| effnetb0-A-alaska2-hill    | camid-test       |  0.05   | 0.649 |





## Grayscale models

The following models were trained using a randomly selected subset 
containing 90% of the images and validated using the remaining 10%.


The Bossbase contains 10000 images and can be downloaded from 
[here](http://agents.fel.cvut.cz/stegodata/BossBase-1.01-cover.tar.bz2).


| Model file                            | val score |
|---------------------------------------|-----------|
| e4s_srm_bossbase_lsbr0.40_gs.model    |   0.99    |
| e4s_srm_bossbase_lsbr0.20_gs.model    |   0.95    |
| e4s_srm_bossbase_lsbr0.10_gs.model    |   0.92    |
| e4s_srm_bossbase_lsbr0.05_gs.model    |   0.87    |
| e4s_srm_bossbase_lsbm0.40_gs.model    |   0.98    |
| e4s_srm_bossbase_lsbm0.40b_gs.model   |   0.98    |
| e4s_srm_bossbase_lsbm0.40c_gs.model   |   0.96    |
| e4s_srm_bossbase_lsbm0.20_gs.model    |   0.96    |
| e4s_srm_bossbase_lsbm0.20b_gs.model   |   0.94    |
| e4s_srm_bossbase_lsbm0.20c_gs.model   |   0.94    |
| e4s_srm_bossbase_lsbm0.10_gs.model    |   0.91    |
| e4s_srm_bossbase_lsbm0.10b_gs.model   |   0.91    |
| e4s_srm_bossbase_lsbm0.10c_gs.model   |   0.91    |
| e4s_srm_bossbase_lsbm0.05_gs.model    |   0.88    |
| e4s_srm_bossbase_lsbm0.05b_gs.model   |   0.87    |
| e4s_srm_bossbase_lsbm0.05c_gs.model   |   0.86    |
| e4s_srm_bossbase_hill0.40_gs.model    |   0.72    |
| e4s_srm_bossbase_uniw0.40_gs.model    |   0.77    |



