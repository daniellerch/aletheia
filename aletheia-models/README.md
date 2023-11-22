
## Color models

The following models were trained using a randomly selected subset containing 
1000 images for validations, 1000 images for testing and the remaining images 
for training.

The Alaska2 database contains 75000 images and can be downloaded from 
[here](https://www.kaggle.com/c/alaska2-image-steganalysis).
Models using Alaska2 in the spatial domain are made up of uncompressed JPEG images.

The Alaska database contains 78000 images and can be downloaded from
[here](https://alaska.utt.fr). In this case raw images are used.
In case of carrying out any transformation in the images, it is indicated 
in the name of the database.



| Model file (JPG images)                   | val score | test score |
|-------------------------------------------|-----------|------------|
| models/effnetb0-A-alaska2-steghide.h5     |   0.944   |   0.945    |
| models/effnetb0-B-alaska2-steghide.h5     |   0.762   |   0.697    |
| models/effnetb0-A-alaska2-outguess.h5     |   0.999   |   0.999    |
| models/effnetb0-B-alaska2-outguess.h5     |   0.851   |   0.856    |
| models/effnetb0-A-alaska2-nsf5.h5         |   0.825   |   0.809    |
| models/effnetb0-B-alaska2-nsf5.h5         |   0.702   |   0.668    |
| models/effnetb0-A-alaska2-jmipod.h5       |   0.873   |   0.837    |
| models/effnetb0-B-alaska2-jmipod.h5       |   0.716   |   0.670    |
| models/effnetb0-A-alaska2-juniw.h5        |   0.769   |   0.759    |
| models/effnetb0-B-alaska2-juniw.h5        |   0.701   |   0.698    |
| models/effnetb0-A-alaska2-juniw+wiener.h5 |   0.723   |   0.706    |
| models/effnetb0-B-alaska2-juniw+wiener.h5 |   0.732   |   0.714    |





| Model file (Uncompressed images)             | val score | test score |
|-----------------------------------------|-----------|------------|
| models/effnetb0-A-alaska2-lsbr.h5       |   0.925   |   0.913    |
| models/effnetb0-A-alaska2-lsbm.h5       |   0.913   |   0.918    |
| models/effnetb0-B-alaska2-lsbm.h5       |   0.838   |   0.849    |
| models/effnetb0-A-alaska2-hill.h5       |   0.942   |   0.923    |
| models/effnetb0-B-alaska2-hill.h5       |   0.910   |   0.873    |
| models/effnetb0-A-alaska-hill.h5        |   0.935   |   0.935    |
| models/effnetb0-B-alaska-hill.h5        |   0.850   |   0.846    |
| models/effnetb0-A-alaska-uniw.h5        |   0.895   |   0.893    |
| models/effnetb0-B-alaska-uniw.h5        |   0.811   |   0.811    |
| models/effnetb0-A-alaska2-hilluniw.h5   |   0.918   |   0.900    |
| models/effnetb0-B-alaska2-hilluniw.h5   |   0.841   |   0.807    |
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


| Models (JPG images)           | Testing set        | No CSM score | CSM score | DCI pred (+)|
|-------------------------------|--------------------|---------|-------|-------------|
| effnetb0-A/B-alaska2-steghide | bossbase-q80-color |  0.945  | 0.985 |   0.830     |
| effnetb0-A/B-alaska2-steghide | bossbase-q95-color |  0.945  | 0.978 |   0.804     |
| effnetb0-A/B-alaska2-steghide | imagenet-mini      |  0.945  | 0.813 |   0.695     |
| effnetb0-A/B-alaska2-steghide | lfw-faces          |  0.945  | 0.665 |   0.639     |
| effnetb0-A/B-alaska2-nsf5     | bossbase-q80-color |  0.809  | 0.771 |   0.780     |
| effnetb0-A/B-alaska2-nsf5     | bossbase-q95-color |  0.809  | 0.653 |   0.708     |
| effnetb0-A/B-alaska2-nsf5     | imagenet-mini      |  0.809  | 0.692 |   0.645     |
| effnetb0-A/B-alaska2-nsf5     | lfw-faces          |  0.809  | 0.667 |   0.687     |
| effnetb0-A/B-alaska2-juniw    | bossbase-q80-color |  0.759  | 0.566 |   0.595     |
| effnetb0-A/B-alaska2-juniw    | bossbase-q95-color |  0.759  | 0.700 |   0.685     |
| effnetb0-A/B-alaska2-juniw    | imagenet-mini      |  0.759  | 0.561 |   0.559     |
| effnetb0-A/B-alaska2-juniw    | lfw-faces          |  0.759  | 0.558 |   0.515     |
| effnetb0-A/B-alaska2-jmipod   | bossbase-q80-color |  0.837  | 0.656 |   0.530     |
| effnetb0-A/B-alaska2-jmipod   | bossbase-q95-color |  0.837  | 0.853 |   0.621     |





| Models  (Uncompressed images)     | Testing        | No CSM score |  CSM score  | DCI pred (+) |
|-----------------------------|----------------|---------|----------|--------------|
| effnetb0-A/B-alaska2-lsbm   | bossbase-color |  0.918  |  0.606   |   0.596      |
| effnetb0-A/B-alaska2-lsbm   | imagenet-mini  |  0.918  |  0.584   |   0.558      |
| effnetb0-A/B-alaska2-hill   | bossbase-color |  0.923  |  0.503   |   0.583      |
| effnetb0-A/B-alaska2-hill   | imagenet-mini  |  0.923  |  0.750   |   0.753      |
| effnetb0-A/B-alaska2-uniw   | bossbase-color |  0.895  |  0.500   |   0.500      |
| effnetb0-A/B-alaska2-uniw   | imagenet-mini  |  0.895  |  0.702   |   0.701      |




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


| Models (Uncompressed images)     | Testing          | Payload | Score |
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



