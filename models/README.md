
These models were trained using a randomly selected subset 
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



These models were trained using a randomly selected subset containing 
1000 images for validations, 1000 images for testing and the remaining images 
for training.


The Alaska2 database contains 75000 images and can be downloaded from 
[here](https://www.kaggle.com/c/alaska2-image-steganalysis).

The size of the message embedded is a random number between 100 bytes and 
the maximum capacity according to steghide.

| Model file                            | val score | test score | 
|---------------------------------------|-----------|------------|
| models/effnetb0-A-alaska2-lsbr.h5     |   0.925   |   0.913    |
| models/effnetb0-A-alaska2-steghide.h5 |   0.944   |   0.945    |
| models/effnetb0-B-alaska2-steghide.h5 |   0.762   |   0.697    |
| models/effnetb0-A-alaska2-nsf5.h5     |   0.788   |   0.762    |
| models/effnetb0-B-alaska2-nsf5.h5     |   0.676   |   0.680    |





