# Aletheia
Aletheia is a steganalysis tool for the detection of hidden messages in images.


- [Install](#install)
- [Statistical attacks to LSB replacement](#statistical-attacks-to-lsb-replacement)
- [Machine Learning based attacks](#machine-learning-based-attacks)
- [Using pre-build models](#using-pre-built-models)



### Install

First you need to clone the GIT repository:

```bash
$ git clone https://github.com/daniellerch/aletheia.git
```

Inside the Aletheia directory you will find a requirements file for installing Python dependencies with pip:

```bash
$ sudo pip install -r requirements.txt 
```

Aletheia uses Octave so you need to install it and some dependencies. You will find the dependencies in the octave-requirements.txt file. In Debian based Linux distributions you can install the dependencies with the following commands. For different distros you can deduce the appropriate ones.

```bash
$ sudo apt-get install octave octave-image
```

After that, you can execute Aletheia with:

```bash
$ ./aletheia.py 

./aletheia.py <command>

COMMANDS:

  Attacks to LSB replacement:
  - spa:   Sample Pairs Analysis.
  - rs:    RS attack.

  Feature extractors:
  - srm:    Full Spatial Rich Models.
  - srmq1:  Spatial Rich Models with fixed quantization q=1c.

  Embedding simulators:
  - hugo-sim:       Embedding using HUGO simulator.
  - wow-sim:        Embedding using WOW simulator.
  - s-uniward-sim:  Embedding using S-UNIWARD simulator.
  - hill-sim:       Embedding using HILL simulator.

  Model training:
  - esvm:  Ensemble of Support Vector Machines.
  - e4s:   Ensemble Classifiers for Steganalysis.

```


### Statistical attacks to LSB replacement

LSB replacement staganographic methods, that is, methods that hide information replacing the least significant bit of each pixel, are flawed. Aletheia implements two attacks to these methods: the Sample Pairs Analysis (SPA) and the RS attack.

To execute the SPA attack to an included image with LSB replacement data hidden, use the following command:

```bash
$./aletheia.py spa sample_images/lena_lsbr.png 
Hiden data found in channel R 0.0930809062336
Hiden data found in channel G 0.0923858529528
Hiden data found in channel B 0.115466382367
```

The command used to perform the RS attack is similar:

```bash
$./aletheia.py rs sample_images/lena_lsbr.png 
Hiden data found in channel R 0.215602586771
Hiden data found in channel G 0.210351910548
Hiden data found in channel B 0.217878287806
```

In both cases the results provides an estimation of the embedding rate. 



### Machine Learning based attacks

Most of the state of the art methods in Steganography use some kind of LSB matching. These methods are verify difficult to detect and there is not enough with simple statistical attacks. We need to use machine learning.

To use machine learning we need to prepare a training dataset, used to train our classifier. For this example we will use a database of grayscale images called Bossbase.

```bash
$ wget http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
$ unzip BOSSbase_1.01.zip
```

We are going to build a detector for the HILL algorithm with payload 0.40. So we need to prepare a set of images with data hidden using this algorithm. The following command embeds information into all the images downloaded:

```bash
$ ./aletheia.py hill-sim bossbase 0.40 bossbase_hill040 
```

With all the images prepared we need to extract features that can be processes by a machine learning algorithm. Aletheia provides different feature extractors, in this case we will use well known Rich Models. The following commands save the features into two files, on file for cover images and one file for stego images. 

```bash
$ ./aletheia.py srm bossbase bossbase.fea 
$ ./aletheia.py srm bossbase_hill040 bossbase_hill040.fea
```

Now, we can train the classifier. Aletheia provides different classifiers, in this case we will use Ensemble Classifiers:

```bash
$ ./aletheia.py e4s bossbase.fea bossbase_hill040.fea hill040.model
Validation score: 73.0
```

As a results, we obtain the score using a validation set (a small subset not used during training). The output is the file "hill040.model", so we can use this for future classifications.

Finally, we can classifiy an image:

```bash
$ ./aletheia.py e4s-predict hill040.model srm my_test_image.png
Stego, probability: 0.81
```


### Using pre-Built models

We provide some pre-built models to facilitate the usage of Aletheia. You can find this models in the "models" folder. For example, you can use the model "e4s_srm_bossbase_lsbm0.10.model" to classify an image with the following command:

```bash
$ ./aletheia.py e4s-predict e4s_srm_bossbase_lsbm0.10.model srm my_test_image.png
Stego, probability: 0.81
```

The name of the file give some details about the model. First we find the classification algorithm "e4s", used for Ensemble Classifiers for Steganalysis. Next we find the name of the feature extractor (srm for Spatial Rich Models). Next we find "bossbase", the name of the image database used to train the model. Finally, we find the embedding algorithm (lsbm, for LSB matching) and the embedding rate (0.10 bits per pixel). This information is needed to execute the program so we need to provide to Aletheia the classification algorithm used to predict (e4s-predict option) and the feature extractor used (srm).

Remember that the reliability of the prediction is highly dependent on the cover source. This means that if the images used to train are very different from the images we want to predict the result may not be accurate. 

You can find some information about the pre-build models [here](/aletheia/blob/master/models/README.md).














































