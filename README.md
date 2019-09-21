# Aletheia


Aletheia is an open source image steganalysis tool for the detection of hidden messages in images. To achieve its objectives, Aletheia uses state-of-the-art machine learning techniques. It is capable of detecting several different steganographic methods as for example LSB replacement, LSB matching and some kind of adaptive schemes.




- [Install](#install)
- [Statistical attacks to LSB replacement](#statistical-attacks-to-lsb-replacement)
- [Machine Learning based attacks](#machine-learning-based-attacks)
- [Using pre-built models](#using-pre-built-models)
- [The ATS attack](#the-ats-attack)
- [Brute-force password attack](#brute-force-password-attack)
- [Case of study: OpenPuff](#case-of-study-openpuff)
- [Case of study: OpenStego](#case-of-study-openstego)
- [References](#references)


### Install

First you need to clone the GIT repository:

```bash
$ git clone https://github.com/daniellerch/aletheia.git
```

Inside the Aletheia directory you will find a requirements file for installing Python dependencies with pip:

```bash
$ sudo pip3 install -r requirements.txt 
```

Aletheia uses Octave so you need to install it and some dependencies. You will find the dependencies in the octave-requirements.txt file. In Debian based Linux distributions you can install the dependencies with the following commands. For different distros you can deduce the appropriate ones.

```bash
$ sudo apt-get install octave octave-image octave-signal
```

After that, you can execute Aletheia with:

```bash
$ ./aletheia.py 

./aletheia.py <command>

COMMANDS:

  Attacks to LSB replacement:
  - spa:   Sample Pairs Analysis.
  - rs:    RS attack.

  ML-based detectors:
  - esvm-predict:   Predict using eSVM.
  - e4s-predict:    Predict using EC.
  - srnet-predict:  Predict using SRNet.

  Feature extractors:
  - srm:           Full Spatial Rich Models.
  - hill-maxsrm:   Selection-Channel-Aware Spatial Rich Models for HILL.
  - srmq1:         Spatial Rich Models with fixed quantization q=1c.
  - scrmq1:        Spatial Color Rich Models with fixed quantization q=1c.
  - gfr:           JPEG steganalysis with 2D Gabor Filters.

  Embedding simulators:
  - lsbr-sim:             Embedding using LSB replacement simulator.
  - lsbm-sim:             Embedding using LSB matching simulator.
  - hugo-sim:             Embedding using HUGO simulator.
  - wow-sim:              Embedding using WOW simulator.
  - s-uniward-sim:        Embedding using S-UNIWARD simulator.
  - j-uniward-sim:        Embedding using J-UNIWARD simulator.
  - j-uniward-color-sim:  Embedding using J-UNIWARD color simulator.
  - hill-sim:             Embedding using HILL simulator.
  - ebs-sim:              Embedding using EBS simulator.
  - ebs-color-sim:        Embedding using EBS color simulator.
  - ued-sim:              Embedding using UED simulator.
  - ued-color-sim:        Embedding using UED color simulator.
  - nsf5-sim:             Embedding using nsF5 simulator.
  - nsf5-color-sim:       Embedding using nsF5 color simulator.

  Model training:
  - esvm:     Ensemble of Support Vector Machines.
  - e4s:      Ensemble Classifiers for Steganalysis.
  - srnet:    Steganalysis Residual Network.

  Unsupervised attacks:
  - ats:      Artificial Training Sets.

  Naive attacks:
  - brute-force:       Brute force attack using a list of passwords.
  - hpf:               High-pass filter.
  - imgdiff:           Differences between two images.
  - imgdiff-pixels:    Differences between two images (show pixel values).
  - rm-alpha:          Opacity of the alpha channel to 255.

  Tools:
  - prep-ml-exp:     Prepare an experiment for testing ML tools.


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
my_test_image.png Stego
```


### Using pre-built models

We provide some pre-built models to facilitate the usage of Aletheia. You can find this models in the "models" folder. For example, you can use the model "e4s_srm_bossbase_lsbm0.10.model" to classify an image with the following command:

```bash
$ ./aletheia.py e4s-predict e4s_srm_bossbase_lsbm0.10_gs.model srm my_test_image.png
my_test_image.png Stego
```

The name of the file give some details about the model. First we find the classification algorithm "e4s", used for Ensemble Classifiers for Steganalysis. Next we find the name of the feature extractor (srm for Spatial Rich Models). Next we find "bossbase", the name of the image database used to train the model. Next, we find the embedding algorithm (lsbm, for LSB matching) and the embedding rate (0.10 bits per pixel). Finally, we find the tag "gs" or "color" depending on the type of images used to train the model. Part of this information is needed to execute the program so we need to provide to Aletheia the classification algorithm used to predict (e4s-predict option) and the feature extractor used (srm).

Remember that the reliability of the prediction is highly dependent on the cover source. This means that if the images used to train are very different from the images we want to predict the result may not be accurate. 

You can find some information about the pre-built models [here](/models/README.md).


### The ATS attack

The ATS attack [[4](#references)] provides a mechanism to deal with Cover Source Mismatch. This is a problem produced by training with an incomplete dataset. Our database, no matter how big it is, does not contains a representation of all the types of images that exist. As a consequence, if the image we want to test is not well represented in our training set, the results are going to be wrong.

The ATS attack is unsupervised, that means that does not need a training database. This attack use the images we want to test (we need a number of images, this method can not be applied to only one image) to create an artificial training set that is used to train the classifier. But this method has an important drawback: we can only use it if we know the training set has some cover and some stego images. The method does not work if all the images are cover or all the images are stego.

For example, we prepare a folder "test_ats" with 10 cover and 10 stego images. We can apply the ATS attack with the following command: 

```bash
$ ./aletheia.py ats lsbm-sim 0.40 srm test_ats/

9903.pgm Stego
9993.pgm Stego
9909.pgm Cover
9996.pgm Stego
9904.pgm Cover
9905.pgm Stego
9907.pgm Cover
9998.pgm Stego
9900.pgm Cover
9999.pgm Stego
9990.pgm Stego
9995.pgm Stego
9991.pgm Stego
9994.pgm Stego
9901.pgm Cover
9906.pgm Cover
9997.pgm Stego
9908.pgm Cover
9992.pgm Stego
9902.pgm Cover
```

Note we are supposed to know (or to have a strong assumption) about which is the embedding algorithm and bitrate used by the steganographer.




### Brute-force password attack

Aletheia can find the password used to hide a message using brute force from a list of passwords. 


```bash
$ ./aletheia.py brute-force
./aletheia.py brute-force <unhide command> <passw file>

Example:
./aletheia.py brute-force 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt

```

We need to provide to Aletheia the command used to extract the secret message. This command needs a "&lt;PASSWORD&gt;" section that will be replaced by every password to try. The second parameter to provide to Aletheia is the list of passwords to try. Aletheia comes with a file that contains 1000000 passwords.

Let's use Aletheia to crack the known steganography tool Steghide. First we are going to hide a message into a JPEG image using steghide:


```bash
$ steghide embed -cf cover.jpg -sf image.jpg -p 12345ab
Hello World!
embedding standard input in "cover.jpg"... done
writing stego file "image.jpg"... done
```

Now, we can find the password using the following command:

```bash
$ ./aletheia.py brute-force 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt
Using 16 processes
Completed: 4.5%    
Password found: 12345ab
```

The secret message was:

```bash
$ cat output.txt
Hello World!
```


### Case of study: OpenPuff

[OpenPuff](https://embeddedsw.net/OpenPuff_Steganography_Home.html) is a proprietary tool for hiding information. At the moment of writing these lines the las version available is v4.0.1.

The tool asks for three different passwords, the carrier image (we use a PNG file) and the bitrate or payload used (we choose the minimum: 12%). For the experiment we use the [Lena](https://daniellerch.me/images/lena.png) image. After saving the output image, we can analyze the modified pixels.

With a simple experiment we can see that the method used for embedding is LSB replacement. That is, the tool hides the bits of the message by replacing the least significant bit (LSB) of the pixel. 

```bash
$ ./aletheia.py print-diffs lena.png stego.png

Channel 1:                                                                                                                            
[(226, 227, 1), (228, 229, 1), (223, 222, -1), (226, 227, 1)] 
[(229, 228, -1), (231, 230, -1), (235, 234, -1), (203, 202, -1)] 
[(170, 171, 1), (174, 175, 1), (175, 174, -1), (178, 179, 1)]
[(182, 183, 1), (194, 195, 1), (203, 202, -1), (197, 196, -1)]
...

Channel 2:
[(134, 135, 1), (127, 126, -1), (129, 128, -1), (130, 131, 1)]
[(150, 151, 1), (143, 142, -1), (145, 144, -1), (142, 143, 1)]
[(86, 87, 1), (65, 64, -1), (75, 74, -1), (78, 79, 1)]
[(92, 93, 1), (100, 101, 1), (103, 102, -1), (103, 102, -1)]
...

Channel 3:
[(133, 132, -1), (116, 117, 1), (119, 118, -1), (121, 120, -1)]
[(121, 120, -1), (98, 99, 1), (82, 83, 1), (90, 91, 1)]
[(87, 86, -1), (86, 87, 1), (92, 93, 1), (93, 92, -1)]
[(95, 94, -1), (93, 92, -1), (98, 99, 1), (93, 92, -1)]
...
```


As you can see in the results, when a pixel of the cover image is even the performed operation is +1 and when a pixel of the cover image is odd the performed operation is -1. This is what happens when the embedding operation is LSB replacement. This anomaly has been exploited by several attacks [[1, 2, 3](#references)].

Let's try the SPA attack:

```bash
$ ./aletheia.py spa stego.png 
Hiden data found in channel R 0.15
Hiden data found in channel G 0.15
Hiden data found in channel B 0.14
```

Obviously, with the original Lena image, the tool does not detect any hidden data:

```bash
$ ./aletheia.py spa lena.png 
No hiden data found
```


### Case of study: OpenStego


[OpenStego](https://www.openstego.com/) is a Java tool for hiding information in the spatial domain (steganography and watermarking). At the moment of writing these lines the last version available is v0.7.3.

First we download a copy of the Lena image, then we prepare a file with some secret data and finally we hide the message.

With a simple experiment we can see that the method used for embedding is LSB replacement. That is, we hide the bits of the message by replacing the least significant bit (LSB) of the pixel. Actually, the tool supports using several pixels per channel, but this is even more detectable.


```bash
$ wget http://daniellerch.me/images/lena.png
$ head -500 /dev/urandom | tr -dc A-Za-z0-9 > secret.txt
$ openstego embed -mf secret.txt -cf lena.png -sf stego.png
```

With Aletheia we can check the modifications performed by OpenStego in the stego image:


```bash
$ ./aletheia.py print-diffs lena.png stego.png

Channel 1:
[(226, 227, 1), (223, 222, -1), (221, 220, -1), (223, 222, -1)]
[(229, 228, -1), (234, 235, 1), (174, 175, 1), (180, 181, 1)]
[(190, 191, 1), (204, 205, 1), (202, 203, 1), (204, 205, 1)]
[(210, 211, 1), (208, 209, 1), (207, 206, -1), (204, 205, 1)]
...


Channel 2:                                                                                                                            
[(226, 227, 1), (223, 222, -1), (221, 220, -1), (223, 222, -1)] 
[(229, 228, -1), (234, 235, 1), (174, 175, 1), (180, 181, 1)]
[(190, 191, 1), (204, 205, 1), (202, 203, 1), (204, 205, 1)]
[(210, 211, 1), (208, 209, 1), (207, 206, -1), (204, 205, 1)]
...

Channel 3:                                                                                                                            
[(226, 227, 1), (223, 222, -1), (221, 220, -1), (223, 222, -1)]
[(229, 228, -1), (234, 235, 1), (174, 175, 1), (180, 181, 1)]
[(190, 191, 1), (204, 205, 1), (202, 203, 1), (204, 205, 1)]
[(210, 211, 1), (208, 209, 1), (207, 206, -1), (204, 205, 1)]
...
```


As you can see in the results, when a pixel of the cover image is even the performed operation is +1 and when a pixel of the cover image is odd the performed operation is -1. This is what happens when the embedding operation is LSB replacement. This anomaly has been exploited by several attacks [[1, 2, 3](#references)].

Let's try a RS attack:

```bash
$ ./aletheia.py rs stego.png 
Hiden data found in channel R 0.25
Hiden data found in channel G 0.24
Hiden data found in channel B 0.27
```

Let's try now with less data:

```bash
$ head -100 /dev/urandom | tr -dc A-Za-z0-9 > secret.txt
$ openstego embed -mf secret.txt -cf lena.png -sf stego.png
$ ./aletheia.py rs stego.png 
Hiden data found in channel R 0.06
Hiden data found in channel G 0.06
Hiden data found in channel B 0.07
```

Obviously, with the original Lena image, the tool does not detect any hidden data:

```bash
$ ./aletheia.py rs lena.png 
No hiden data found
```






## References
[1]. Attacks on Steganographic Systems. A. Westfeld and A. Pfitzmann. Lecture Notes in Computer Science, vol.1768, Springer-Verlag, Berlin, 2000, pp. 61−75. 

[2]. Reliable Detection of LSB Steganography in Color and Grayscale Images. Jessica Fridrich, Miroslav Goljan and Rui Du.
Proc. of the ACM Workshop on Multimedia and Security, Ottawa, Canada, October 5, 2001, pp. 27-30. 

[3]. Detection of LSB steganography via sample pair analysis. S. Dumitrescu, X. Wu and Z. Wang. IEEE Transactions on Signal Processing, 51 (7), 1995-2007.

[4]. Unsupervised Steganalysis Based on Artificial Training Sets. Daniel Lerch-Hostalot and David Megías. Engineering Applications of Artificial Intelligence, Volume 50, 2016, Pages 45-59.











