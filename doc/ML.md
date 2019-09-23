

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



