
### The ATS attack

The ATS attack [[4](/doc/REFERENCES.md)] provides a mechanism to deal with Cover Source Mismatch. This is a problem produced by training with an incomplete dataset. Our database, no matter how big it is, does not contains a representation of all the types of images that exist. As a consequence, if the image we want to test is not well represented in our training set, the results are going to be wrong.

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




