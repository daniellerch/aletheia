
### Steganalysis of StegHide


[StegHide](http://steghide.sourceforge.net/) is a steganography program that is able to hide data in various kinds of image and audio-files. The color respectively sample-frequencies are not changed thus making the embedding resistant against first-order statistical tests. At the moment of writing these lines the last version available is v0.5.1.

We can prepare two images for testing with the following commands:


```bash
$ head -100 /dev/urandom | tr -dc A-Za-z0-9 > secret.txt
$ mkdir test
$ cp sample_images/lena.jpg test/lena.jpg
$ cp sample_images/lena.jpg test/lena_steghide.jpg
$ steghide embed -cf test/lena_steghide.jpg -ef secret.txt -p mypass
```

There are differents attacks that we can perform agains steghide. In this case we are going to use a pre-trained **deep learning** model. The output for each image is the probability of being stego.


```bash
./aletheia.py effnetb0-predict test/ models/effnetb0-A-alaska2-steghide-best.h5 0
laska2-steghide-best.h5 0
Loading models/effnetb0-A-alaska2-steghide-best.h5 ...
2/2 [==============================] - 8s 4s/step
test/lena.jpg 0.0
test/lena_steghide.jpg 1.0

```












