
### Steganalysis of StegHide


[StegHide](http://steghide.sourceforge.net/) Steghide is a steganography program that is able to hide data in various kinds of image and audio-files. The color respectively sample-frequencies are not changed thus making the embedding resistant against first-order statistical tests. At the moment of writing these lines the last version available is v0.5.1.

If we hide a message in a JPEG image we can see that, there are no anomalies in the DCT coefficients produced by LSB replacement.


```bash
$ head -100 /dev/urandom | tr -dc A-Za-z0-9 > secret.txt
$ cp sample_images/lena.jpg sample_images/lena_steghide.jpg
$ steghide embed -cf sample_images/lena_steghide.jpg -ef secret.txt -p mypass
```

With Aletheia we can check the modifications performed by OpenStego in the DCT coefficients of the stego image:


```bash
$ ./aletheia.py print-dct-diffs sample_images/lena.jpg sample_images/lena_steghide.jpg

Channel 0:
[(2.0, 1.0, -1.0), (2.0, 1.0, -1.0), (74.0, 75.0, 1.0), (1.0, 2.0, 1.0), (2.0, 1.0, -1.0)]
[(2.0, 1.0, -1.0), (-3.0, -4.0, -1.0), (-65.0, -66.0, -1.0), (1.0, 2.0, 1.0), (-2.0, -1.0, 1.0)]
[(-2.0, -1.0, 1.0), (1.0, 2.0, 1.0), (-2.0, -1.0, 1.0), (-2.0, -1.0, 1.0), (-4.0, -3.0, 1.0)]
[(6.0, 7.0, 1.0), (-2.0, -1.0, 1.0), (6.0, 5.0, -1.0), (6.0, 5.0, -1.0), (-1.0, -2.0, -1.0)]
[(2.0, 1.0, -1.0), (-2.0, -1.0, 1.0), (1.0, 2.0, 1.0), (-1.0, -2.0, -1.0), (3.0, 2.0, -1.0)]
[(-1.0, -2.0, -1.0), (17.0, 18.0, 1.0), (2.0, 1.0, -1.0), (-2.0, -1.0, 1.0), (2.0, 1.0, -1.0)]
[(7.0, 6.0, -1.0), (10.0, 9.0, -1.0), (2.0, 1.0, -1.0), (11.0, 10.0, -1.0), (-2.0, -1.0, 1.0)]
[(-2.0, -1.0, 1.0), (14.0, 13.0, -1.0), (5.0, 4.0, -1.0), (2.0, 1.0, -1.0), (-1.0, -2.0, -1.0)]
...
```

So in this case we are going to use a machine learning based approach.


... work in progress ...











