
### Steganalysis of StegHide


[StegHide](http://steghide.sourceforge.net/) is a steganography program that is able to hide data in various kinds of image and audio-files. The color respectively sample-frequencies are not changed thus making the embedding resistant against first-order statistical tests. At the moment of writing these lines the last version available is v0.5.1.

If we hide a message in a JPEG image we can see that, there are no anomalies in the DCT coefficients produced by LSB replacement.


```bash
$ head -100 /dev/urandom | tr -dc A-Za-z0-9 > secret.txt
$ cp sample_images/lena.jpg sample_images/lena_steghide.jpg
$ steghide embed -cf sample_images/lena_steghide.jpg -ef secret.txt -p mypass
```

With Aletheia we can check the modifications performed by Steghide in the DCT coefficients of the stego image:


```bash
$ ./aletheia.py print-dct-diffs sample_images/lena.jpg sample_images/lena_steghide.jpg

Channel 0:
[(2.0, 1.0, -1.0), (-2.0, -1.0, 1.0), (-2.0, -1.0, 1.0), (71.0, 70.0, -1.0), (112.0, 111.0, -1.0)]
[(105.0, 106.0, 1.0), (2.0, 1.0, -1.0), (-1.0, -2.0, -1.0), (1.0, 2.0, 1.0), (-2.0, -1.0, 1.0)]
[(-2.0, -1.0, 1.0), (-4.0, -5.0, -1.0), (-4.0, -3.0, 1.0), (1.0, 2.0, 1.0), (2.0, 1.0, -1.0)]
[(-1.0, -2.0, -1.0), (-2.0, -1.0, 1.0), (12.0, 11.0, -1.0), (2.0, 1.0, -1.0), (5.0, 4.0, -1.0)]
[(-1.0, -2.0, -1.0), (-2.0, -1.0, 1.0), (-18.0, -17.0, 1.0), (-47.0, -48.0, -1.0), (-3.0, -2.0, 1.0)]
[(-2.0, -1.0, 1.0), (-2.0, -1.0, 1.0), (5.0, 6.0, 1.0), (2.0, 1.0, -1.0), (66.0, 67.0, 1.0)]
[(-1.0, -2.0, -1.0), (75.0, 74.0, -1.0), (6.0, 7.0, 1.0), (-3.0, -4.0, -1.0), (2.0, 3.0, 1.0)]
[(5.0, 4.0, -1.0), (-27.0, -26.0, 1.0), (-1.0, -2.0, -1.0), (-1.0, -2.0, -1.0), (-19.0, -20.0, -1.0)]
[(4.0, 5.0, 1.0), (-15.0, -14.0, 1.0), (-4.0, -5.0, -1.0), (-2.0, -1.0, 1.0), (1.0, 2.0, 1.0)]
[(1.0, 2.0, 1.0), (-1.0, -2.0, -1.0), (2.0, 1.0, -1.0), (-21.0, -22.0, -1.0), (4.0, 5.0, 1.0)]
[(5.0, 6.0, 1.0), (2.0, 1.0, -1.0), (1.0, 2.0, 1.0), (-4.0, -5.0, -1.0), (2.0, 1.0, -1.0)]
...
```

We can not see the anomalies produced by LSB replacement. So, in this case we are going to use the "calibration" attack. This attack uses a cropped version of the stego image to estimate the statistics of the cover image.

Actually, Steghide hide information by swapping the value of similar DCT coefficients to hold the first order statistics. But this swapping does not take into account the "mode" or relative position of each coefficient inside the block. So the method is vulnerable to calibration attacks.


```bash
./aletheia.py calibration sample_images/lena_steghide.jpg 
Hidden data found in channel 0: 0.06852303791779654
Hidden data found in channel 1: 0.06437836223405324
```

Obviously, with the original Lena image, the tool does not detect any hidden data:

```bash
./aletheia.py calibration sample_images/lena.jpg 
No hidden data found
```













