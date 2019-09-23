
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


As you can see in the results, when a pixel of the cover image is even the performed operation is +1 and when a pixel of the cover image is odd the performed operation is -1. This is what happens when the embedding operation is LSB replacement. This anomaly has been exploited by several attacks [[1, 2, 3](/doc/REFERENCES.md)].

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

