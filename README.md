
## What is Aletheia?

Aletheia is an open source image steganalysis tool for the detection of hidden messages in images. To achieve its objectives, Aletheia uses state-of-the-art machine learning techniques. It is capable of detecting several different steganographic methods as for example LSB replacement, LSB matching and some kind of adaptive schemes.


## Examples:

#### JPEG images
```bash
./aletheia.py auto sample_images/alaska2jpg
...

                        Outguess  Steghide  nsF5  J-UNIWARD *
-------------------------------------------------------------
08929_nsf5.jpg          0.0       0.0       1.0       0.0
74006.jpg               0.0       0.0       0.3       0.1
45762_juniw.jpg         0.0       0.0       0.2       0.6
76538_steghide.jpg      0.0       1.0       0.2       0.7
04965.jpg               0.0       0.0       0.1       0.1
35517_juniw.jpg         0.0       0.0       0.1       0.8
64639_outguess.jpg      1.0       1.0       0.9       0.6
01497_juniw.jpg         0.0       0.0       0.2       0.7
72950_nsf5.jpg          0.0       0.0       0.9       0.5
09098_steghide.jpg      0.0       1.0       0.4       0.0
35800_outguess.jpg      1.0       1.0       0.8       1.0
08452_outguess.jpg      1.0       1.0       0.7       1.0
23199_steghide.jpg      0.0       0.7       0.5       0.2
27733_nsf5.jpg          0.0       0.0       0.9       0.4
01294.jpg               0.0       0.0       0.2       0.5

* Probability of being stego using the indicated steganographic method.

```

#### Bitmap images
```bash
$ ./aletheia.py auto sample_images/alaska2
...

                        LSBR      LSBM   SteganoGAN   HILL *                                              
------------------------------------------------------------                                              
25422.png               0.0       0.0       0.0       0.0                                        
74051_hill.png          0.0       0.0       0.0       0.9                                        
04686.png               0.0       0.0       0.0       0.0                                        
37831_lsbm.png          1.0       1.0       1.0       0.7                                        
34962_hill.png          0.0       0.0       0.0       0.5                                        
01502_steganogan.png    0.8       1.0       1.0       1.0                                        
00839_hill.png          0.0       0.8       0.8       1.0                                        
22845_steganogan.png    0.8       1.0       1.0       1.0                                        
46694_steganogan.png    0.7       1.0       1.0       1.0                                        
74648_lsbm.png          1.0       1.0       1.0       0.6                                        
74664.png               0.0       0.0       0.0       0.0                                        
55453_lsbm.png          0.6       0.9       0.9       0.9  

* Probability of being stego using the indicated steganographic method.
```


## Documentation

- [Install](/doc/INSTALL.md)
- [Statistical attacks to LSB replacement](/doc/LSBR.md)
- [Detection using pre-trained models](/doc/MODELS.md)
- [Training new models](/doc/TRAIN_MODEL.md)
- [Brute-force password attack](/doc/BRUTE-FORCE.md)
- Examples:
	* [Steganalysis of OpenPuff](/doc/OPENPUFF.md)
	* [Steganalysis of OpenStego](/doc/OPENSTEGO.md)
	* [Steganalysis of F5](/doc/F5.md)
	* [Steganalysis of Steghide](/doc/STEGHIDE.md)
- [How to cite Aletheia?](/doc/CITING.md)
- [References](/doc/REFERENCES.md)



