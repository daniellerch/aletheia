
# Aletheia: an open-source toolbox for steganalysis
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05982/status.svg)](https://doi.org/10.21105/joss.05982)


## What is Aletheia?

Aletheia is an open source image steganalysis tool for the detection of hidden messages in images. To achieve its objectives, Aletheia uses state-of-the-art machine learning techniques. It is capable of detecting several different steganographic methods as for example F5, Steghide, LSB replacement, LSB matching and some kind of adaptive schemes.

Aletheia addresses two main needs. Firstly, it aims to provide specialized analysts with a tool that implements modern steganalysis algorithms, leveraging deep learning techniques. These algorithms are designed to effectively handle even the most advanced steganography techniques. Secondly, Aletheia serves as a valuable tool for researchers by simplifying the process of conducting experiments and comparing methods. It includes simulators for common algorithms as well as state-of-the-art steganography methods enabling researchers to prepare and evaluate their work efficiently.


## Examples:

#### JPEG images
```bash
./aletheia.py auto sample_images/alaska2jpg
...

                     Outguess  Steghide   nsF5  J-UNIWARD *
-----------------------------------------------------------
08929_nsf5.jpg          0.0      0.0     [1.0]     0.0   
74006.jpg               0.0      0.0      0.3      0.1   
45762_juniw.jpg         0.0      0.0      0.2     [0.6]  
76538_steghide.jpg      0.0     [1.0]     0.2     [0.7]  
04965.jpg               0.0      0.0      0.1      0.1   
35517_juniw.jpg         0.0      0.0      0.1     [0.8]  
64639_outguess.jpg     [1.0]    [1.0]    [0.9]    [0.6]  
01497_juniw.jpg         0.0      0.0      0.2     [0.7]  
72950_nsf5.jpg          0.0      0.0     [0.9]    [0.5]  
09098_steghide.jpg      0.0     [1.0]     0.4      0.0   
35800_outguess.jpg     [1.0]    [1.0]    [0.8]    [1.0]  
08452_outguess.jpg     [1.0]    [1.0]    [0.7]    [1.0]  
23199_steghide.jpg      0.0     [0.7]     0.5      0.2   
27733_nsf5.jpg          0.0      0.0     [0.9]     0.4   
01294.jpg               0.0      0.0      0.2     [0.5] 


* Probability of steganographic content using the indicated method.

```

#### Uncompressed images
```bash
$ ./aletheia.py auto sample_images/alaska2
...

                       LSBR      LSBM  SteganoGAN  HILL *
---------------------------------------------------------
25422.png               0.0       0.0      0.0      0.0   
27693_steganogan.png   [0.9]     [1.0]    [1.0]    [0.9]  
74051_hill.png          0.0       0.0      0.0     [0.9]  
36466_steganogan.png   [0.9]     [1.0]    [1.0]    [1.0]  
04686.png               0.0       0.0      0.0      0.0   
37831_lsbm.png         [1.0]     [1.0]     0.0     [0.7]  
34962_hill.png          0.0       0.0      0.0     [0.5]  
00839_hill.png          0.0      [0.8]     0.0     [1.0]  
74648_lsbm.png         [1.0]     [1.0]     0.0     [0.6]  
74664.png               0.0       0.0      0.0      0.0   
55453_lsbm.png         [0.6]     [0.9]     0.0     [0.9]  
67104_steganogan.png   [0.9]     [0.9]    [1.0]    [0.8] 

* Probability of steganographic content using the indicated method.

```


## Documentation

- [Installation](/INSTALL.md)
- Doc by example:
	* [Introduction to steganalysis using Aletheia](https://daniellerch.me/stego/aletheia/intro-en/)
    * [Identifying the steganographic scheme](https://daniellerch.me/stego/aletheia/identify-en)
	* [Practical attack on Steghide](https://daniellerch.me/stego/aletheia/steghide-attack-en/)
	* [Practical attack on F5](https://daniellerch.me/stego/aletheia/f5-attack-en/)
	* [Practical attack on LSB replacement: OpenStego and OpenPuff](https://daniellerch.me/stego/aletheia/lsbr-attack-en/)
   * [Solving Stego-Puzzles with Aletheia](https://daniellerch.me/stego/aletheia/stego-puzzles-en/)
   * [Comparison of Image Steganography Tools](https://daniellerch.me/stego/aletheia/tool-comparison-en/).
   * [Training models for Aletheia](https://daniellerch.me/stego/aletheia/training-en/).

- [Available models](/aletheia-models/README.md)
- [External resources](/EXTERNAL.md)
- [How to cite Aletheia?](/CITING.md)
- [How to contribute?](/CONTRIBUTING.md)
- [References](/REFERENCES.md)



