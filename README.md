
## What is Aletheia?

Aletheia is an open source image steganalysis tool for the detection of hidden messages in images. To achieve its objectives, Aletheia uses state-of-the-art machine learning techniques. It is capable of detecting several different steganographic methods as for example LSB replacement, LSB matching and some kind of adaptive schemes.


## Examples:

#### JPEG images
```bash
./aletheia.py auto sample_images/lena_f5.jpg

Checking for nsF5 ...
Probability of being stego: 0.847

Checking for Steghide ...
Probability of being stego: 0.0
```

#### Bitmap images
```bash
./aletheia.py auto sample_images/alaska2/74648_lsbm.png 
 
 Checking for LSB replacement ...
 Hidden data not found

 Checking for LSB matching ...
 Probability of being stego: 1.0
```


## Documentation

- [Install](/doc/INSTALL.md)
- [Statistical attacks to LSB replacement](/doc/LSBR.md)
- [Machine Learning based attacks](/doc/ML.md)
- [Using pre-built models](/doc/PRE-BUILT.md)
- [The ATS attack](/doc/ATS.md)
- [Brute-force password attack](/doc/BRUTE-FORCE.md)
- Examples:
	* [Steganalysis of OpenPuff](/doc/OPENPUFF.md)
	* [Steganalysis of OpenStego](/doc/OPENSTEGO.md)
	* [Steganalysis of F5](/doc/F5.md)
	* [Steganalysis of Steghide](/doc/STEGHIDE.md)
- [How to cite Aletheia?](/doc/CITING.md)
- [References](/doc/REFERENCES.md)



