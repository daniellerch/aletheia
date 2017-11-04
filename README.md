# Aletheia
Aletheia is a tool for the detection of hidden messages in images.


- [Install](#install)
- [Statistical attacks to LSB replacement](#statistical-attacks-to-lsb-replacement)
- [Machine Learning based attacks](#machine-learning-based-attacks)



### Install

First you need to download or clone the GIT repository:

```bash
$ git clone https://github.com/daniellerch/aletheia.git
```

Inside the Aletheia directory you will find a requirements file for installing Python dependences with pip:

```bash
$ sudo pip install -r requirements.txt 
```

Aletheia uses Octave so you need to install it and some dependences. You will find the depdendences in the octave-requirements.txt file. In Debian based Linux distributions you can install the dependences with the following commands. For different distros you can deduce the appropriate ones.

```bash
$ sudo apt-get install octave octave-image
```

After that, you can execute Aletheia with:

```bash
$ ./aletheia.py <command>

COMMANDS:

  Attacks to LSB replacement:
  - spa:   Sample Pairs Analysis.
  - rs:    RS attack.

  Feature extractors:
  - srm:    Full Spatial Rich Models.
  - srmq1:  Spatial Rich Models with fixed quantization q=1c.

  Embedding simulators:
  - hugo-sim:       Embedding using HUGO simulator.
  - wow-sim:        Embedding using WOW simulator.
  - s-uniward-sim:  Embedding using S-UNIWARD simulator.
  - hill-sim:       Embedding using HILL simulator.

  Model training:
  - esvm:  Ensemble of Support Vector Machines.
  - e4s:   Ensemble Classifiers for Steganalysis.

```



### Statistical attacks to LSB replacement

### Machine Learning based attacks

