### Install

First you need to clone the GIT repository:

```bash
$ git clone https://github.com/daniellerch/aletheia.git
```

Inside the Aletheia directory you will find a requirements file for installing Python dependencies with pip:

```bash
$ sudo pip3 install -r requirements.txt 
```

Aletheia uses Octave so you need to install it and some dependencies. You will find the dependencies in the octave-requirements.txt file. In Debian based Linux distributions you can install the dependencies with the following commands. For different distros you can deduce the appropriate ones.

```bash
$ sudo apt-get install octave octave-image octave-signal
```

You can find other dependencies in the other-requirements.txt.

```bash
$ sudo apt-get install liboctave-dev imagemagick
```

There are some functionalities that require native code. At this moment, you have to make the compilation manually.

The JPEG toolbox:

```bash
$ cd external/jpeg_toolbox
$ make
$ cd ..
```

The maxSRM feature extractor:

```bash
$ cd external/maxSRM
$ make
$ cd ..
```


After that, you can execute Aletheia with:

```bash
$ ./aletheia.py 

./aletheia.py <command>

COMMANDS:

  Automatic steganalysis:
  - auto:      Try different steganalysis methods.

  Structural LSB detectors (Statistical attacks to LSB replacement):
  - sp:            Sample Pairs Analysis (Octave vesion).
  - ws:            Weighted Stego Attack.
  - triples:       Triples Attack.
  - spa:           Sample Pairs Analysis.
  - rs:            RS attack.

  Calibration attacks to JPEG steganography:
  - calibration:   Calibration attack to JPEG images.

  Feature extractors:
  - srm:           Full Spatial Rich Models.
  - srmq1:         Spatial Rich Models with fixed quantization q=1c.
  - scrmq1:        Spatial Color Rich Models with fixed quantization q=1c.
  - gfr:           JPEG steganalysis with 2D Gabor Filters.
  - hill-maxsrm:   Selection-Channel-Aware Spatial Rich Models for HILL.

  Embedding simulators:
  - lsbr-sim:             Embedding using LSB replacement simulator.
  - lsbm-sim:             Embedding using LSB matching simulator.
  - hugo-sim:             Embedding using HUGO simulator.
  - wow-sim:              Embedding using WOW simulator.
  - s-uniward-sim:        Embedding using S-UNIWARD simulator.
  - s-uniward-color-sim:  Embedding using S-UNIWARD color simulator.
  - j-uniward-sim:        Embedding using J-UNIWARD simulator.
  - j-uniward-color-sim:  Embedding using J-UNIWARD color simulator.
  - hill-sim:             Embedding using HILL simulator.
  - ebs-sim:              Embedding using EBS simulator.
  - ebs-color-sim:        Embedding using EBS color simulator.
  - ued-sim:              Embedding using UED simulator.
  - ued-color-sim:        Embedding using UED color simulator.
  - nsf5-sim:             Embedding using nsF5 simulator.
  - nsf5-color-sim:       Embedding using nsF5 color simulator.
  - steghide-sim:         Embedding using Steghide simulator.
  - steganogan-sim:       Embedding using SteganoGAN simulator.

  ML-based steganalysis:
  - split-sets:            Prepare sets for training and testing.
  - split-sets-dci:        Prepare sets for training and testing (DCI).
  - effnetb0:              Train a model with EfficientNet B0.
  - effnetb0-score:        Score with EfficientNet B0.
  - effnetb0-predict:      Predict with EfficientNet B0.
  - effnetb0-dci-score:    DCI Score with EfficientNet B0.
  - effnetb0-dci-predict:  DCI Predict with EfficientNet B0.
  - esvm:                  Train an ensemble of Support Vector Machines.
  - e4s:                   Train Ensemble Classifiers for Steganalysis.
  - esvm-predict:          Predict using eSVM.
  - e4s-predict:           Predict using EC.

  Find password by brute force using a list of passwords:
  - bf-generic:   Generic tool for finding the password using a command tool.

  Tools:
  - hpf:                   High-pass filter.
  - print-diffs:           Differences between two images.
  - print-dct-diffs:       Differences between the DCT coefficients of two JPEG images.
  - rm-alpha:              Opacity of the alpha channel to 255.
  - plot-histogram:        Plot histogram.
  - plot-histogram-diff:   Plot histogram of differences.
  - plot-dct-histogram:    Plot DCT histogram.

```





