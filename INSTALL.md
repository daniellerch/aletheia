### Install

You can install Aletheia with the following command:

```bash
$ pip3 install git+https://github.com/daniellerch/aletheia
```

But some things might not work properly, because Aletheia has external dependencies.

Aletheia uses Octave, so you need to install it and some of its libraries. You will find the dependencies in the octave-requirements.txt file. In Debian based Linux distributions you can install the dependencies with the following commands. For different distros you can deduce the appropriate ones.

```bash
$ sudo apt-get install octave octave-image octave-signal octave-nan
```

You can find other dependencies in the other-requirements.txt.

```bash
$ sudo apt-get install liboctave-dev imagemagick steghide outguess
```

After that, you can execute Aletheia with:

```bash
$ ./aletheia.py 

COMMANDS:

  Automated tools:
  - auto:      Try different steganalysis methods.
  - dci:       Predicts a set of images using DCI evaluation.

  Structural LSB detectors (Statistical attacks to LSB replacement):
  - spa:           Sample Pairs Analysis.
  - rs:            RS attack.
  - ws:            Weighted Stego Attack.
  - triples:       Triples Attack.


  Calibration attacks to JPEG steganography:
  - calibration:   Calibration attack on F5.

  Feature extractors:
  - srm:           Full Spatial Rich Models.
  - srmq1:         Spatial Rich Models with fixed quantization q=1c.
  - scrmq1:        Spatial Color Rich Models with fixed quantization q=1c.
  - gfr:           JPEG steganalysis with 2D Gabor Filters.
  - dctr:          JPEG Low complexity features extracted from DCT residuals.


  Embedding simulators:
  - lsbr-sim:             LSB replacement simulator.
  - lsbm-sim:             LSB matching simulator.
  - hugo-sim:             HUGO simulator.
  - wow-sim:              WOW simulator.
  - s-uniward-sim:        Spatial UNIWARD simulator.
  - s-uniward-color-sim:  Spatial UNIWARD color simulator.
  - j-uniward-sim:        JPEG UNIWARD simulator.
  - j-uniward-color-sim:  JPEG UNIWARD color simulator.
  - hill-sim:             HILL simulator.
  - hill-color-sim:       HILL color simulator.
  - ebs-sim:              EBS simulator.
  - ebs-color-sim:        EBS color simulator.
  - ued-sim:              UED simulator.
  - ued-color-sim:        UED color simulator.
  - nsf5-sim:             nsF5 simulator.
  - nsf5-color-sim:       nsF5 color simulator.
  - steghide-sim:         Steghide simulator.
  - outguess-sim:         Outguess simulator.
  - steganogan-sim:       SteganoGAN simulator.

  ML-based steganalysis:
  - split-sets:            Prepare sets for training and testing.
  - split-sets-dci:        Prepare sets for training and testing (DCI).
  - create-actors:         Prepare actors for training and testing.
  - effnetb0:              Train a model with EfficientNet B0.
  - effnetb0-score:        Score with EfficientNet B0.
  - effnetb0-predict:      Predict with EfficientNet B0.
  - effnetb0-dci-score:    DCI Score with EfficientNet B0.
  - effnetb0-dci-predict:  DCI Prediction with EfficientNet B0.
  - esvm:                  Train an ensemble of Support Vector Machines.
  - e4s:                   Train Ensemble Classifiers for Steganalysis.
  - esvm-predict:          Predict using eSVM.
  - e4s-predict:           Predict using EC.
  - actor-predict-fea:     Predict features for an actor.
  - actors-predict-fea:    Predict features for a set of actors.

  Find password by brute force using a list of passwords:
  - brute-force-f5:            Brute force a password using F5
  - brute-force-steghide:      Brute force a password using StegHide
  - brute-force-outguess:      Brute force a password using Outguess
  - brute-force-openstego:     Brute force a password using OpenStego
  - brute-force-generic:       Generic tool for finding the password using a command

  Tools:
  - hpf:                   High-pass filter.
  - print-diffs:           Differences between two images.
  - print-dct-diffs:       Differences between the DCT coefficients of two JPEG images.
  - print-pixels:          Print a range of píxels.
  - print-coeffs:          Print a range of JPEG coefficients.
  - rm-alpha:              Opacity of the alpha channel to 255.
  - plot-histogram:        Plot histogram.
  - plot-histogram-diff:   Plot histogram of differences.
  - plot-dct-histogram:    Plot DCT histogram.

```





