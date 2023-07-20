---
title: 'Aletheia: an open-source toolbox for steganalysis'
tags:
  - steganalysis
  - Python
  - image processing
authors:
  - name: Daniel Lerch-Hostalot
    orcid: 0000-0003-2602-672X
    affiliation: 1
  - name: David Megías
    orcid: 0000-0002-0507-7731
    affiliation: 2
affiliations:
 - name: Universitat Oberta de Catalunya (UOC), Barcelona, Spain
   index: 1
 - name: Universitat Oberta de Catalunya (UOC), Barcelona, Spain
   index: 2
date: 1 August 2023
bibliography: paper.bib
---

# Summary

Steganalysis is the practice of detecting the presence of hidden information 
within digital media, such as images, audio, or video. It involves analyzing 
the media for signs of steganography, which is the technique used to conceal 
information within the carrier file. Steganalysis techniques can include 
statistical analysis, visual inspection, and machine learning algorithms to 
uncover the hidden data. The goal of steganalysis is to determine whether a 
file contains covert information and potentially identify the steganographic 
method used.

Steganalysis has become increasingly important in the face of rising espionage 
and stegomalware threats, particularly in the context of data exfiltration. 
In this scenario, malicious actors leverage steganographic techniques to 
conceal sensitive data within innocent-looking files, evading traditional 
security measures. By detecting and analyzing such covert communication 
channels, steganalysis helps identify and prevent data exfiltration attempts, 
safeguarding critical information from falling into the wrong hands. 

In recent years, there has been a significant growth in the interest of 
researchers towards the field of steganalysis. The application of deep learning
[@Boroumand:2019:SRNet;@Yousfi:2020:alaska2]
in steganalysis has opened up new avenues for research, 
leading to improved detection rates and enhanced accuracy. As the field 
continues to evolve, experts are actively exploring novel architectures and 
training methodologies to further refine the performance of deep learning-based 
steganalysis.


# Statement of need

Aletheia addresses two main needs. Firstly, it aims to provide 
specialized analysts with a tool that implements modern steganalysis algorithms, 
leveraging deep learning techniques. These algorithms are designed to 
effectively handle even the most advanced steganography techniques. Secondly, 
Aletheia serves as a valuable tool for researchers by simplifying the process 
of conducting experiments and comparing methods. It includes simulators for 
common algorithms 
[@Sharp:2001:lsbm;@Provos:2001:outguess;@Hetzl:2005:steghide]
as well as state-of-the-art steganography methods
[@Guo:2014:UED;@Fridrich:2007:nsF5;@Li:2014:hill;@Holub:2014:uniward;@zhang:2019:steganogan]
, enabling researchers to prepare and evaluate their work efficiently.


On the other hand, to the best of the authors' knowledge, Aletheia stands out 
as the sole steganalysis tool currently available that incorporates the 
latest detection techniques [@Lerch-Hostalot:2019;@Megias:2023]
specifically designed to address the challenges posed by Cover Source Mismatch 
(CSM) in real-world steganalysis scenarios
[@Ker:2013:real_world]. This capability is particularly significant for 
conducting effective steganalysis in practical applications.


# Description

Aletheia incorporates various image steganography simulators, as well as tools for preparing datasets with different payload sizes using these simulators. This enables researchers to prepare experiments for their articles. Therefore, having access to the original implementations of the different simulators is important. Since it is common for these implementations to be developed in Matlab, Aletheia includes several of these simulators in its original code, slightly modified to be executed using Octave.
Aletheia also implements other simulators directly in Python, the programming language of Aletheia as well as tools that directly utilize their binaries.


```bash
$ ./aletheia.py

./aletheia.py <command>

COMMANDS:

...

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

...
```

...






# Acknowledgements

We acknowledge the funding obtained by the Detection
of fake newS on SocIal MedIa pLAtfoRms (DISSIMILAR) project
from the EIG CONCERT-Japan with grant PCI2020-120689-2 (Gov-
ernment of Spain), and to the PID2021-125962OB-C31 “SECURING”
project granted by the Spanish Ministry of Science and Innovation.
We wish to express our sincere gratitude towards NVIDIA Corporation for their 
generous donation of an NVIDIA TITAN Xp GPU card, which has been instrumental 
in the training of our models.














# References
