---
title: 'Aletheia: an open-source toolbox for steganalysis'
tags:
  - steganalysis
  - Python
  - image processing
authors:
  - name: Daniel Lerch-Hostalot
    orcid: 0000-0003-2602-672X
    affiliation: "1, 2, 3"
  - name: David Megías
    orcid: 0000-0002-0507-7731
    affiliation: "1, 2, 3"
affiliations:
 - name: Internet Interdisciplinary Institute (IN3), Barcelona, Spain
   index: 1
 - name: Universitat Oberta de Catalunya (UOC), Barcelona, Spain
   index: 2
 - name: CYBERCAT-Center for Cybersecurity Research of Catalonia, Barcelona, Spain
   index: 3
   
date: 12 January 2024
bibliography: paper.bib
---

# Summary

Steganalysis is the practice of detecting the presence of hidden information
within digital media, such as images, audio, or video. It involves analyzing
the media for signs of steganography, which is a set of techniques used to conceal
information within the carrier file. Steganalysis techniques can include
statistical analysis, visual inspection, and machine learning algorithms to
uncover hidden data. The goal of steganalysis is to determine whether a
file contains covert information and potentially identify the steganographic
method used.

Steganalysis has become increasingly important in the face of rising spying
and stegomalware threats, particularly in the context of data exfiltration.
In this scenario, malicious actors leverage steganographic techniques to
conceal sensitive data within innocent-looking files, evading traditional
security measures. By detecting and analyzing such covert communication
channels, steganalysis helps to identify and prevent data exfiltration attempts,
safeguarding critical information and preventing it from falling into the wrong hands.

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
[@Guo:2014:UED;@Fridrich:2007:nsF5;@Li:2014:hill;@Holub:2014:uniward;@zhang:2019:steganogan],
enabling researchers to prepare and evaluate their work efficiently.


On the other hand, to the best of the authors' knowledge, Aletheia stands out
as the sole steganalysis tool currently available that incorporates the
latest detection techniques [@Lerch-Hostalot:2019;@Megias:2023]
specifically designed to address the challenges posed by Cover Source Mismatch
(CSM) in real-world steganalysis scenarios
[@Ker:2013:real_world]. This capability is particularly significant for
conducting effective steganalysis in practical applications.


# Description

Aletheia incorporates various image steganography simulators, as well as tools
for preparing datasets with different payload sizes using these simulators.
This enables researchers to prepare experiments for their articles. Therefore,
having access to the original implementations of the different simulators is
relevant. Since it is common for these implementations to be developed in
Matlab, Aletheia includes several of these simulators in its original code,
slightly modified to be executed using Octave. These simulators frequently 
have licenses that can be incompatible with the MIT license used by 
Aletheia. For this reason, this code is in an external repository and is downloaded
separately after a confirmation by the user.
Aletheia also implements other simulators directly in Python, the programming
language of Aletheia, as well as tools that directly utilize their binaries.



These simulators can be used to conduct experiments, as shown in the following 
example. Here, you can observe how the simulator uses the HILL algorithm 
and embeds a random payload, which ranges from 5% to 25% of the image's maximum 
capacity when hiding 1 bit per pixel, within images sourced from the "images" 
folder. The resulting data is then saved in the "experiment" folder.

```bash
./aletheia.py hill-color-sim images 0.05-0.25 experiment
```

Although Aletheia allows for the preparation of experiments using multiple
simulators, its primary objective is steganalysis. This is achieved through
the implementation of various structural attacks on LSB replacement, as well as
employing deep learning techniques with models optimized for a vast range of
steganography algorithms. These algorithms include both commonly used tools in
the real world and state-of-the-art steganographic methods.

Aletheia also offers automated tools that allow for a preliminary analysis,
greatly aiding the investigation of the steganalyst. 
For example, the automated analysis below showcases the modeled probabilities 
of each image being generated using various steganographic methods.


```verb
$ ./aletheia.py auto actors/A2/

                    Outguess  Steghide   nsF5  J-UNIWARD *
-----------------------------------------------------------
2.jpg                  [1.0]    [1.0]    [0.9]     0.3
4.jpg                  [1.0]    [1.0]    [0.7]     0.3
10.jpg                  0.0     [1.0]     0.3      0.2
6.jpg                   0.0     [1.0]     0.1      0.0
7.jpg                  [1.0]    [1.0]     0.3      0.1
8.jpg                   0.0     [1.0]     0.1      0.2
9.jpg                  [0.8]    [1.0]    [0.7]     0.1
1.jpg                  [1.0]    [1.0]    [0.8]     0.1
3.jpg                  [1.0]    [1.0]    [1.0]     0.3
5.jpg                   0.0      0.1     [0.7]    [0.6]

* Probability of steganographic content using the indicated method.
```

Aletheia offers many other functionalities for steganalysis that are not covered
in this article and can be found in Aletheia's documentation
([github.com/daniellerch/aletheia](https://github.com/daniellerch/aletheia)).
Some examples include calibration attacks, custom model preparation, high-pass
filters, image difference analysis using pixels and DCT coefficients, DCI 
techniques to deal with Cover Source Mismatch (CSM), and more.


# Acknowledgements

We acknowledge the funding obtained by the Detection
of fake newS on SocIal MedIa pLAtfoRms (DISSIMILAR) project
from the EIG CONCERT-Japan with grant PCI2020-120689-2
(Government of Spain), and the PID2021-125962OB-C31 “SECURING”
project granted by the Spanish Ministry of Science and Innovation.
We wish to express our sincere gratitude towards NVIDIA Corporation for their
generous donation of an NVIDIA TITAN Xp GPU card, which has been instrumental
in the training of our models.



# References
