
## Folder structure


Throughout this document, we will explore the folder structure of the Aletheia project, specifically focusing on the significant directories. Each folder will be examined, providing an explanation of its contents and shedding light on its purpose and contribution to the overall structure. By highlighting these important directories, we aim to provide valuable insights into the project's architecture, aiding in efficient code navigation and comprehension. 


### [aletheia-models](/aletheia-models)
 
In this folder, the trained models used by Aletheia in all machine learning-based attacks can be found. The models encompass both deep learning models utilizing EfficientNet B0 and Ensemble Classifiers for Steganalysis (E4S).

By leveraging both deep learning models and ensemble classifiers, Aletheia possesses a diverse array of tools to tackle various attack forms and analyze digital media for potential steganographic content. The availability of the trained models in this folder forms the fundamental basis for Aletheia's machine learning-based attack capabilities. It is worth noting that these models are extensively utilized in research papers, further emphasizing their significance and effectiveness in the field.


### [aletheia-octave](/aletheia-octave)

This folder contains Octave code that implements state-of-the-art steganography methods and feature extractors used in steganalysis. Aletheia utilizes this code, which is originally written in Matlab, with slight modifications to make it compatible with Octave, as Octave is an open source tool.

While Aletheia primarily operates using Python as its main programming language, there are instances where it is crucial, particularly for research purposes and article publication, to have conducted experiments using the original tools provided by the respective authors. This includes utilizing the original Matlab implementation of these techniques, especially when working with steganography simulators. Having access to the authors' original tools ensures the integrity and reproducibility of the research findings, allowing for a comprehensive evaluation of the proposed methods.

In this folder, alongside Octave/Matlab code, you will also find C/C++ code. This is because Octave requires access to C/C++ extensions for tasks such as JPEG processing. By incorporating C/C++ code, Octave can effectively handle these operations, ensuring efficient JPEG processing within the software.

### [aletheia-resources](/aletheia-resources)

In the following folder, software for steganography is stored, which cannot be easily rewritten in Python. This is a common occurrence in steganography, as it often involves generating random positions within an image based on a key. Since this process heavily relies on the pseudo-random number generator used by each programming language, implementing compatible software can be challenging. Currently, the folder contains only one implementation of the F5 steganography algorithm, originally written in Java.

Additionally, this folder contains some essential resources, such as a list of passwords for brute-force attacks.

### [aletheialib](/aletheialib)

This folder contains the Python code that implements the various functionalities of Aletheia. 

### [actors](/actors) and [sample_images](/sample_images)

These folders contain sample images that can be used for testing purposes with Aletheia. The "sample_images" folder consists of individual images, whereas the "actors" folder contains images organized into groups. This organization helps simulate scenarios where a culpable or innocent actor sends multiple images, allowing for comprehensive testing and evaluation of Aletheia's capabilities in different scenarios.





