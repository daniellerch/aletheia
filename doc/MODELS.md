
### Detection using pre-trained models

We provide some pre-trained models to detect common steganography algorithms. You can find this models in the "models" folder. For example, you can use the model "effnetb0-A-alaska2-steghide.h5" to predict the probability that a set of images contain hidden information using Steghide:

```bash
./aletheia.py effnetb0-predict sample_images/alaska2jpg/ models/effnetb0-A-alaska2-steghide.h5 0
Loading models/effnetb0-A-alaska2-steghide.h5 ...
sample_images/alaska2jpg/01294.jpg 0.0
sample_images/alaska2jpg/04965.jpg 0.0
sample_images/alaska2jpg/08929_nsf5.jpg 0.001
sample_images/alaska2jpg/09098_steghide.jpg 0.992
sample_images/alaska2jpg/23199_steghide.jpg 0.655
sample_images/alaska2jpg/27733_nsf5.jpg 0.0
sample_images/alaska2jpg/72950_nsf5.jpg 0.0
sample_images/alaska2jpg/74006.jpg 0.0
sample_images/alaska2jpg/76538_steghide.jpg 0.982
```

The name of the file give some details about the model. First we find the deep learning network: effnetb0. Next we find the letter A for normal models that split into cover and stego, or the letter B for models that split into stego and double stego. Next we find "alaska2", the name of the image database used to train the model. Next, we find the embedding algorithm: Steghide. You can find some information about models in [models/README.md](/models/README.md).

Remember that the reliability of the prediction is highly dependent on the cover source. This means that if the images used to train are very different from the images we want to predict the result may not be accurate. 






