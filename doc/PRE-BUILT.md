
### Using pre-built models

We provide some pre-built models to facilitate the usage of Aletheia. You can find this models in the "models" folder. For example, you can use the model "e4s_srm_bossbase_lsbm0.10.model" to classify an image with the following command:

```bash
$ ./aletheia.py e4s-predict e4s_srm_bossbase_lsbm0.10_gs.model srm my_test_image.png
my_test_image.png Stego
```

The name of the file give some details about the model. First we find the classification algorithm "e4s", used for Ensemble Classifiers for Steganalysis. Next we find the name of the feature extractor (srm for Spatial Rich Models). Next we find "bossbase", the name of the image database used to train the model. Next, we find the embedding algorithm (lsbm, for LSB matching) and the embedding rate (0.10 bits per pixel). Finally, we find the tag "gs" or "color" depending on the type of images used to train the model. Part of this information is needed to execute the program so we need to provide to Aletheia the classification algorithm used to predict (e4s-predict option) and the feature extractor used (srm).

Remember that the reliability of the prediction is highly dependent on the cover source. This means that if the images used to train are very different from the images we want to predict the result may not be accurate. 

You can find some information about the pre-built models [here](/models/README.md).





