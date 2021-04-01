

### Training new models

Modern steganography methods are very difficult to detect and there is not enough with simple statistical attacks. We need to use deep learning.

To use deep learning we need to prepare a training dataset, used to train our classifier. For this example we are going to use the dataset of the Alaska 2 competition, tha you can find [here](https://www.kaggle.com/c/alaska2-image-steganalysis).

## Preparing the dataset

Let's suppose that we have a folder with cover images and a folder with stego images, for a given steganographic algorithm. Then we can use the following command to prepare a dataset.

```bash
./aletheia.py split-sets <cover-dir> <stego-dir> <output-dir> <#valid> <#test>

     cover-dir:    Directory containing cover images
     stego-dir:    Directory containing stego images
     output-dir:   Output directory. Three sets will be created
     #valid:       Number of images for the validation set
     #test:        Number of images for the testing set
     seed:         Seed for reproducible results
```

The seed parameter allows us to generate the same split in different executions. One example of execution could be:

```bash
./aletheia.py split-sets COVER_DIR STEGO_DIR DS 1000 500 0 0

ls DS
test train valid
```

## Training

The command for training is "effnetb0":

```bash
./aletheia.py effnetb0 <trn-cover-dir> <trn-stego-dir> <val-cover-dir> <val-stego-dir> <model-file> [dev] [ES]

     trn-cover-dir:    Directory containing training cover images
     trn-stego-dir:    Directory containing training stego images
     val-cover-dir:    Directory containing validation cover images
     val-stego-dir:    Directory containing validation stego images
     model-name:       A name for the model
     dev:        Device: GPU Id or 'CPU' (default='CPU')
     ES:         early stopping iterations x1000 (default=100)
```


Now, we can train the new model with the following command:


```bash
./aletheia.py effnetb0 DS/train/cover DS/train/stego DS/valid/cover DS/valid/stego mymodel 0 10
```

The model files will be stored into the "models" folder.




