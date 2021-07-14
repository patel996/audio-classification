# audio-classification

## Downloading the audiosets from Google AudioSet

The code for accessing the audiosets from the Google AudioSet can be found in the audioset-preprocessing. It has a separate README file with detailed instructions to do so.

## Classification of Child And Adult Voice

The audios are converted to MelSpectograms and then a Vision image transformer is implemented for the classification purposes with the following parameters:

* image_size = 224
* patch_size = 16
* num_classes = 2
* dim = 768
* depth = 12
* heads = 12
* mlp_dim = 3072
* dropout = 0.0
* emb_dropout = 0.1

The file inference.py does live audio classfication.
