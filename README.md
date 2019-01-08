# Situation Recognition with Graph Neural Networks

This is the Torch implementation of [Situation Recognition with Graph Neural Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Situation_Recognition_With_ICCV_2017_paper.pdf)

## Setup
- Train a CNN to predict 504 verbs with Cross-Entropy loss
- Train a CNN to predict top 2K most frequent nouns with Binary Cross-Entropy loss
- Extract the CNN features for each image and save them together with roles information into a HDF5 file

## Training
Specify several options and then run `th train.lua`
- `-input_h5`: The input HDF5 file
- `-embedding_size`: Embedding size of verb and role
- `-rnn_size`: GGNN hidden state dimension
- `-num_updates`: Number of updates in GGNN
- `-checkpoint_path`: Where to save the models

## Test
There are two steps to test the model
1. Save the outputs by running `th eval.lua`, the options are the same as training
2. Calculate the accuracy `th clc_accuracy.lua`

## Cite
If you use this code, please consider citing
```bash
@inproceedings{li2017situation, 
title={Situation recognition with graph neural networks}, 
author={Li, Ruiyu and Tapaswi, Makarand and Liao, Renjie and Jia, Jiaya and Urtasun, Raquel and Fidler, Sanja}, 
booktitle={ICCV}, year={2017} 
}
```