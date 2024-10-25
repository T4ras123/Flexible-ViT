# Flexible Vision Transformer

This repository contains an implementation of the Vision Transformer (ViT) model as described in the paper by [Dosovitskiy et al. 2021](https://arxiv.org/abs/2010.11929v2).

## Model Overview

The Vision Transformer leverages the scalable architecture proposed in the original [Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762) paper, adapting it for image classification tasks.

![Model Architecture](./media/image.png)

## Key Changes

1. **Embedding Flattened Image Patches**: Instead of tokens, the model embeds flattened image patches.
2. **Class Token**: A class token is added to the sequence of patches.
3. **Layer Normalization**: Applied before Multi-Head Self-Attention (MHSA) and Multi-Layer Perceptron (MLP) blocks.

![Class Token](media/image1.png)

## Note

This implementation is for educational purposes, and no pre-trained weights are provided.

## Model Definition

The model is defined in [model.ipynb](https://github.com/T4ras123/Implement-16x16-ViT-Paper/blob/main/model.ipynb) and [train.py](https://github.com/T4ras123/Implement-16x16-ViT-Paper/blob/main/train.py). 

## Key classes include

- VisionTransformer
- Block
- MultiHeadAttention
- Head
- FeedForward

## Example

An example of how to use the model can be found in [model.ipynb](https://github.com/T4ras123/Implement-16x16-ViT-Paper/blob/main/model.ipynb).

References
Dosovitskiy et al. 2021
Vaswani et al. 2017
