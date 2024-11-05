# Flexible Vision Transformer

A flexible PyTorch implementation of the Vision Transformer (ViT) model for image classification tasks, inspired by the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

## Overview

This repository provides a modular and customizable Vision Transformer (ViT) model that adapts the Transformer architecture for image classification. By treating an image as a sequence of patches, the model leverages self-attention mechanisms to capture global contextual relationships within the image.

## Features

- **Patch Embedding**: Divides images into fixed-size patches and embeds them.
- **Positional Embedding**: Adds positional information to patch embeddings to retain spatial structure.
- **Transformer Encoder Blocks**: Utilizes multi-head self-attention and feed-forward networks with residual connections and layer normalization.
- **Classification Head**: Outputs class probabilities from the encoded features.
- **Configurable Parameters**: Easily adjust model dimensions, number of layers, attention heads, and more.
- **Checkpointing**: Save and load model checkpoints during training.
- **Visualization**: Utility functions to visualize image samples.

## Installation

### Clone the Repository and Install Dependencies

```sh
git clone https://github.com/T4ras123/Flexible-ViT.git
cd Flexible-ViT
pip install -r requirements.txt
```

### Install via PyPI

```sh
pip install vision-transformer
```

## Usage

### Training the Model

Train the ViT model using the provided `train.py` script with default parameters:

```sh
python train.py --data_path /path/to/dataset --epochs 100
```

#### Customizing Training Parameters

You can customize the training process by providing additional command-line arguments:

```sh
python train.py \
    --data_path ./data \
    --epochs 200 \
    --learning_rate 0.0005 \
    --batch_size 64 \
    --image_size 224 \
    --patch_size 16 \
    --emb_dim 768 \
    --n_layers 12 \
    --heads 12 \
    --dropout 0.1
```

#### Available Arguments

- `--data_path`: Path to the dataset.
- `--epochs`: Number of training epochs.
- `--learning_rate`: Learning rate for the optimizer.
- `--batch_size`: Number of samples per batch.
- `--image_size`: Dimension of input images (default: 144).
- `--patch_size`: Size of each image patch (default: 4).
- `--emb_dim`: Embedding dimension (default: 32).
- `--n_layers`: Number of Transformer encoder layers (default: 6).
- `--heads`: Number of attention heads (default: 2).
- `--dropout`: Dropout rate (default: 0.1).

### Loading a Saved Model

Load a previously saved model checkpoint:

```python
import torch
from ViT.train import ViT, load_model
import torch.optim as optim

model = ViT(
    ch=3,
    img_size=224,
    patch_size=16,
    emb_dim=768,
    n_layers=12,
    out_dim=1000,
    dropout=0.1,
    heads=12
).to('cuda')

optimizer = optim.AdamW(model.parameters(), lr=0.0005)
epoch, loss = load_model(model, optimizer, 'ViT/models/vit_checkpoint.pt')
```

### Evaluating the Model

Evaluate the trained model on the test dataset:

```sh
python evaluate.py --data_path /path/to/dataset --model_path ViT/models/vit_checkpoint.pt
```

## Model Architecture

The Vision Transformer model consists of the following components:

1. **Patch Embedding**: Converts input images into a sequence of flattened patch embeddings.
2. **Positional Embedding**: Adds positional information to each patch embedding.
3. **Transformer Encoder Blocks**: Comprises layers of multi-head self-attention and feed-forward networks with residual connections and layer normalization.
4. **Classification Head**: Maps the encoded features to output class probabilities.

### Key Components

- `PatchEmbedding`: Splits the image into patches and projects them into an embedding space.
- `Attention`: Implements multi-head self-attention mechanisms.
- `FeedForward`: A two-layer fully connected network with GELU activation and dropout.
- `Block`: Combines attention and feed-forward layers with layer normalization and residual connections.
- `ViT`: The main Vision Transformer model class that assembles all components.

### Example Code

```python
import torch
from ViT.train import ViT

model = ViT(
    ch=3,
    img_size=224,
    patch_size=16,
    emb_dim=768,
    n_layers=12,
    out_dim=1000,
    dropout=0.1,
    heads=12
)

inputs = torch.randn(1, 3, 224, 224)
outputs = model(inputs)
print(outputs.shape)  # torch.Size([1, 1000])
```

## Requirements

- Python ≥ 3.8
- PyTorch
- torchvision
- einops
- matplotlib
- numpy

### Install Dependencies

```sh
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929), 2021.
- [Vaswani et al., "Attention Is All You Need"](https://arxiv.org/abs/1706.03762), 2017.

## Citation

If you use this implementation in your research, please cite:

```md
@misc{vision-transformer,
  author       = {vover},
  title        = {Flexible Vision Transformer Implementation},
  year         = {2024},
  publisher    = {vover},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/T4ras123/Flexible-ViT}},
}
```
