import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from einops import repeat
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import numpy as np
import argparse
import os
from datetime import datetime
device = 'cuda' if torch.cuda.is_available() else 'cpu'

to_tensor = [Resize((144,144)), ToTensor()]


class Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms
    
    
  def __call__(self, image):
    for t in self.transforms:
      image = t(image)
      
    return image
  
  
def show_img(images, num_samples=20, columns=4):
  plt.figure(figsize=(15,15))
  idx = int(len(dataset)/num_samples)
  for i, img in enumerate(images):
    if i% idx == 0:
      plt.subplot(int(num_samples/columns)+1, columns, int(i/idx)+1)
      plt.imshow(to_pil_image(img[0]))
  

dataset = OxfordIIITPet(root=".", download=True, transform=Compose(to_tensor))

train_split = int(0.8 * len(dataset))
train, test = random_split(dataset, [train_split, len(dataset) - train_split])

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=True)


def save_model(model, optimizer, epoch, loss, path="ViT/models"):
    """Save model checkpoint with timestamp."""
    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'vit_checkpoint_epoch{epoch}_{timestamp}.pt'
    filepath = os.path.join(path, filename)
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(model, optimizer, checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


class PatchEmbedding(nn.Module):
  def __init__(self, in_channels=3, patch_size = 8, emb_size = 128):
    super().__init__()
    self.patch_size = patch_size
    self.projection = nn.Sequential(
      Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
      nn.Linear(patch_size*patch_size*in_channels, emb_size)
    )
    
  def forward(self, x: Tensor) -> Tensor:
    x = self.projection(x)
    return x 


class Block(nn.Module):
    def __init__(self, emb_dim=32) -> None:
        super().__init__()
        self.sa = Attention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)


    def forward(self, ix):
        ix = ix + self.ffwd(self.ln1(ix))
        ix = ix + self.sa(self.ln2(ix))
        return ix


class Attention(nn.Module):
  def __init__(self, dim=32, n_heads=2, dropout=0.1) -> None:
    super().__init__()
    self.n_heads = n_heads
    self.att = torch.nn.MultiheadAttention(
      embed_dim=dim,
      num_heads=n_heads,
      dropout=dropout
    )
    self.q = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    
  
  def forward(self, x):
    q = self.q(x)
    v = self.v(x)
    k = self.k(x)
    attn_output, _ = self.att(q, k, v)
    return attn_output


class FeedForward(nn.Sequential):
    def __init__(self, dim=32, hidden_dim=128, dropout = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


class ViT(nn.Module):
  def __init__(self, ch=3, img_size=144, patch_size=4, emb_dim=32,
                n_layers=6, out_dim=37, dropout=0.1, heads=2):
    super(ViT, self).__init__()

    self.channels = ch
    self.height = img_size
    self.width = img_size
    self.patch_size = patch_size
    self.n_layers = n_layers
    self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
    self.l_head = nn.Linear(emb_dim, out_dim)

    self.patch_embedding = PatchEmbedding(in_channels=ch, patch_size=patch_size, emb_size=emb_dim)
    num_patches = (img_size // patch_size) ** 2
    self.pos_embedding = nn.Parameter(
        torch.randn(1, num_patches + 1, emb_dim))
    self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))
    
    
  def forward(self, img):
    x = self.patch_embedding(img)
    b, n, _ = x.shape
    cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
    x = torch.cat([cls_tokens, x], dim=1)
    x += self.pos_embedding[:, :(n + 1)]

    x = self.blocks(x)

    return self.l_head(x[:, 0, :])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='.', help='Path to the data')
  parser.add_argument('-ch', '--channels', type=int, default=3, help='Number of channels in the input image')
  parser.add_argument('-ims', '--image_size', type=int, default=144, help='Size of the input image')
  parser.add_argument('-ps', '--patch_size', type=int, default=4, help='Size of the patch')
  parser.add_argument('-emd', '--emb_dim', type=int, default=32, help='Embedding dimension')
  parser.add_argument('-nl', '--n_layers', type=int, default=6, help='Number of layers')
  parser.add_argument('-od', '--out_dim', type=int, default=37, help='Output dimension')
  parser.add_argument('-dr', '--dropout', type=float, default=0.1, help='Dropout')
  parser.add_argument('-nh', '--heads', type=int, default=2, help='Number of heads')
  parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
  args = parser.parse_args()
  
  model = ViT(args.channels, args.image_size, args.patch_size, args.emb_dim, args.n_layers, args.out_dim, args.dropout, args.heads).to(device)
  optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
  criterion = nn.CrossEntropyLoss()
  
  for epoch in range(args.epochs):
      epoch_losses = []
      model.train()
      
      for step, (inputs, labels) in enumerate(train_dataloader):
          inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          epoch_losses.append(loss.item())
      
      if epoch % 10 == 0:
          print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
          epoch_losses = []
          avg_loss = np.mean(epoch_losses)
          save_model(model, optimizer, epoch, avg_loss)
          
          for step, (inputs, labels) in enumerate(test_dataloader):
              inputs, labels = inputs.to(device), labels.to(device)
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              epoch_losses.append(loss.item())
              
          avg_loss = np.mean(epoch_losses)
          save_model(model, optimizer, epoch, avg_loss)   
          print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))
          
  inputs, labels = next(iter(test_dataloader))
  inputs, labels = inputs.to(device), labels.to(device)
  outputs = model(inputs)

  print("Predicted classes", outputs.argmax(-1))
  print("Actual classes", labels)


if __name__=='__main__':
    main()