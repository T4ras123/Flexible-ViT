import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

H, W, C = 224, 224, 3   # height x width x channels (3 - red, green, blue)
batch_size = 4096
P = 16                  # resolution of a patch
N = H*W//P**2            # Number of patches 
D = 768                 # Constant latent layer
lr = 2e-4
max_iters = 1000000
n_layers = 12
num_heads = 12
head_size = D//num_heads
num_classes = 1000


    

class VisionTransformer(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Embedding(C*P**2, D)
        self.positional_embedding = nn.Embedding(N+1, D)
        self.classification = nn.Parameter(torch.randn(1, 1, D))
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.l_head = nn.Linear(D, num_classes)

        
    def forward(self, ix, targets = None): # ix.shape (batch_size x N x (C*P**2)) 
        patch_emb = self.patch_embedding(ix)
        pos_emb = self.positional_embedding(torch.arange(N+1))
        cls_token = self.classification.expand(batch_size, -1, -1)  # (batch_size, 1, D)
        ix = torch.cat((cls_token, pos_emb), dim=1)
        ix = patch_emb + pos_emb
        ix = self.blocks(ix)
        logits = self.l_head(ix)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
            
class Block(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)


    def forward(self, ix):
        ix = ix + self.ffwd(self.ln1(ix))
        ix = ix + self.sa(self.ln2(ix))
        return ix
        
    
class Head(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.key = nn.Linear(D, head_size, bias=False)
        self.query = nn.Linear(D, head_size,  bias=False)
        self.value = nn.Linear(D, head_size,  bias=False)
        self.head_size = head_size
        
        
    def forward(self, ix): 
        k = self.key(ix)
        q = self.query(ix)
        
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = F.softmax(wei, dim=-1)
        v = self.value(ix)

        out = wei @ v
        
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = num_heads
        self.proj = nn.Linear(D, D)
        
        
    def forward(self, ix):
        out = torch.cat([ix for _ in range(self.heads)], dim=-1)
        return out

        
class FeedForward(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(D, 4*D)
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(4*D, D)
    
    
    def forward(self, ix):
        out = self.lin1(ix)
        out = self.tanh(out)
        out = ix + self.lin2(out)
        return out
    

if __name__=='__main__':

    model = VisionTransformer()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xb, yb = 0,0 # for lack of actual data

    for _ in range(max_iters):
    
        logits ,loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
