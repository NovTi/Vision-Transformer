import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.querys = nn.Linear(self.embed_size, self.embed_size)
        self.keys = nn.Linear(self.embed_size, self.embed_size)
        self.values = nn.Linear(self.embed_size, self.embed_size)
        self.out_project = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, x):
        # get the 8 keys, querys, and values
        # b: batch size  p: number of patches
        # c: changed embed size (one query vector dimension)  h: number of heads
        keys = rearrange(self.keys(x), 'b p (h c) -> b h p c', h=self.num_heads)
        querys = rearrange(self.querys(x), 'b p (h c) -> b h p c', h=self.num_heads)
        values = rearrange(self.values(x), 'b p (h c) -> b h p c', h=self.num_heads)
        # softmax of q \dot key.T
        # b: batch size   h: number of heads
        # c: changed embed size (one query vector dimension)
        # q: query patch numbers   k: keys patch numbers
        softmax_in = torch.einsum('b h q c, b h k c -> b h q k', querys, keys)
        softmax_divider = np.sqrt(self.embed_size)
        softmax_out = F.softmax(softmax_in, dim=-1) / softmax_divider
        # get the outputs
        # b: batch size   h: number of heads
        # c: changed embed size (one query vector dimension)
        out = torch.einsum('b h s p, b h p c -> b h s c', softmax_out, values)
        # b: batch size  p: number of patches  h: number of heads
        # c: changed embed size (one query vector dimension)
        out = rearrange(out, 'b h p c -> b p (h c)')
        out = self.out_project(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, SA, embed_size, num_heads, expand, drop_p: float = 0.):
        super().__init__()
        self.self_att = SA(embed_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * expand),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(embed_size * expand, embed_size),
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # first residual connection
        identity1 = x
        x = self.norm(x)
        x = self.self_att(x)
        x = x + identity1
        # second residual connection
        identity2 = x
        x = self.norm(x)
        x = self.mlp(x)
        x = x + identity2

        return x
