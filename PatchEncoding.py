import torch

from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange


class PatchEncoding(nn.Module):
    def __init__(self, in_channels, patch_size, image_size, embed_size):
        super().__init__()
        self.patch_embed = nn.Sequential(
            # use the conv layer instead of the linear layer for better performance
            nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # class embedding
        self.cls_token = nn.Parameter(torch.rand((1, embed_size)))
        # position embedding
        self.position_token = nn.Parameter(torch.rand(((image_size // patch_size) ** 2 + 1, embed_size)))

    def forward(self, x):
        b, _, _, _ = x.shape
        out = self.patch_embed(x)
        # concat class token
        cls_tokens = repeat(self.cls_token, 'h e -> b h e', b=b)
        out = torch.cat([cls_tokens, out], dim=1)
        # add position token
        out = out + self.position_token

        return out
