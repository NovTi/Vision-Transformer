from torch import nn
from einops.layers.torch import Reduce


class ClassificationHead(nn.Module):
    def __init__(self, embed_size, class_num):
        super().__init__()
        self.cls = nn.Sequential(
            # Rarrange('b n e -> b (n e)')
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, class_num),
        )

    def forward(self, x):
        x = self.cls(x)
        return x
