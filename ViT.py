import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, Parts, in_channels, patch_size, image_size, cls_num, heads_num):
        super().__init__()
        self.embed_size = ((image_size // patch_size) ** 2) * in_channels
        # Patch Embedding
        self.PE = Parts[0](in_channels, patch_size, image_size, self.embed_size)
        # Transformer Encoder
        self.TF = Parts[2](Parts[1], self.embed_size, heads_num, 4)
        # Classification Head
        self.CLS = Parts[3](self.embed_size, cls_num)

    def forward(self, x):
        x = self.PE(x)
        for i in range(12):
            x = self.TF(x)
        x = self.CLS(x)
        return x
