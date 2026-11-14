from torch import nn


class PVTv2OverlapPatchEmbed(nn.Module):

    def __init__(self, patch_size=7, stride=4, padding=3, in_channels=3, embed_dim=32):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)

    def forward(self, x):
        x = self.projection(x)
        height, width = x.shape[-2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, height, width
