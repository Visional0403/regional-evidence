from collections import OrderedDict

import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .pvt_blocks import DropPath, PVTv2MixFFN, PVTv2SRAAttention
from .pvt_patch import PVTv2OverlapPatchEmbed
from .pvt_stage import PVTv2Stage


class PVTBlock(nn.Module):
    def __init__(self, width, heads, reduction, expansion, drop_path):
        super().__init__()
        self.attn = PVTv2SRAAttention(width, heads, reduction)
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.ffn = PVTv2MixFFN(width, expansion)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.ffn(self.norm2(x), height, width))


class PVTBackbone(nn.Module):
    stage_channels = (32, 64, 160, 256)

    def __init__(self, drop_path_rate=0.0):
        super().__init__()
        self.stages = nn.ModuleList()
        rates = torch.linspace(0, drop_path_rate, 8).tolist()
        for i, (width, heads, expansion, reduction) in enumerate(
            zip(self.stage_channels, (1, 2, 5, 8), (8, 8, 4, 4), (8, 4, 2, 1))
        ):
            patch = PVTv2OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                padding=3 if i == 0 else 1,
                in_channels=3 if i == 0 else self.stage_channels[i - 1],
                embed_dim=width,
            )
            blocks = nn.ModuleList(
                PVTBlock(width, heads, reduction, expansion, rates[2 * i + j])
                for j in range(2)
            )
            self.stages.append(PVTv2Stage(patch, blocks, nn.LayerNorm(width, eps=1e-6)))

    def forward(self, pixels):
        result = []
        for stage in self.stages:
            pixels = stage(pixels)
            result.append(pixels)
        return tuple(result)


class DomainExpert(nn.Module):
    out_channels = 256

    def __init__(self):
        super().__init__()
        self.encoder = PVTBackbone()
        self.projections = nn.ModuleList(nn.Conv2d(c, 256, 1) for c in (32, 64, 160, 256))
        self.fpn = FeaturePyramidNetwork([256] * 4, 256, extra_blocks=LastLevelMaxPool())

    def forward(self, pixels):
        mean = pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        features = self.encoder((pixels - mean) / std)
        projected = OrderedDict(
            (str(i), projection(feature))
            for i, (projection, feature) in enumerate(zip(self.projections, features))
        )
        return self.fpn(projected)


def load_coco_backbone(backbone, checkpoint):
    """Read the official MMDetection PVTv2-B0 COCO backbone."""
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    converted = {}
    for i in range(4):
        old, new = f"backbone.layers.{i}", f"stages.{i}"
        for suffix in ("weight", "bias"):
            for a, b in (("patch_embed.projection", "0.projection"), ("patch_embed.norm", "0.norm"), ("norm", "2")):
                converted[f"{new}.{a}.{suffix}"] = source[f"{old}.{b}.{suffix}"]
        for j in range(2):
            a, b = f"{new}.blocks.{j}", f"{old}.1.{j}"
            for suffix in ("weight", "bias"):
                for norm in ("norm1", "norm2"):
                    converted[f"{a}.{norm}.{suffix}"] = source[f"{b}.{norm}.{suffix}"]
                q, k, v = source[f"{b}.attn.attn.in_proj_{suffix}"].chunk(3, dim=0)
                for name, tensor in zip(("query", "key", "value"), (q, k, v)):
                    converted[f"{a}.attn.{name}.{suffix}"] = tensor
                converted[f"{a}.attn.proj.{suffix}"] = source[f"{b}.attn.attn.out_proj.{suffix}"]
                if i < 3:
                    for name in ("sr", "norm"):
                        converted[f"{a}.attn.{name}.{suffix}"] = source[f"{b}.attn.{name}.{suffix}"]
                for layer in (0, 1, 4):
                    tensor = source[f"{b}.ffn.layers.{layer}.{suffix}"]
                    if suffix == "weight" and layer in (0, 4):
                        tensor = tensor.squeeze(-1).squeeze(-1)
                    converted[f"{a}.ffn.layers.{layer}.{suffix}"] = tensor
    backbone.load_state_dict(converted, strict=True)
