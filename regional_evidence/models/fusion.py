from collections import OrderedDict

import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .expert import PVTBackbone
from .sampling import LocalWindowSampler


class LocalAttention(nn.Module):
    def __init__(self, hidden_size, fusion_dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads
        self.query_proj = nn.Linear(hidden_size, fusion_dim)
        self.key_proj = nn.Linear(fusion_dim, fusion_dim)
        self.value_proj = nn.Linear(fusion_dim, fusion_dim)
        self.out_proj = nn.Linear(fusion_dim, hidden_size)
        self.stage_embedding = nn.Embedding(4, fusion_dim)
        self.relative_position = nn.Linear(2, num_heads, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, queries, local, valid, offsets, stages, p1_only=False):
        n, slots, _ = local.shape
        local = local + self.stage_embedding(stages)
        query = self.query_proj(queries).view(n, self.num_heads, self.head_dim)
        key = self.key_proj(local).view(n, slots, self.num_heads, self.head_dim)
        value = self.value_proj(local)
        if p1_only:
            # Retain the contribution of every valid key to the shared softmax.
            value = value * (stages == 0).unsqueeze(-1).to(value.dtype)
        value = value.view(n, slots, self.num_heads, self.head_dim)
        scores = torch.einsum("qhd,qkhd->qhk", query, key) * self.head_dim**-0.5
        bias = self.relative_position(offsets.to(local.dtype)).permute(0, 2, 1)
        scores = (scores + bias).masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("qhk,qkhd->qhd", weights, value)
        return self.out_proj(context.reshape(n, -1))


class PyramidProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector_projections = nn.ModuleList(nn.Conv2d(c, 256, 1) for c in (32, 64, 160, 256))
        self.fpn = FeaturePyramidNetwork([256] * 4, 256, extra_blocks=LastLevelMaxPool())
        self.output_projections = nn.ModuleList(nn.Conv2d(256, 512, 1) for _ in range(4))

    def forward(self, stages):
        projected = OrderedDict(
            (str(i), projection(stage))
            for i, (projection, stage) in enumerate(zip(self.detector_projections, stages, strict=True))
        )
        pyramid = self.fpn(projected)
        return tuple(projection(pyramid[str(i)]) for i, projection in enumerate(self.output_projections))


class LocalFusion(nn.Module):
    def __init__(self, hidden_size, gamma=0.5, chunk_size=128):
        super().__init__()
        self.projector = PyramidProjector()
        self.sampler = LocalWindowSampler((8, 4, 2, 1))
        self.attention = LocalAttention(hidden_size)
        self.register_buffer("gamma", torch.tensor(gamma))
        self.chunk_size = chunk_size
        self.p1_only = False

    def forward(self, tokens, stages, centers):
        features = self.projector(stages)
        deltas = []
        for start in range(0, len(tokens), self.chunk_size):
            end = start + self.chunk_size
            local, valid, offsets, stages = self.sampler(features, centers[start:end])
            deltas.append(self.attention(tokens[start:end], local, valid, offsets, stages, self.p1_only))
        return tokens + self.gamma.to(tokens.dtype) * torch.cat(deltas)


class ExpertAdapter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.backbone = PVTBackbone()
        self.fusion = LocalFusion(hidden_size)

    def forward(self, tokens, normalized_image, height, width):
        with torch.no_grad():
            stages = self.backbone(normalized_image)
        centers = token_centers(height, width, tokens.device)
        return self.fusion(tokens, stages, centers)

    def train_fusion(self, p1_only=False):
        self.eval().requires_grad_(False)
        self.fusion.p1_only = p1_only
        self.fusion.projector.output_projections.train().requires_grad_(True)
        self.fusion.attention.train().requires_grad_(True)

    def load_frozen_expert(self, weights):
        self.backbone.load_state_dict({key.removeprefix("backbone."): item for key, item in weights.items() if key.startswith("backbone.")})
        self.fusion.projector.detector_projections.load_state_dict({key.removeprefix("detector_projections."): item for key, item in weights.items() if key.startswith("detector_projections.")})
        self.fusion.projector.fpn.load_state_dict({key.removeprefix("fpn."): item for key, item in weights.items() if key.startswith("fpn.")})


def token_centers(height, width, device):
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
        (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)
