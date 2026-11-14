import torch
from torch import nn

from .pvt_patch import PVTv2OverlapPatchEmbed


class PVTv2Stage(nn.Module):

    def __init__(self, patch_embed: PVTv2OverlapPatchEmbed, blocks: nn.ModuleList, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = blocks
        self.norm = norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states, height, width = self.patch_embed(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, height, width)
        hidden_states = self.norm(hidden_states)
        return hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
