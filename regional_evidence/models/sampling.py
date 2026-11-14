from collections.abc import Sequence

import torch
from torch import nn


class LocalWindowSampler(nn.Module):

    def __init__(self, stage_windows: Sequence[int]=(8, 4, 2, 1)) -> None:
        super().__init__()
        if len(stage_windows) != 4 or any(window <= 0 for window in stage_windows):
            raise ValueError('stage_windows must contain four positive values.')
        self.stage_windows = tuple(int(window) for window in stage_windows)
        self.max_local_tokens = sum(window * window for window in self.stage_windows)

    def forward(self, projected_features: Sequence[torch.Tensor], query_centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(projected_features) != 4:
            raise ValueError('Expected four projected PVT stages.')
        if query_centers.ndim != 2 or query_centers.shape[-1] != 2:
            raise ValueError('query_centers must have shape [queries, 2].')
        local_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        relative_parts: list[torch.Tensor] = []
        stage_parts: list[torch.Tensor] = []
        query_count = query_centers.shape[0]
        for stage_index, (feature, window) in enumerate(zip(projected_features, self.stage_windows, strict=True)):
            if feature.ndim != 4 or feature.shape[0] != 1:
                raise ValueError('Each projected feature must have shape [1, D, H, W].')
            _, channels, height, width = feature.shape
            centers_x = torch.floor(query_centers[:, 0] * width).long().clamp(max=width - 1)
            centers_y = torch.floor(query_centers[:, 1] * height).long().clamp(max=height - 1)
            offsets_y, offsets_x = torch.meshgrid(torch.arange(window, device=feature.device), torch.arange(window, device=feature.device), indexing='ij')
            offsets_x = offsets_x.reshape(1, -1) - window // 2
            offsets_y = offsets_y.reshape(1, -1) - window // 2
            sample_x = centers_x[:, None] + offsets_x
            sample_y = centers_y[:, None] + offsets_y
            valid = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
            safe_x = sample_x.clamp(0, width - 1)
            safe_y = sample_y.clamp(0, height - 1)
            flat_indices = safe_y * width + safe_x
            flat_feature = feature[0].reshape(channels, height * width).transpose(0, 1)
            local = flat_feature[flat_indices]
            local = local.masked_fill(~valid.unsqueeze(-1), 0)
            sample_centers = torch.stack(((sample_x.to(torch.float32) + 0.5) / width, (sample_y.to(torch.float32) + 0.5) / height), dim=-1)
            relative = sample_centers - query_centers[:, None, :].to(sample_centers.dtype)
            local_parts.append(local)
            mask_parts.append(valid)
            relative_parts.append(relative)
            stage_parts.append(torch.full((query_count, window * window), stage_index, dtype=torch.long, device=feature.device))
        local_tokens = torch.cat(local_parts, dim=1)
        kv_mask = torch.cat(mask_parts, dim=1)
        relative_offsets = torch.cat(relative_parts, dim=1)
        stage_ids = torch.cat(stage_parts, dim=1)
        if local_tokens.shape[1] != self.max_local_tokens or not kv_mask.any(dim=1).all():
            raise RuntimeError('Local PVT sampling produced an invalid attention row.')
        return (local_tokens, kv_mask, relative_offsets, stage_ids)
