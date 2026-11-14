import math
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def fit_kimi_canvas(image, max_tokens):
    width, height = image.size
    unit = 28
    if math.ceil(width / unit) * math.ceil(height / unit) <= max_tokens:
        return image
    scale = min(1.0, math.sqrt(max_tokens * unit**2 / (width * height)))
    while True:
        width_out, height_out = max(56, int(width * scale)), max(56, int(height * scale))
        if math.ceil(width_out / unit) * math.ceil(height_out / unit) <= max_tokens:
            return image.resize((width_out, height_out), Image.Resampling.BICUBIC)
        scale *= 0.99


def _normalize(rgb):
    mean = rgb.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = rgb.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (rgb - mean) / std


def _kimi_image(batch):
    height, width = map(int, batch["image_grid_hws"][0].tolist())
    patches = batch["pixel_values"].float().reshape(height, width, 3, 14, 14)
    image = patches.permute(2, 0, 3, 1, 4).reshape(1, 3, height * 14, width * 14)
    # The native Kimi processor scales RGB from [0, 1] to [-1, 1].
    return _normalize((image + 1.0) * 0.5), height // 2, width // 2


def _qwen_image(image, batch):
    _, height, width = map(int, batch["image_grid_thw"][0].tolist())
    pixels = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)[None] / 255.0
    pixels = F.interpolate(pixels, (height * 14, width * 14), mode="bicubic", align_corners=False, antialias=True)
    return _normalize(pixels), height // 2, width // 2


@contextmanager
def expert_readout(model, expert, family, image, batch):
    if expert is None:
        yield
        return
    if family == "kimi":
        pixels, height, width = _kimi_image(batch)
        projector = model.multi_modal_projector
    else:
        pixels, height, width = _qwen_image(image, batch)
        projector = model.model.visual
    pixels = pixels.to(next(expert.parameters()))

    def augment(_module, _inputs, output):
        if family == "kimi":
            return expert(output, pixels, height, width)
        # Vision output has already been restored to image order after window attention.
        output.pooler_output = expert(output.pooler_output, pixels, height, width)
        return output

    handle = projector.register_forward_hook(augment)
    try:
        yield
    finally:
        handle.remove()
