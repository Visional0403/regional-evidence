import math
from contextlib import contextmanager
from types import MethodType

import torch
from PIL import Image, ImageDraw

from .predict import generation_truncated
from .regions import crop_bounds, first_token_score, region_positions


def canvas_size(width, height):
    scale = min(1.0, 1008 / max(width, height), (4096 * 14**2 / (width * height))**0.5)

    def candidates(length):
        scaled = scale * length / 84
        return {max(1, math.floor(scaled)) * 84, max(1, math.ceil(scaled)) * 84}

    sizes = [(w, h) for w in candidates(width) for h in candidates(height) if min(w, h) >= 252 and max(w, h) <= 1008 and w * h <= 4096 * 14**2]
    return min(sizes, key=lambda size: (abs(math.log((size[0] / size[1]) / (width / height))), -size[0] * size[1]))


def masked(image, qwen_size, region, boxes, internal):
    mask = Image.new("1", tuple(qwen_size))
    draw = ImageDraw.Draw(mask)
    for left, top, right, bottom in boxes:
        draw.rectangle((left, top, right - 1, bottom - 1), fill=1)
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    bounds = crop_bounds(image, region)
    if internal:
        inside = mask.crop(bounds)
        mask = Image.new("1", image.size)
        mask.paste(inside, bounds[:2])
    else:
        left, top, right, bottom = bounds
        ImageDraw.Draw(mask).rectangle((left, top, right - 1, bottom - 1), fill=0)
    image = image.copy()
    image.paste((127, 127, 127), (0, 0, image.width, image.height), mask)
    return image


def encode(processor, image, question, device):
    prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}], tokenize=False, add_generation_prompt=True)
    values = processor(text=[prompt], images=[image], return_tensors="pt")
    return {key: tensor.to(device, dtype=torch.bfloat16) if tensor.is_floating_point() else tensor.to(device) for key, tensor in values.items()}


@torch.inference_mode()
def kimi_region(model, processor, image, region, question, *, encode_then_select=True):
    device = next(model.parameters()).device
    local = encode(processor, image.crop(crop_bounds(image, region)), question, device)
    if not encode_then_select:
        return local, model._extract_image_features(local["pixel_values"], local["image_grid_hws"])
    full = encode(processor, image, question, device)
    height, width = map(int, full["image_grid_hws"][0].tolist())
    tokens = model._extract_image_features(full["pixel_values"], full["image_grid_hws"])
    return local, tokens[region_positions(height // 2, width // 2, region)]


@contextmanager
def inject_kimi_tokens(model, tokens):
    def image_features(_self, *_args, **_kwargs):
        return tokens

    model._extract_image_features = MethodType(image_features, model)
    try:
        yield
    finally:
        del model._extract_image_features


@torch.inference_mode()
def generate_kimi_region(model, processor, batch, tokens, *, with_score=False, max_new_tokens=16):
    first = []

    def capture(_module, _inputs, logits):
        if not first:
            first.append(logits[0, -1].detach())

    hook = model.language_model.lm_head.register_forward_hook(capture) if with_score else None
    try:
        with inject_kimi_tokens(model, tokens):
            result = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True)
    finally:
        if hook is not None:
            hook.remove()
    ids = result[0, batch["input_ids"].shape[1]:]
    reply = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    truncated = generation_truncated(ids, max_new_tokens, model.generation_config.eos_token_id)
    return (reply, first_token_score(processor.tokenizer, first[0]), truncated) if with_score else (reply, truncated)
