from contextlib import contextmanager
from types import MethodType

import torch
from PIL import Image, ImageDraw

from .models.readout import expert_readout
from .predict import generation_truncated

PRESENCE = (
    "Is there at least one visible vehicle in this image? Vehicles include cars, vans, "
    "trucks, buses, bicycles, motorcycles, tricycles, and awning tricycles. Answer only yes or no."
)
COUNT = (
    "How many vehicles are in this image? Count cars, vans, trucks, buses, bicycles, "
    "motorcycles, tricycles, and awning tricycles. Count a partially visible vehicle "
    "only if its center is inside the image. Answer with a single non-negative integer."
)


def mask_boxes(image, boxes):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for left, top, right, bottom in boxes:
        draw.rectangle((left, top, right - 1, bottom - 1), fill=(127, 127, 127))
    return image


def prepared_image(path, size):
    with Image.open(path) as source:
        return source.convert("RGB").resize(size, Image.Resampling.BICUBIC)


def crop_bounds(image, region):
    width, height = image.size
    row, column = divmod(region, 3)
    return (column * width // 3, row * height // 3, (column + 1) * width // 3, (row + 1) * height // 3)


def region_positions(height, width, region):
    row, column = divmod(region, 3)
    return [r * width + c for r in range(row * height // 3, (row + 1) * height // 3) for c in range(column * width // 3, (column + 1) * width // 3)]


def encode_qwen(processor, image, question, device):
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
        tokenize=False, add_generation_prompt=True,
    )
    return dict(processor(text=[prompt], images=[image], do_resize=False, return_tensors="pt").to(device))


def visual_tokens(model, batch):
    return model.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"]).pooler_output[0]


@contextmanager
def inject_qwen_tokens(model, tokens):
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    core = model.model

    def image_features(_self, *_args, **_kwargs):
        return BaseModelOutputWithPooling(pooler_output=(tokens,))

    core.get_image_features = MethodType(image_features, core)
    try:
        yield
    finally:
        del core.get_image_features


@torch.inference_mode()
def qwen_region(model, processor, image, region, question, *, encode_then_select=True, expert=None):
    device = next(model.parameters()).device
    local = encode_qwen(processor, image.crop(crop_bounds(image, region)), question, device)
    if not encode_then_select:
        return local, visual_tokens(model, local)
    full = encode_qwen(processor, image, question, device)
    _, height, width = map(int, full["image_grid_thw"][0].tolist())
    with expert_readout(model, expert, "qwen", image, full):
        tokens = visual_tokens(model, full)
    return local, tokens[region_positions(height // 2, width // 2, region)]


@torch.inference_mode()
def generate_qwen_region(model, processor, batch, tokens, *, with_score=False, max_new_tokens=16):
    model.model.rope_deltas = None
    with inject_qwen_tokens(model, tokens):
        result = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True, return_dict_in_generate=with_score, output_logits=with_score)
    sequence = result.sequences if with_score else result
    ids = sequence[0, batch["input_ids"].shape[1]:]
    reply = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    truncated = generation_truncated(ids, max_new_tokens, model.generation_config.eos_token_id)
    return (reply, first_token_score(processor.tokenizer, result.logits[0][0]), truncated) if with_score else (reply, truncated)


def first_token_score(tokenizer, logits):
    def mass(words):
        ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in words]
        return torch.logsumexp(logits[ids].double(), dim=0)

    return float(mass(("yes", "Yes", "YES")) - mass(("no", "No", "NO")))
