import argparse
import json
import math
import random
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint

from .models.fusion import ExpertAdapter
from .models.readout import expert_readout
from .predict import MODELS, encode, load_model


def batch_for(model, processor, records_root, row, max_length):
    with Image.open(records_root / row["image"]) as image:
        image = image.convert("RGB")
    prefix = encode(model, processor, image, row["question"])
    batch = encode(model, processor, image, row["question"], row["answer"])
    if batch["input_ids"].shape[1] > max_length:
        raise ValueError("Training record exceeds the sequence budget.")
    labels = batch["input_ids"].clone()
    labels[:, :prefix["input_ids"].shape[1]] = -100
    batch["labels"] = labels
    return batch, image


def _loss(family, model, batch):
    if family == "kimi":
        return model(**batch, use_cache=False).loss
    shifted = torch.nn.functional.pad(batch["labels"], (0, 1), value=-100)[..., 1:]
    positions = (shifted[0] != -100).nonzero().flatten()
    return model(**batch, use_cache=False, logits_to_keep=positions, shift_labels=shifted[:, positions]).loss


def _prepare_language(model, family, recompute):
    if family == "kimi":
        model.eval()
        language = model.language_model.train()
        for gate in language.modules():
            if gate.__class__.__name__ == "MoEGate" and gate.topk_method == "noaux_tc":
                gate.eval()

                def routed(hidden, original=gate.forward):
                    indices, weights, _ = original(hidden)
                    return indices, weights, weights.new_zeros(())

                gate.forward = routed
        if recompute:
            for layer in language.model.layers:
                def run(*args, original=layer.forward, **kwargs):
                    return checkpoint(original, *args, use_reentrant=False, **kwargs)

                layer.forward = run
    else:
        model.train()
        model.model.visual.eval()
        if recompute:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()


def _lora(model, family, settings):
    from peft import LoraConfig, TaskType, get_peft_model

    path = r".*language_model\.model\.layers" if family == "kimi" else r".*language_model\.layers"
    suffix = "|".join(map(re.escape, settings["target_modules"]))
    targets = rf"{path}\.\d+\.self_attn\.({suffix})$"
    return get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=settings["rank"], lora_alpha=settings["alpha"], lora_dropout=settings["dropout"], bias="none", target_modules=targets))


def _score(model, processor, family, root, rows, max_length):
    model.eval()
    losses = []
    with torch.inference_mode():
        for row in rows:
            batch, _ = batch_for(family, processor, root, row, max_length)
            losses.append(float(model(**batch, use_cache=False).loss))
    model.train()
    model.get_base_model().model.visual.eval()
    return sum(losses) / len(losses)


def train(args):
    config = json.loads(args.config.read_bytes())
    stage = args.stage
    family = args.model
    settings = config[stage]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tf32 = settings.get("tf32", False)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    model, processor, _ = load_model(family, args.base, args.lora, attention=settings["attention"])
    model.requires_grad_(False)
    _prepare_language(model, family, settings["gradient_checkpointing"])
    expert = None
    if stage == "lora":
        model = _lora(model, family, settings)
        parameters = [param for param in model.parameters() if param.requires_grad]
    else:
        expert = ExpertAdapter(config["hidden_size"])
        expert.load_frozen_expert(load_file(args.expert))
        expert.to(device="cuda:0", dtype=torch.bfloat16)
        expert.train_fusion(args.train_values == "p1")
        parameters = [param for param in expert.parameters() if param.requires_grad]

    rows = json.loads(args.records.read_bytes())
    validation = json.loads(args.validation.read_bytes()) if args.validation else []
    if stage == "lora" and family == "qwen7b" and not validation:
        raise ValueError("7B LoRA requires validation data for minimum-loss checkpoint selection.")
    accumulation = config["gradient_accumulation"]
    steps = math.ceil(len(rows) / accumulation)
    warmup = math.ceil(settings["warmup_ratio"] * steps)

    def lr_scale(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))

    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(config["seed"])).tolist() if settings["shuffle"] else range(len(rows))
    best = math.inf
    backends = {
        "sdpa_fast_only": [torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION],
        "sdpa_math": [torch.nn.attention.SDPBackend.MATH],
    }
    selected = backends.get(settings["attention"])
    attention = torch.nn.attention.sdpa_kernel(selected) if selected else nullcontext()
    args.output.mkdir(parents=True, exist_ok=True)
    with attention:
        for index, row_index in enumerate(order, 1):
            batch, image = batch_for(family, processor, args.image_root, rows[row_index], config["max_sequence_tokens"])
            with expert_readout(model, expert, family, image, batch):
                loss = model(**batch, use_cache=False).loss if family == "qwen7b" and stage == "lora" else _loss(family, model, batch)
                (loss / accumulation).backward()
            if index % accumulation == 0 or index == len(rows):
                torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step = scheduler.last_epoch
                if stage == "lora" and family == "qwen7b" and (step % settings["eval_steps"] == 0 or step == steps):
                    score = _score(model, processor, family, args.image_root, validation, config["max_sequence_tokens"])
                    if score < best:
                        model.save_pretrained(args.output, safe_serialization=True)
                        best = score

    if stage == "lora" and family != "qwen7b":
        model.save_pretrained(args.output, safe_serialization=True)
    elif stage == "fusion":
        save_file({key: value.detach().cpu().contiguous() for key, value in expert.state_dict().items()}, args.output / "fusion.safetensors")


def main():
    parser = argparse.ArgumentParser(description="Train a language LoRA or local expert fusion.")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("stage", choices=("lora", "fusion"))
    parser.add_argument("config", type=Path)
    parser.add_argument("records", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--expert", type=Path)
    parser.add_argument("--train-values", choices=("all", "p1"), default="all")
    args = parser.parse_args()
    if args.stage == "fusion" and (args.lora is None or args.expert is None):
        parser.error("fusion requires --lora and --expert")
    train(args)


if __name__ == "__main__":
    main()
