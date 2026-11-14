import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from .models.fusion import ExpertAdapter
from .models.readout import expert_readout, fit_kimi_canvas

MODELS = {
    "qwen3b": ("Qwen/Qwen2.5-VL-3B-Instruct", 2048),
    "qwen7b": ("Qwen/Qwen2.5-VL-7B-Instruct", 3584),
    "qwen32b": ("Qwen/Qwen2.5-VL-32B-Instruct", 5120),
    "kimi": ("moonshotai/Kimi-VL-A3B-Instruct", 2048),
}


def seed_inference():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def configure_inference():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_inference()


def load_model(family, model_path, lora=None, fusion=None, attention=None):
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    checkpoint, hidden_size = MODELS[family]
    checkpoint = model_path or checkpoint
    if family == "kimi":
        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation=attention or "eager")
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(checkpoint, use_fast=False, min_pixels=256 * 28**2, max_pixels=1024 * 28**2)
    if lora:
        model = PeftModel.from_pretrained(model, lora).merge_and_unload(safe_merge=True)
    expert = None
    if fusion:
        expert = ExpertAdapter(hidden_size)
        expert.load_state_dict(load_file(fusion), strict=True)
        expert.to(device="cuda:0", dtype=torch.bfloat16).eval()
        expert.fusion.p1_only = True
    return model.eval(), processor, expert


def encode(family, processor, image, question, answer=None):
    if family == "kimi":
        image = fit_kimi_canvas(image, 1024)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=answer is None)
    return dict(processor(text=[text], images=[image], return_tensors="pt").to("cuda:0"))


def generation_truncated(tokens, max_tokens, eos_token_id):
    stop_ids = (eos_token_id,) if isinstance(eos_token_id, int) else eos_token_id or ()
    return len(tokens) == max_tokens and int(tokens[-1]) not in stop_ids


def predict(family, model, processor, expert, image, question, max_tokens):
    batch = encode(family, processor, image, question)
    if family != "kimi":
        model.model.rope_deltas = None
    sdpa = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if family != "kimi" else nullcontext()
    with torch.inference_mode(), sdpa, expert_readout(model, expert, family, image, batch):
        generated = model.generate(**batch, max_new_tokens=max_tokens, do_sample=False, num_beams=1, use_cache=True)
    tokens = generated[0, batch["input_ids"].shape[1]:]
    truncated = generation_truncated(tokens, max_tokens, model.generation_config.eos_token_id)
    return processor.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False), truncated


def main():
    parser = argparse.ArgumentParser(description="Run the original-image five-task interface.")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("records", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--fusion", type=Path)
    parser.add_argument("--values", choices=("all", "p1"), default="p1")
    args = parser.parse_args()
    configure_inference()
    model, processor, expert = load_model(args.model, args.base, args.lora, args.fusion)
    if expert:
        expert.fusion.p1_only = args.values == "p1"
    records = json.loads(args.records.read_bytes())
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            seed_inference()
            with Image.open(args.image_root / record["image"]) as source:
                reply, truncated = predict(args.model, model, processor, expert, source.convert("RGB"), record["question"], 384)
            output.write(json.dumps({"id": record["id"], "reply": reply, "truncated": truncated}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
