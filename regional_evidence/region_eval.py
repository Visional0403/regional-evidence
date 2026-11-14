import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from .kimi_regions import canvas_size, generate_kimi_region, kimi_region, masked
from .models.readout import expert_readout
from .predict import (
    MODELS,
    configure_inference,
    generation_truncated,
    load_model,
    seed_inference,
)
from .regions import (
    COUNT,
    PRESENCE,
    encode_qwen,
    generate_qwen_region,
    mask_boxes,
    prepared_image,
    qwen_region,
    visual_tokens,
)

REGIONS = ("top-left", "top-center", "top-right", "middle-left", "center", "middle-right", "bottom-left", "bottom-center", "bottom-right")
CATEGORY = (
    "Which vehicle categories are present in this image? In this exact order: bicycle, car, van, truck, tricycle, "
    "awning tricycle, bus, motorcycle. Return only a JSON array of exactly 8 integers: 1 if that category is present, "
    "0 if absent. Treat the categories separately. Include a partially visible vehicle only if its center is inside the image."
)


@torch.inference_mode()
def full_image_control(model, processor, image, row):
    from transformers import GenerationConfig

    canvases = {"original": image}
    canvases.update({name: mask_boxes(image, boxes) for name, boxes in row["mask_boxes"].items()})
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=16, bos_token_id=151643,
        eos_token_id=[151645, 151643], pad_token_id=151643, use_cache=True,
        temperature=1.0, top_p=1.0, top_k=0,
    )
    replies = {}
    for name, canvas in canvases.items():
        seed_inference()
        model.model.rope_deltas = None
        batch = encode_qwen(processor, canvas, row["question"], next(model.parameters()).device)
        tokens = model.generate(**batch, generation_config=generation)[0, batch["input_ids"].shape[1]:]
        reply = processor.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        replies[name] = (reply, generation_truncated(tokens, generation.max_new_tokens, generation.eos_token_id))
    return replies


def write_region(output, image_id, region_index, replies):
    output.write(json.dumps({
        "image_id": image_id,
        "region_index": region_index,
        "replies": {name: reply for name, (reply, _) in replies.items()},
        "truncated": {name: truncated for name, (_, truncated) in replies.items()},
    }) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the regional-token interface.")
    parser.add_argument("task", choices=("encoding", "category", "presence", "local_count", "full_count", "full_image_control"))
    parser.add_argument("split", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=MODELS, default="qwen7b")
    parser.add_argument("--base")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--fusion", type=Path)
    args = parser.parse_args()
    if args.model == "kimi" and (args.fusion or args.task in ("full_count", "full_image_control")):
        parser.error("The regional expert and full-image regional counts use Qwen.")
    if args.task == "full_image_control" and (args.model != "qwen7b" or args.lora or args.fusion):
        parser.error("The full-image control uses pretrained Qwen2.5-VL-7B.")
    configure_inference()
    model, processor, expert = load_model(args.model, args.base, args.lora, args.fusion)
    records = json.loads(args.split.read_bytes())
    if args.task == "presence" and args.lora:
        records = [row for row in records if row["split"] == "val"]
    kimi = args.model == "kimi"
    make_region, generate = (kimi_region, generate_kimi_region) if kimi else (qwen_region, generate_qwen_region)
    readout = {} if kimi else {"expert": expert}
    attention = nullcontext() if kimi else torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    with args.output.open("w", encoding="utf-8") as output, attention:
        for row in records:
            seed_inference()
            size = row["processed_size"]
            image = prepared_image(args.image_root / row["image"], canvas_size(*size) if kimi else size)
            region = row["region_index"]
            if args.task == "full_image_control":
                replies = full_image_control(model, processor, image, row)
                write_region(output, row["image_id"], region, replies)
                continue
            if args.task == "full_count":
                question = f"Divide the image into a 3 by 3 grid. How many vehicles are in the {REGIONS[region]} cell? Answer with a single non-negative integer."
                batch = encode_qwen(processor, image, question, next(model.parameters()).device)
                with torch.inference_mode(), expert_readout(model, expert, "qwen", image, batch):
                    tokens = visual_tokens(model, batch)
                replies = {"full_count": generate(model, processor, batch, tokens)}
                write_region(output, row["image_id"], region, replies)
                continue
            if args.task == "local_count":
                conditions, question = (("local_count", image, True),), COUNT
            elif args.task in ("encoding", "category"):
                conditions = (("crop_then_encode", image, False), ("encode_then_select", image, True))
                question = row.get("question", COUNT) if args.task == "encoding" else CATEGORY
                if args.model == "qwen32b":
                    question += " No explanation is needed."
            else:
                if kimi:
                    ivem = masked(image, size, region, row["exterior_boxes"], False)
                    imer = masked(image, size, region, row["internal_boxes"], True)
                    imem = masked(imer, size, region, row["exterior_boxes"], False)
                else:
                    ivem = mask_boxes(image, row["exterior_boxes"])
                    imer = mask_boxes(image, row["internal_boxes"])
                    imem = mask_boxes(imer, row["exterior_boxes"])
                conditions = (("IVER", image, True), ("IVEM", ivem, True), ("IMER", imer, True), ("IMEM", imem, True))
                question = row.get("question", PRESENCE)
            replies = {}
            for name, canvas, select in conditions:
                seed_inference()
                batch, tokens = make_region(model, processor, canvas, region, question, encode_then_select=select, **readout)
                replies[name] = generate(model, processor, batch, tokens, max_new_tokens=128 if args.task == "category" else 16)
            write_region(output, row["image_id"], region, replies)


if __name__ == "__main__":
    main()
