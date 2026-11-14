import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from .interventions import (
    block_positions,
    donor_effect,
    exchange_tokens,
    image_kv,
    joint_interaction,
    matched_random_blocks,
    random_tokens,
    spatial_tiles,
)
from .kimi_regions import canvas_size, generate_kimi_region, kimi_region, masked
from .predict import MODELS, configure_inference, load_model, seed_inference
from .regions import (
    PRESENCE,
    generate_qwen_region,
    mask_boxes,
    prepared_image,
    qwen_region,
)


def transfer(model, processor, paired, family, identity):
    kimi = family == "kimi"
    generate = generate_kimi_region if kimi else generate_qwen_region

    def run(arm, selected=None, donors=None, sources=None):
        seed_inference()
        batch, tokens = paired[arm]
        if selected is not None and donors is None:
            tokens = exchange_tokens(tokens, paired[1 - arm][1], selected)
        if donors is None:
            return generate(model, processor, batch, tokens, with_score=True)
        placeholder = model.config.media_placeholder_token_id if kimi else model.config.image_token_id
        positions = (batch["input_ids"][0] == placeholder).nonzero().flatten().tolist()
        with image_kv(positions, donors, selected, sources):
            return generate(model, processor, batch, tokens, with_score=True)

    batch = paired[0][0]
    height, width = (batch["image_grid_hws"][0] if kimi else batch["image_grid_thw"][0, 1:]).tolist()
    tiles = spatial_tiles(height // 2, width // 2)
    native = [run(arm) for arm in (0, 1)]
    if any(entry[2] for entry in native) or [entry[0].casefold().removesuffix(".") for entry in native] != ["yes", "no"]:
        return {"native": native}
    single = {tile: [run(arm, positions) for arm in (0, 1)] for tile, positions in tiles.items()}
    order = sorted(tiles, key=lambda block: (-donor_effect(native[0][1], native[1][1], single[block][0][1], single[block][1][1]), block))
    rank = [block_positions(tiles, order[:k]) for k in range(1, len(order) + 1)]
    namespace = "kimi_visdrone_v4019_random" if kimi else "local_token_causal_v1"
    random = [[random_tokens(identity, len(paired[0][1]), len(ids), size, repeat, namespace) for repeat in range(3)] for size, ids in enumerate(rank, 1)]
    grid = [1, height, width]
    match = [block_positions(tiles, choice) for choice in matched_random_blocks(tiles, width // 2, order[:3], identity, "v402_spatial_block_control_v1")]
    placeholder = model.config.media_placeholder_token_id if kimi else model.config.image_token_id
    saved = []
    for arm in (0, 1):
        positions = (paired[arm][0]["input_ids"][0] == placeholder).nonzero().flatten().tolist()
        with image_kv(positions) as state:
            run(arm)
        saved.append(state)
    combinations = ((0, 0), (1, 0), (0, 1), (1, 1))
    fixed = {
        sources: [run(arm, rank[2], saved, sources) for arm in (0, 1)]
        for sources in combinations
    }
    labels = ("IMER", "IMEM")
    scores = [[fixed[sources][arm][1] for sources in combinations] for arm in (0, 1)]
    return {
        "native": native,
        "grid": grid,
        "order": order,
        "single_blocks": single,
        "ranked": [[run(arm, ids) for arm in (0, 1)] for ids in rank],
        "random_tokens": [[[run(arm, ids) for arm in (0, 1)] for ids in groups] for groups in random],
        "matched_blocks": [[run(arm, ids) for arm in (0, 1)] for ids in match],
        "top3_kv": {
            "K": [fixed[1, 0][0], fixed[0, 1][1]],
            "V": [fixed[0, 1][0], fixed[1, 0][1]],
            "KV": [fixed[1, 1][0], fixed[0, 0][1]],
        },
        "top3_kv_sources": {
            f"{labels[k]}/{labels[v]}": fixed[k, v] for k, v in combinations
        },
        "top3_kv_interaction": dict(zip(labels, joint_interaction(*scores), strict=True)),
        "top3_random_kv": [[run(arm, ids, saved, (1 - arm, 1 - arm)) for arm in (0, 1)] for ids in random[2]],
    }


def main():
    parser = argparse.ArgumentParser(description="Exchange visual states and language keys and values.")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("split", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--selected", type=Path)
    args = parser.parse_args()
    configure_inference()
    model, processor, _ = load_model(args.model, args.base)
    kimi = args.model == "kimi"
    make_region = kimi_region if kimi else qwen_region
    attention = nullcontext() if kimi else torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    rows = json.loads(args.split.read_bytes())
    if args.selected:
        selected = set(json.loads(args.selected.read_bytes()))
        rows = [row for row in rows if f"{row['image_id']}_{row['region_index']}" in selected]
    with args.output.open("w", encoding="utf-8") as output, attention:
        for row in rows:
            image = prepared_image(args.image_root / row["image"], canvas_size(*row["processed_size"]) if kimi else row["processed_size"])
            region = row["region_index"]
            if kimi:
                imer = masked(image, row["processed_size"], region, row["internal_boxes"], True)
                imem = masked(imer, row["processed_size"], region, row["exterior_boxes"], False)
            else:
                imer = mask_boxes(image, row["internal_boxes"])
                imem = mask_boxes(imer, row["exterior_boxes"])
            question = row.get("question", PRESENCE)
            paired = [make_region(model, processor, arm, region, question) for arm in (imer, imem)]
            key = f"{row['image_id']}_{region}"
            output.write(json.dumps({"region": key, **transfer(model, processor, paired, args.model, key)}) + "\n")


if __name__ == "__main__":
    main()
