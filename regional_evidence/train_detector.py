import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from .data.detection import DetectionSample, parse_annotation
from .data.detection_dataset import VisDroneDetectionDataset
from .models.detection import make_retinanet
from .models.expert import DomainExpert, load_coco_backbone


def samples(split):
    for image in sorted((split / "images").glob("*.jpg")):
        annotation = split / "annotations" / f"{image.stem}.txt"
        with Image.open(image) as source:
            width, height = source.size
        boxes, ignored = parse_annotation(annotation)
        yield DetectionSample(image, annotation, image.stem, width, height, boxes, ignored)


def train(config, split, coco_checkpoint, output):
    torch.manual_seed(config["seed"])
    dataset = VisDroneDetectionDataset(list(samples(split)), min_pixels=config["min_pixels"], max_pixels=config["max_pixels"], training=True, seed=config["seed"])
    expert = DomainExpert()
    load_coco_backbone(expert.encoder, coco_checkpoint)
    model = make_retinanet(expert).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    accumulation = config["gradient_accumulation"]
    warmup = config["warmup_epochs"]
    epochs = config["epochs"]
    for epoch in range(1, epochs + 1):
        scale = epoch / warmup if epoch <= warmup else 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * scale
        dataset.set_epoch(epoch - 1)
        for index in range(len(dataset)):
            image, target = dataset[index]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model([image.cuda()], [{key: value.cuda() for key, value in target.items()}])
            (sum(losses.values()) / accumulation).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(dataset):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    state = {}
    for prefix, module in (("backbone", expert.encoder), ("detector_projections", expert.projections), ("fpn", expert.fpn)):
        state.update({f"{prefix}.{key}": value.detach().cpu().contiguous() for key, value in module.state_dict().items()})
    save_file(state, output)


def main():
    parser = argparse.ArgumentParser(description="Train the VisDrone PVTv2-B0 domain expert.")
    parser.add_argument("split", type=Path)
    parser.add_argument("coco_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/domain_expert.json"))
    args = parser.parse_args()
    train(json.loads(args.config.read_bytes()), args.split, args.coco_checkpoint, args.output)


if __name__ == "__main__":
    main()
