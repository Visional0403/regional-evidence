import argparse
import json
import random
from pathlib import Path

from .instructions import REGISTRY
from .instructions import TASK_ORDER as CHINESE_TASKS
from .instructions_en import GENERATORS, TASK_ORDER
from .scene_graph import build_scene


def prepare(root, output, language, seed):
    rng = random.Random(seed)
    records = []
    if language == "en":
        generators = tuple(zip(TASK_ORDER, GENERATORS, strict=True))
    else:
        generators = tuple((name, REGISTRY[family][name].generate) for family, name in CHINESE_TASKS)
    for image in sorted((root / "images").glob("*.jpg")):
        reference = f"{root.name}/images/{image.name}"
        scene = build_scene(image, root / "annotations" / f"{image.stem}.txt", reference)
        for index, (task, generate) in enumerate(generators):
            question, answer, *facts = generate(scene, rng)
            records.append({"id": f"{image.stem}:{index:02d}", "image": reference, "task": task, "question": question, "answer": answer, **({"reference_facts": facts[0]} if facts else {})})
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build VisDrone questions from official instance annotations.")
    parser.add_argument("split", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    prepare(args.split, args.output, args.language, args.seed)


if __name__ == "__main__":
    main()
