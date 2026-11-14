from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

IGNORED_CLASSES = (0, 11)

@dataclass(frozen=True)
class DetectionBox:
    x: float
    y: float
    width: float
    height: float
    category: int
    truncation: int
    occlusion: int

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    annotation_path: Path
    sequence_id: str
    width: int
    height: int
    boxes: tuple[DetectionBox, ...]
    ignored_boxes: tuple[DetectionBox, ...]

def parse_annotation(path: str | Path) -> tuple[tuple[DetectionBox, ...], tuple[DetectionBox, ...]]:
    positives: list[DetectionBox] = []
    ignored: list[DetectionBox] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not raw.strip():
            continue
        fields = [item.strip() for item in raw.split(',')]
        if len(fields) < 8:
            raise ValueError(f'{path}:{line_number} has {len(fields)} fields; expected at least 8.')
        x, y, width, height = map(float, fields[:4])
        score, category, truncation, occlusion = map(int, fields[4:8])
        if width <= 0 or height <= 0:
            continue
        box = DetectionBox(x, y, width, height, category, truncation, occlusion)
        if score == 0 or category in IGNORED_CLASSES:
            ignored.append(box)
        elif 1 <= category <= 10:
            positives.append(box)
        else:
            raise ValueError(f'Unexpected VisDrone category {category} in {path}:{line_number}.')
    return (tuple(positives), tuple(ignored))

def smart_resize(height: int, width: int, *, factor: int=28, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if min(height, width) <= 0 or factor <= 0:
        raise ValueError('Image dimensions and factor must be positive.')
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError("Image aspect ratio exceeds Qwen's supported limit.")
    resized_h = max(factor, round(height / factor) * factor)
    resized_w = max(factor, round(width / factor) * factor)
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        resized_h = max(factor, math.floor(height / beta / factor) * factor)
        resized_w = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor
    return (resized_h, resized_w)

def scale_boxes(boxes: Sequence[DetectionBox], original_size: tuple[int, int], resized_size: tuple[int, int]) -> tuple[DetectionBox, ...]:
    original_h, original_w = original_size
    resized_h, resized_w = resized_size
    scale_x, scale_y = (resized_w / original_w, resized_h / original_h)
    return tuple(DetectionBox(box.x * scale_x, box.y * scale_y, box.width * scale_x, box.height * scale_y, box.category, box.truncation, box.occlusion) for box in boxes)
