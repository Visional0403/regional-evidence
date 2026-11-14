import random
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .detection import DetectionBox, DetectionSample, scale_boxes, smart_resize


class VisDroneDetectionDataset(Dataset):

    def __init__(self, samples: Sequence[DetectionSample], *, min_pixels: int, max_pixels: int, training: bool, seed: int=42) -> None:
        self.samples = tuple(samples)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _box_tensor(boxes: Sequence[DetectionBox]) -> torch.Tensor:
        return torch.tensor([box.xyxy for box in boxes], dtype=torch.float32).reshape(-1, 4)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        resized_h, resized_w = smart_resize(sample.height, sample.width, factor=28, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        image = Image.open(sample.image_path).convert('RGB').resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        boxes = list(scale_boxes(sample.boxes, (sample.height, sample.width), (resized_h, resized_w)))
        ignored = list(scale_boxes(sample.ignored_boxes, (sample.height, sample.width), (resized_h, resized_w)))
        rng = random.Random(self.seed << 32 ^ self.epoch << 16 ^ index)
        if self.training:
            brightness = 1.0 + rng.uniform(-0.1, 0.1)
            contrast = 1.0 + rng.uniform(-0.1, 0.1)
            saturation = 1.0 + rng.uniform(-0.1, 0.1)
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
            image = ImageEnhance.Color(image).enhance(saturation)
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes = [DetectionBox(resized_w - box.x - box.width, box.y, box.width, box.height, box.category, box.truncation, box.occlusion) for box in boxes]
                ignored = [DetectionBox(resized_w - box.x - box.width, box.y, box.width, box.height, box.category, box.truncation, box.occlusion) for box in ignored]
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        target = {'boxes': self._box_tensor(boxes), 'labels': torch.tensor([box.category for box in boxes], dtype=torch.int64), 'ignored_boxes': self._box_tensor(ignored), 'image_id': torch.tensor(index, dtype=torch.int64), 'area': torch.tensor([box.width * box.height for box in boxes], dtype=torch.float32), 'iscrowd': torch.zeros(len(boxes), dtype=torch.int64), 'occlusion': torch.tensor([box.occlusion for box in boxes], dtype=torch.int64), 'truncation': torch.tensor([box.truncation for box in boxes], dtype=torch.int64)}
        return (tensor, target)
