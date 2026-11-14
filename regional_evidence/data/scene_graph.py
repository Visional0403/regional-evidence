from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from PIL import Image

from .constants import CATEGORY_NAMES, REGION_NAMES, VEHICLE_CATEGORIES

LOGGER = logging.getLogger(__name__)
COLUMNS = ('left', 'center', 'right')

def _partition_index(coordinate: float, extent: int) -> int:
    return max(0, min(int(coordinate / max(extent, 1) * 3), 2))

def _grid_region(center_x: float, center_y: float, width: int, height: int) -> str:
    column = _partition_index(center_x, width)
    row = _partition_index(center_y, height)
    return REGION_NAMES[row * 3 + column]

def _column(center_x: float, width: int) -> str:
    return COLUMNS[_partition_index(center_x, width)]

def _read_annotations(annotation_path: Path) -> list[dict]:
    objects: list[dict] = []
    with annotation_path.open('r', encoding='utf-8-sig') as file:
        for line_number, raw_line in enumerate(file, start=1):
            fields = raw_line.strip().rstrip(',').split(',')
            if fields == ['']:
                continue
            if len(fields) < 8:
                raise ValueError(f'Malformed VisDrone row at {annotation_path}:{line_number}')
            try:
                x, y, box_width, box_height, score, category, truncation, occlusion = (int(value) for value in fields[:8])
            except ValueError as error:
                raise ValueError(f'Non-integer VisDrone row at {annotation_path}:{line_number}') from error
            if score == 0 or category == 0:
                continue
            if box_width <= 0 or box_height <= 0:
                LOGGER.warning('Skipping non-positive bbox at %s:%d', annotation_path, line_number)
                continue
            if category not in CATEGORY_NAMES:
                raise ValueError(f'Unknown VisDrone category {category} at {annotation_path}:{line_number}')
            objects.append({'x': x, 'y': y, 'width': box_width, 'height': box_height, 'category_id': category, 'truncation': truncation, 'occlusion': occlusion})
    return objects

def _all_maxima(counts: dict[str, int], order: tuple[str, ...]) -> list[str]:
    maximum = max(counts.values(), default=0)
    if maximum == 0:
        return []
    return [name for name in order if counts[name] == maximum]

def build_scene(image_path: Path, annotation_path: Path, image_reference: str) -> dict:
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f'Invalid image dimensions: {image_path}')
    annotations = _read_annotations(annotation_path)
    raw_counts = Counter(CATEGORY_NAMES[item['category_id']] for item in annotations)
    category_counts = {CATEGORY_NAMES[category_id]: raw_counts.get(CATEGORY_NAMES[category_id], 0) for category_id in range(1, 12)}
    object_distribution = {name: 0 for name in COLUMNS}
    vehicle_distribution = {name: 0 for name in COLUMNS}
    grid = {name: {'object_count': 0, 'vehicle_count': 0} for name in REGION_NAMES}
    for item in annotations:
        center_x = item['x'] + item['width'] / 2
        center_y = item['y'] + item['height'] / 2
        column = _column(center_x, image_width)
        region = _grid_region(center_x, center_y, image_width, image_height)
        object_distribution[column] += 1
        grid[region]['object_count'] += 1
        if item['category_id'] in VEHICLE_CATEGORIES:
            vehicle_distribution[column] += 1
            grid[region]['vehicle_count'] += 1
    vehicle_count = sum(category_counts[CATEGORY_NAMES[index]] for index in VEHICLE_CATEGORIES)
    pedestrian_related = category_counts['pedestrian'] + category_counts['people']
    object_grid_counts = {name: grid[name]['object_count'] for name in REGION_NAMES}
    vehicle_grid_counts = {name: grid[name]['vehicle_count'] for name in REGION_NAMES}
    return {'image': image_reference, 'annotation': annotation_path.name, 'image_size': {'width': image_width, 'height': image_height}, 'category_counts': category_counts, 'totals': {'valid_objects': len(annotations), 'vehicles': vehicle_count, 'pedestrian_related': pedestrian_related}, 'spatial': {'object_distribution': object_distribution, 'vehicle_distribution': vehicle_distribution, 'object_grid': object_grid_counts, 'vehicle_grid': vehicle_grid_counts, 'crowded_regions': _all_maxima(object_grid_counts, REGION_NAMES), 'largest_vehicle_cluster_regions': _all_maxima(vehicle_grid_counts, REGION_NAMES), 'busiest_vehicle_columns': _all_maxima(vehicle_distribution, COLUMNS)}}
