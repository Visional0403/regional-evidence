import hashlib
import itertools
import json
from contextlib import contextmanager

import numpy as np


def spatial_tiles(height, width):
    tiles = {i: [] for i in range(9)}
    for row in range(height):
        band = min(2, (2 * row + 1) * 3 // (2 * height))
        for column in range(width):
            strip = min(2, (2 * column + 1) * 3 // (2 * width))
            tiles[3 * band + strip].append(row * width + column)
    return {block: positions for block, positions in tiles.items() if positions}


def block_positions(tiles, blocks):
    return sorted(position for block in blocks for position in tiles[block])


def _geometry(blocks, tiles, width):
    dimensions = []
    for block in blocks:
        positions = tiles[block]
        rows = len({position // width for position in positions})
        columns = len({position % width for position in positions})
        dimensions.append((rows, columns))
    adjacency = sum(
        abs(first // 3 - second // 3) + abs(first % 3 - second % 3) == 1
        for first, second in itertools.combinations(blocks, 2)
    )
    return sorted(dimensions), len(block_positions(tiles, blocks)), adjacency


def matched_random_blocks(tiles, width, ranked, identity, namespace):
    target = _geometry(ranked, tiles, width)
    pool = [blocks for blocks in itertools.combinations(tiles, 3) if _geometry(blocks, tiles, width) == target]
    for repeat in range(3):
        encoded = json.dumps([namespace, identity, repeat], sort_keys=True, ensure_ascii=True).encode()
        seed = int(hashlib.sha256(encoded).hexdigest()[:16], 16)
        yield pool[int(np.random.default_rng(seed).integers(len(pool)))]


def random_tokens(identity, total, count, size, repeat, namespace):
    key = json.dumps([namespace, identity, size, repeat], sort_keys=True, ensure_ascii=True).encode()
    seed = int(hashlib.sha256(key).hexdigest()[:16], 16)
    return np.sort(np.random.default_rng(seed).choice(total, count, replace=False)).tolist()


def exchange_tokens(base, donor, positions):
    patched = base.clone()
    patched[positions] = donor[positions]
    return patched


def donor_effect(imer, imem, imer_patched, imem_patched):
    return ((imer - imer_patched) + (imem_patched - imem)) / 2


def joint_interaction(imer_scores, imem_scores):
    return tuple(both - key - value + baseline for baseline, key, value, both in (imer_scores, imem_scores))


@contextmanager
def image_kv(positions, donors=None, selected=None, sources=(0, 1)):
    """Capture or exchange post-position-encoding image K/V during prefill."""
    from transformers.cache_utils import DynamicCache

    captured = {}
    original = DynamicCache.update
    slots = [positions[index] for index in selected] if donors is not None else None

    def update(cache, key, value, layer_idx, *args, **kwargs):
        if cache.get_seq_length(layer_idx) == 0:
            if donors is None:
                captured[layer_idx] = (key[..., positions, :].detach().clone(), value[..., positions, :].detach().clone())
            else:
                key, value = key.clone(), value.clone()
                key[..., slots, :] = donors[sources[0]][layer_idx][0][..., selected, :]
                value[..., slots, :] = donors[sources[1]][layer_idx][1][..., selected, :]
        return original(cache, key, value, layer_idx, *args, **kwargs)

    DynamicCache.update = update
    try:
        yield captured
    finally:
        DynamicCache.update = original
