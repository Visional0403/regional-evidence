import argparse
import json
import re
from collections import Counter
from functools import cache
from pathlib import Path
from statistics import mean

NUMBER = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?")
PROTOCOLS = {"en-paired": 2, "zh-paired": 2, "en-three-stage": 3, "zh-three-stage": 3}
COUNT_FIELDS = {
    "object_statistics": ("object_count",),
    "spatial_distribution": ("vehicle_distribution", "vehicle_grid"),
}
SET_METRICS = {
    "largest_vehicle_cluster_regions": "maximum_grid_set_accuracy",
    "busiest_vehicle_columns": "maximum_column_set_accuracy",
}
EN_ALIASES = (
    ("pedestrian", "pedestrians"), ("people",), ("bicycle", "bicycles"), ("car", "cars"),
    ("van", "vans"), ("truck", "trucks"), ("tricycle", "tricycles"),
    ("awning_tricycle", "awning_tricycles", "awning tricycle", "awning tricycles"),
    ("bus", "buses"), ("motor", "motors", "motorcycle", "motorcycles"), ("other", "others"),
    ("left",), ("center",), ("right",), ("center-grid", "center grid cell", "center cell"),
    *((cell,) for cell in ("top-left", "top-center", "top-right", "center-left", "center-right",
                           "bottom-left", "bottom-center", "bottom-right")),
)
ZH_ALIASES = (
    ("pedestrian", "行人"), ("people", "人员"), ("bicycle", "自行车", "单车"),
    ("car", "小汽车", "轿车"), ("van", "厢式车", "面包车"), ("truck", "卡车", "货车"),
    ("tricycle", "三轮车"), ("awning_tricycle", "awning-tricycle", "带棚三轮车"),
    ("bus", "公交车", "巴士"), ("motor", "摩托车"), ("others", "other", "其他目标"),
    ("left", "左侧", "左区域", "画面左边", "画面左侧"),
    ("center", "中部", "中区域", "中央列", "画面中间"),
    ("right", "右侧", "右区域", "画面右边", "画面右侧"),
    ("top-left", "左上区域", "左上方", "画面左上"),
    ("top-center", "上方中央区域", "上中区域", "顶部中央"),
    ("top-right", "右上区域", "右上方", "画面右上"),
    ("center-left", "左侧中部区域", "中左区域", "画面中左"),
    ("画面中央区域", "中心区域", "九宫格中央"),
    ("center-right", "右侧中部区域", "中右区域", "画面中右"),
    ("bottom-left", "左下区域", "左下方", "画面左下"),
    ("bottom-center", "下方中央区域", "下中区域", "底部中央"),
    ("bottom-right", "右下区域", "右下方", "画面右下"),
)


def complete_json(text, reference):
    start, end = text.find("{"), text.rfind("}")
    stripped = text.strip()
    candidates = [stripped, re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)]
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    else:
        return None

    def valid(x, expected):
        if isinstance(expected, dict):
            return isinstance(x, dict) and x.keys() == expected.keys() and all(valid(x[k], v) for k, v in expected.items())
        if type(expected) is int:
            return type(x) in (int, float)
        return isinstance(x, list) and all(isinstance(item, str) for item in x)

    return value if valid(value, reference) else None


def numeric_pairs(value, reference, fields):
    return [(value[field][key], truth) for field in fields for key, truth in reference[field].items()]


def numbers(text):
    return {round(float(value), 6) for value in NUMBER.findall(text)}


@cache
def word_pattern(alias):
    return re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(alias)}(?![A-Za-z0-9_-])", re.IGNORECASE)


def contains_alias(text, alias, language):
    if not alias.isascii():
        return alias in text
    if language == "en" and alias in ("tricycle", "tricycles"):
        text = re.sub(r"awning[ _-]tricycles?", "", text, flags=re.IGNORECASE)
    if language == "en" and alias == "center-grid":
        parts = re.findall(r"highest-count grid cells? (?:is|are) ([^.]+)", text, flags=re.IGNORECASE)
        return any(contains_alias(part, "center", "en") for part in parts)
    return word_pattern(alias).search(text) is not None


def concepts(text, language):
    aliases = EN_ALIASES if language == "en" else ZH_ALIASES
    return {group[0] for group in aliases if any(contains_alias(text, alias, language) for alias in group)}


def text_scores(reference, reply, language):
    expected, actual = (re.sub(r"\s+", "", value) for value in (reference, reply))
    overlap = sum((Counter(expected) & Counter(actual)).values())
    total = len(expected) + len(actual)
    expected_numbers, expected_concepts = numbers(reference), concepts(reference, language)
    values = {"exact_text_match": expected == actual, "character_f1": 2 * overlap / total if total else 1.0}
    if expected_numbers:
        values["number_recall"] = len(expected_numbers & numbers(reply)) / len(expected_numbers)
    if expected_concepts:
        values["concept_recall"] = len(expected_concepts & concepts(reply, language)) / len(expected_concepts)
    return values


def task_scores(records, outputs, protocol):
    if len(outputs) != PROTOCOLS[protocol]:
        raise ValueError(f"{protocol} requires {PROTOCOLS[protocol]} prediction files.")
    language = protocol[:2]
    strict = protocol == "zh-three-stage"
    english_three_stage = protocol == "en-three-stage"
    scores = {}
    for task in dict.fromkeys(row["task"] for row in records):
        rows = [row for row in records if row["task"] == task]
        arm_scores = [{} for _ in outputs]
        complete = [0 for _ in outputs]
        truncated = [0 for _ in outputs]
        common = 0
        fields = COUNT_FIELDS.get(task)
        for row in rows:
            predictions = [arm[row["id"]] for arm in outputs]
            replies = [item["reply"] for item in predictions]
            stops = [item["truncated"] for item in predictions]
            truncated = [count + flag for count, flag in zip(truncated, stops, strict=True)]
            reference = row["answer"]
            if fields:
                expected = json.loads(reference)
                values = [complete_json(reply, expected) for reply in replies]
                for index, value in enumerate(values):
                    complete[index] += value is not None
                eligible = all(value is not None for value in values) and (not strict or not any(stops))
            else:
                eligible = not strict or (all(reply.strip() for reply in replies) and not any(stops))
            common += eligible
            if strict and not eligible:
                continue
            for index, arm in enumerate(arm_scores):
                if fields:
                    value = values[index]
                    metrics = {}
                    pairs = numeric_pairs(value, expected, fields) if value is not None else []
                    if eligible:
                        metrics["count_mae"] = mean(abs(actual - truth) for actual, truth in pairs)
                    if not english_three_stage:
                        field_count = sum(len(expected[field]) for field in fields)
                        metrics["count_field_accuracy"] = sum(actual == truth for actual, truth in pairs) / field_count
                        metrics["json_exact_match"] = value == expected
                        for field, metric in SET_METRICS.items():
                            if field in expected:
                                metrics[metric] = value is not None and set(value[field]) == set(expected[field])
                else:
                    metrics = text_scores(reference, replies[index], language)
                for metric, value in metrics.items():
                    arm.setdefault(metric, []).append(value)
        result = {
            "planned": len(rows),
            "paired": common,
            "truncated": truncated,
            "arms": [
                {metric: {"value": mean(values), "n": len(values)} for metric, values in arm.items()}
                for arm in arm_scores
            ],
        }
        if fields:
            result["complete"] = complete
        scores[task] = result
    return scores


def main():
    parser = argparse.ArgumentParser(description="Score the original-image five-task comparisons.")
    parser.add_argument("records", type=Path)
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_bytes())
    outputs = []
    for path in args.predictions:
        rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        outputs.append({row["id"]: row for row in rows})
    print(json.dumps(task_scores(records, outputs, args.protocol), indent=2))


if __name__ == "__main__":
    main()
