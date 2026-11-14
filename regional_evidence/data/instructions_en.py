"""English five-task questions and answers derived from annotation facts."""

from __future__ import annotations

import json
import random

from .constants import REGION_NAMES

TASK_ORDER = (
    "object_statistics",
    "vehicle_composition",
    "pedestrian_statistics",
    "spatial_distribution",
    "spatial_comparison",
)
OBJECT_KEYS = (
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning_tricycle", "bus", "motor", "others",
)
VEHICLE_KEYS = (
    "bicycle", "car", "van", "truck", "tricycle", "awning_tricycle", "bus", "motor",
)
VEHICLE_NAMES = (
    "bicycle", "car", "van", "truck", "tricycle", "awning tricycle",
    "bus", "motorcycle",
)
VEHICLE_SCOPE = (
    "Vehicles comprise bicycles, cars, vans, trucks, tricycles, awning tricycles, "
    "buses, and motorcycles; exclude the pedestrian, people, and others categories."
)
COUNT_SCOPE = (
    "Count valid annotated targets only; ignore regions marked with score=0 or category=0. "
    "Report exact counts without inferring hidden or unannotated targets."
)
SPATIAL_RULE = (
    "Divide image width and height into three equal parts and assign each vehicle "
    "by its bounding-box center. A center on an internal boundary belongs to the "
    "cell to its right or below. Left, center, and right refer to image columns, "
    "not travel directions. Use these grid-cell names in row-major order: "
    + ", ".join(REGION_NAMES) + "."
)
FACT_RULE = (
    "Report counts and image-coordinate locations only; do not infer congestion, "
    "traffic pressure, speed, road structure, or future events."
)
JSON_RULE = (
    "Keep every key, including zero-count categories, and replace placeholder zeros "
    "with non-negative integers. Return only the JSON object, without Markdown "
    "fences or surrounding text."
)
OBJECT_SCHEMA = json.dumps(
    {"object_count": dict.fromkeys((*OBJECT_KEYS, "total_valid_objects"), 0)},
    separators=(",", ":"),
)
SPATIAL_SCHEMA = json.dumps(
    {
        "vehicle_distribution": dict.fromkeys(("left", "center", "right"), 0),
        "vehicle_grid": dict.fromkeys(REGION_NAMES, 0),
        "largest_vehicle_cluster_regions": [],
        "busiest_vehicle_columns": [],
    },
    separators=(",", ":"),
)


def object_statistics(scene: dict, rng: random.Random) -> tuple[str, str, dict]:
    opening = rng.choice((
        "Count the objects in every category in this image.",
        "Report the number of objects of each category in this image.",
        "Provide the complete object-category counts for this image.",
    ))
    question = f"{opening} {COUNT_SCOPE} Use this exact schema: {OBJECT_SCHEMA} {JSON_RULE} {FACT_RULE}"
    counts = {key: scene["category_counts"][key] for key in OBJECT_KEYS}
    counts["total_valid_objects"] = scene["totals"]["valid_objects"]
    facts = {"object_count": counts}
    return question, json.dumps(facts, separators=(",", ":")), facts


def vehicle_composition(scene: dict, rng: random.Random) -> tuple[str, str, dict]:
    opening = rng.choice((
        "Report the total number of vehicles and the count of each vehicle category.",
        "Describe the vehicle composition of this image with exact counts.",
        "How many vehicles are there in total and in each vehicle category?",
    ))
    question = (
        f"{opening} {COUNT_SCOPE} {VEHICLE_SCOPE} Include all eight categories, "
        f"even when their count is zero. Answer in one or two English sentences. {FACT_RULE}"
    )
    facts = {"total": scene["totals"]["vehicles"]}
    facts.update({key: scene["category_counts"][key] for key in VEHICLE_KEYS})
    details = ", ".join(
        f"{facts[key]} {name if facts[key] == 1 else ('buses' if name == 'bus' else name + 's')}"
        for key, name in zip(VEHICLE_KEYS, VEHICLE_NAMES)
    )
    answer = rng.choice((
        f"The total vehicle count is {facts['total']}: {details}.",
        f"The total vehicle count is {facts['total']}, comprising {details}.",
    ))
    return question, answer, facts


def pedestrian_statistics(scene: dict, rng: random.Random) -> tuple[str, str, dict]:
    opening = rng.choice((
        "Count the pedestrian-related targets in this image.",
        "Report the counts of the pedestrian and people categories and their sum.",
        "How many targets belong to the pedestrian and people categories?",
    ))
    question = (
        f"{opening} {COUNT_SCOPE} Pedestrian-related targets are the sum of the "
        "pedestrian and people categories. Report both category counts and their "
        f"combined total in one English sentence. {FACT_RULE}"
    )
    facts = {
        "pedestrian": scene["category_counts"]["pedestrian"],
        "people": scene["category_counts"]["people"],
        "total": scene["totals"]["pedestrian_related"],
    }
    answer = rng.choice((
        (
            f"The counts are {facts['pedestrian']} for the pedestrian category and "
            f"{facts['people']} for the people category, with a combined total of {facts['total']}."
        ),
        (
            f"The pedestrian count is {facts['pedestrian']}, the people count is "
            f"{facts['people']}, and their combined total is {facts['total']}."
        ),
    ))
    return question, answer, facts


def spatial_distribution(scene: dict, rng: random.Random) -> tuple[str, str, dict]:
    opening = rng.choice((
        "Report the vehicle counts in the three image columns and nine grid cells.",
        "Provide the complete image-coordinate distribution of vehicles.",
        "Count vehicles by image column and by cell of a 3-by-3 grid.",
    ))
    question = (
        f"{opening} {COUNT_SCOPE} {VEHICLE_SCOPE} {SPATIAL_RULE} "
        "Include all tied highest-count columns and grid cells in the corresponding "
        "arrays, in the given order. If there are no vehicles, both arrays must be empty. "
        f"Use this exact schema: {SPATIAL_SCHEMA} {JSON_RULE} {FACT_RULE}"
    )
    facts = {key: scene["spatial"][key] for key in (
        "vehicle_distribution", "vehicle_grid", "largest_vehicle_cluster_regions",
        "busiest_vehicle_columns",
    )}
    return question, json.dumps(facts, separators=(",", ":")), facts


def spatial_comparison(scene: dict, rng: random.Random) -> tuple[str, str, dict]:
    opening = rng.choice((
        "Compare vehicle counts across image columns and identify the highest-count grid cells.",
        "Describe the vehicle distribution across the left, center, and right columns.",
        "Which image columns and grid cells contain the most vehicles?",
    ))
    question = (
        f"{opening} {COUNT_SCOPE} {VEHICLE_SCOPE} {SPATIAL_RULE} "
        "Report all three column counts, every tied highest-count column, and every "
        "tied highest-count grid cell, with their counts. If there are no vehicles, "
        f"state this explicitly. Answer in two or three English sentences. {FACT_RULE}"
    )
    spatial = scene["spatial"]
    distribution = spatial["vehicle_distribution"]
    facts = {
        "vehicle_distribution": distribution,
        "busiest_vehicle_columns": spatial["busiest_vehicle_columns"],
        "largest_vehicle_cluster_regions": spatial["largest_vehicle_cluster_regions"],
        "maximum_column_count": max(distribution.values()),
        "maximum_grid_count": max(spatial["vehicle_grid"].values()),
    }
    first = (
        f"The left, center, and right columns contain {distribution['left']}, "
        f"{distribution['center']}, and {distribution['right']} vehicles, respectively."
    )
    if scene["totals"]["vehicles"] == 0:
        answer = (
            f"{first} All nine grid cells also contain zero vehicles, so there is "
            "no highest-count vehicle column or grid cell."
        )
    else:
        columns = ", ".join(facts["busiest_vehicle_columns"])
        cells = ", ".join(facts["largest_vehicle_cluster_regions"])
        column_label = "column is" if len(facts["busiest_vehicle_columns"]) == 1 else "columns are"
        cell_label = "grid cell is" if len(facts["largest_vehicle_cluster_regions"]) == 1 else "grid cells are"
        column_join = "with" if len(facts["busiest_vehicle_columns"]) == 1 else "each with"
        cell_join = "with" if len(facts["largest_vehicle_cluster_regions"]) == 1 else "each with"
        column_unit = "vehicle" if facts["maximum_column_count"] == 1 else "vehicles"
        cell_unit = "vehicle" if facts["maximum_grid_count"] == 1 else "vehicles"
        answer = (
            f"{first} The highest-count {column_label} {columns}, {column_join} "
            f"{facts['maximum_column_count']} {column_unit}. The highest-count {cell_label} "
            f"{cells}, {cell_join} {facts['maximum_grid_count']} {cell_unit}."
        )
    return question, answer, facts


GENERATORS = (
    object_statistics, vehicle_composition, pedestrian_statistics,
    spatial_distribution, spatial_comparison,
)
