import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np


def summarize(results):
    def read(figure):
        return json.loads((results / f"figure{figure}.json").read_bytes())

    one, two, three, four, five = (read(index) for index in range(1, 6))
    rows = one["count_points"]
    count = {route: mean(row[f"{route}_absolute_error"] for row in rows) for route in ("crop", "full")}
    presence = {condition: yes / two["presence_regions"] for condition, yes in two["yes_counts"].items()}
    block = three["spatial_block_control"]
    spatial = {"ranked": block["ranked_count"] / block["regions"], "random": mean(block["random_repeat_counts"]) / block["regions"]}
    languages = {}
    for language in ("en", "zh"):
        run = five[language]
        languages[language] = {
            "regional_count_mae": run["counting"]["regional_tokens"]["mae"],
            "full_image_models": run["models"],
            "intervals": {key: np.quantile(value["draws"], (0.025, 0.975)).tolist() for key, value in run["bootstrap"].items()},
        }
    return {"encoding_count_mae": count, "affirmative_rates": presence, "three_block_success": spatial, "generalization": four["datasets"], "expert": languages}


def main():
    parser = argparse.ArgumentParser(description="Summarize the published figure data.")
    parser.add_argument("results", type=Path, nargs="?", default=Path("results"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), indent=2))


if __name__ == "__main__":
    main()
