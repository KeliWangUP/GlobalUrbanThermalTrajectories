import json
import re
from pathlib import Path
from typing import Any

import ee


FAILED_BATCH_FILE = Path(__file__).resolve().parents[1] / "failed.json"
BATCH_PATTERN = re.compile(r"_city_batch_(\d+)_of_(\d+)$")


def read_failed_batches(path: Path) -> list[tuple[int, int, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Failed-batch file not found: {path}")
    with path.open("r", encoding="utf-8") as failed_file:
        failure: dict[str, Any] = json.load(failed_file)
    batch_names = failure.get("batches")
    if batch_names is None:
        batch_name = failure.get("batch")
        batch_names = [batch_name]
    if (
        not isinstance(batch_names, list)
        or not batch_names
        or any(not isinstance(name, str) for name in batch_names)
    ):
        raise ValueError(
            f"{path} must contain a non-empty 'batches' string list."
        )
    parsed_batches = []
    seen_indices = set()
    expected_counts = set()
    for batch_name in batch_names:
        match = BATCH_PATTERN.search(batch_name)
        if match is None:
            raise ValueError(
                f"Cannot parse a city batch number from '{batch_name}'."
            )
        batch_index = int(match.group(1))
        expected_count = int(match.group(2))
        if batch_index in seen_indices:
            raise ValueError(
                f"Duplicate failed city batch index: {batch_index}."
            )
        seen_indices.add(batch_index)
        expected_counts.add(expected_count)
        parsed_batches.append(
            (batch_index, expected_count, batch_name)
        )
    if len(expected_counts) != 1:
        raise ValueError(
            "Failed task names contain inconsistent total batch counts."
        )
    expected_count = expected_counts.pop()
    invalid_indices = [
        batch_index
        for batch_index, _, _ in parsed_batches
        if not 1 <= batch_index <= expected_count
    ]
    if invalid_indices:
        raise ValueError(
            f"Failed city batch indices are out of range: {invalid_indices}."
        )
    return parsed_batches


def read_failed_batch(path: Path) -> tuple[int, int, str]:
    parsed_batches = read_failed_batches(path)
    if len(parsed_batches) != 1:
        raise ValueError(
            f"{path} contains {len(parsed_batches)} failed batches; "
            "the single-batch retry and probe scripts require exactly one."
        )
    return parsed_batches[0]


def get_failed_city_ids(
    config: dict[str, Any],
    failed_batches: list[tuple[int, int, str]]
) -> list[Any]:
    point_asset = config["export"]["asset_root"]
    id_field = config["cities"]["id_field"]
    points = ee.FeatureCollection(point_asset)
    city_ids = points.aggregate_array(id_field).distinct().sort().getInfo()
    if not city_ids or any(city_id is None for city_id in city_ids):
        raise ValueError("Sample-point asset has no valid city IDs.")

    batch_size = config["sample_points"]["city_batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError(
            "sample_points.city_batch_size must be a positive integer."
        )
    batch_count = (len(city_ids) + batch_size - 1) // batch_size
    expected_counts = {total for _, total, _ in failed_batches}
    if expected_counts != {batch_count}:
        raise ValueError(
            "Failed task batch count does not match the current "
            f"sample-point asset/config: {expected_counts} vs {batch_count}. "
            "Refusing to map indices to a potentially different city list."
        )

    failed_city_ids = set()
    for batch_index, _, batch_name in failed_batches:
        start = (batch_index - 1) * batch_size
        batch_city_ids = city_ids[start:start + batch_size]
        if not batch_city_ids:
            raise ValueError(
                f"No city IDs found for failed batch '{batch_name}'."
            )
        failed_city_ids.update(batch_city_ids)

    if len(failed_city_ids) != sum(
        min(batch_size, len(city_ids) - (batch_index - 1) * batch_size)
        for batch_index, _, _ in failed_batches
    ):
        raise ValueError(
            "Failed batches map to overlapping city IDs; refusing to "
            "submit duplicate city exports."
        )
    return [city_id for city_id in city_ids if city_id in failed_city_ids]
