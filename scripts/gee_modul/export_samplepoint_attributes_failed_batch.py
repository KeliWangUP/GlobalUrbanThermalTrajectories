import json
import re
from pathlib import Path
from typing import Any

from export_samplepoint_attributes import export_samplepoint_attributes
from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)


FAILED_BATCH_FILE = Path(__file__).with_name("failed.json")
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


def main() -> None:
    config = load_config()
    initialize_gee(config)
    failed_batch_index, expected_batch_count, batch_name = (
        read_failed_batch(FAILED_BATCH_FILE)
    )
    configured_batch_size = config["sample_points"]["city_batch_size"]
    if type(configured_batch_size) is not int or configured_batch_size < 1:
        raise ValueError(
            "sample_points.city_batch_size must be a positive integer."
        )

    projection, scale = get_modis_reference_grid(config)
    tasks = export_samplepoint_attributes(
        config=config,
        projection=projection,
        scale=scale,
        selected_batch_index=failed_batch_index,
        expected_batch_count=expected_batch_count
    )
    if len(tasks) != 1:
        raise RuntimeError(
            f"Expected one retry task for '{batch_name}', got {len(tasks)}."
        )
    print(
        f"[INFO] Retried only batch {failed_batch_index:04d} of "
        f"{expected_batch_count:04d}. Verify that the current task description "
        "has the same total batch count before using its output."
    )


if __name__ == "__main__":
    main()
