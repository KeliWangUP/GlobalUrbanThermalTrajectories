import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee

from export_samplepoint_attributes import export_samplepoint_attributes
from export_samplepoint_attributes_failed_batch import (
    FAILED_BATCH_FILE,
    read_failed_batches
)
from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)


TASK_LOG_FILE = Path(__file__).with_name(
    "failed_city_retry_task_log.jsonl"
)


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


def record_city_task(city_id: Any, task: ee.batch.Task) -> None:
    record = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "city_id": city_id,
        "task_id": task.id
    }
    with TASK_LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")


def main() -> None:
    config = load_config()
    initialize_gee(config)
    failed_batches = read_failed_batches(FAILED_BATCH_FILE)
    city_ids = get_failed_city_ids(config, failed_batches)

    retry_config = config["failed_samplepoint_retry"]
    city_offset = retry_config["city_offset"]
    max_cities_per_run = retry_config["max_cities_per_run"]
    if type(city_offset) is not int or city_offset < 0:
        raise ValueError(
            "failed_samplepoint_retry.city_offset must be a "
            "non-negative integer."
        )
    if type(max_cities_per_run) is not int or max_cities_per_run < 1:
        raise ValueError(
            "failed_samplepoint_retry.max_cities_per_run must be a "
            "positive integer."
        )
    if city_offset >= len(city_ids):
        raise ValueError(
            "failed_samplepoint_retry.city_offset must be less than the "
            f"{len(city_ids)} failed cities."
        )

    selected_city_ids = city_ids[
        city_offset:city_offset + max_cities_per_run
    ]
    print(
        f"[INFO] Failed batches: {len(failed_batches)}; "
        f"failed cities: {len(city_ids)}."
    )
    print(
        f"[INFO] Submitting city range "
        f"{city_offset + 1}-{city_offset + len(selected_city_ids)} "
        f"of {len(city_ids)}: {selected_city_ids}"
    )

    projection, scale = get_modis_reference_grid(config)
    tasks = export_samplepoint_attributes(
        config=config,
        projection=projection,
        scale=scale,
        expected_batch_count=failed_batches[0][1],
        selected_city_ids=selected_city_ids,
        on_city_task_submitted=record_city_task
    )
    if len(tasks) != len(selected_city_ids):
        raise RuntimeError(
            f"Expected {len(selected_city_ids)} city retry tasks, "
            f"got {len(tasks)}."
        )
    print(
        f"[INFO] Submitted {len(tasks)} city retry tasks. For the next "
        "wave, increase failed_samplepoint_retry.city_offset by "
        f"{len(selected_city_ids)}."
    )


if __name__ == "__main__":
    main()
