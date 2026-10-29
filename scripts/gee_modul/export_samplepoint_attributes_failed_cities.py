import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee

from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.samplepoint_attribute_exports import export_samplepoint_attributes
from src.samplepoint_recovery import (
    FAILED_BATCH_FILE,
    get_failed_city_ids,
    read_failed_batches
)


TASK_LOG_FILE = Path(__file__).with_name(
    "failed_city_retry_task_log.jsonl"
)


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
