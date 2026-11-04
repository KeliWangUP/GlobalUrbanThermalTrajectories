import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee

from src.export_utils import CONFIG_PATH, export_table_to_drive
from src.samplepoint_attributes import prepare_isa_source
from src.samplepoint_isa import (
    build_fullcell_isa_image,
    build_samplepoint_isa_selectors,
    sample_fullcell_isa
)
from src.samplepoint_utils import (
    get_samplepoint_city_ids,
    load_sample_points,
    partition_samplepoint_cities,
    samplepoint_base_fields
)


def export_samplepoint_isa(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> list[ee.batch.Task]:
    """Submit ISA-only full-cell exports for one city or selected batches."""
    settings = config["samplepoint_isa_export"]
    selection = settings["selection"]
    offset = settings["batch_offset"]
    maximum = settings["max_batches_per_run"]
    dataset_name = settings["dataset_name"]
    version = settings["version"]
    task_log = settings["task_log"]
    if selection not in ("test_city", "batches"):
        raise ValueError("samplepoint_isa_export.selection must be 'test_city' or 'batches'.")
    if type(offset) is not int or offset < 0:
        raise ValueError("samplepoint_isa_export.batch_offset must be non-negative.")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("samplepoint_isa_export.max_batches_per_run must be positive.")
    if selection == "test_city" and offset != 0:
        raise ValueError("batch_offset must be 0 for test_city selection.")
    if any(not isinstance(value, str) or not value.strip() for value in (dataset_name, version, task_log)):
        raise ValueError("ISA dataset_name, version and task_log must be non-empty strings.")
    log_path = Path(task_log)
    if not log_path.is_absolute():
        log_path = CONFIG_PATH.parent / log_path
    log_path = log_path.resolve()

    id_field = config["cities"]["id_field"]
    base_fields = samplepoint_base_fields(config)
    points, point_count = load_sample_points(config, base_fields)
    valid_points = points.filter(ee.Filter.notNull(base_fields)).size().getInfo()
    if valid_points != point_count:
        raise ValueError(
            f"{point_count - valid_points} sample points have missing required "
            f"properties: {base_fields}."
        )
    city_ids = get_samplepoint_city_ids(points, id_field)
    batches = partition_samplepoint_cities(
        city_ids, config["sample_points"]["city_batch_size"]
    )
    queue_hash = hashlib.sha256(
        json.dumps(city_ids, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if selection == "test_city":
        city_id = config["test"]["city_id"]
        if city_id not in city_ids:
            raise ValueError(f"Test city {city_id} is not in the sample-point asset.")
        specs = [(None, [city_id])]
    else:
        if offset >= len(batches):
            raise ValueError(f"batch_offset {offset} is outside the {len(batches)} city batches.")
        specs = [
            (index + 1, batches[index])
            for index in range(offset, min(offset + maximum, len(batches)))
        ]

    isa_image, isa_band, years = prepare_isa_source(config)
    fraction_image = build_fullcell_isa_image(
        isa_image, years, config["samplepoint_attributes"]["max_pixels_per_cell"]
    )
    selectors = build_samplepoint_isa_selectors(config, years)
    folder = config["export"]["folder"]
    print(
        f"[INFO] Sample-point asset: {config['export']['asset_root']}; "
        f"{point_count} points, {len(city_ids)} cities, {len(batches)} batches; "
        f"city queue SHA-256={queue_hash}."
    )
    print(f"[INFO] ISA asset: {config['samplepoint_attributes']['isa_asset']}; band={isa_band}.")

    tasks = []
    for batch_index, batch_city_ids in specs:
        selected_points = points.filter(ee.Filter.inList(id_field, batch_city_ids))
        if batch_index is None:
            point_total = selected_points.size().getInfo()
            if not point_total:
                raise ValueError(f"No sample points for test city {batch_city_ids[0]}.")
            suffix = f"city_{batch_city_ids[0]}_test"
            print(f"[INFO] Test city {batch_city_ids[0]}: {point_total} points.")
        else:
            suffix = f"city_batch_{batch_index:04d}_of_{len(batches):04d}"
            print(
                f"[INFO] Batch {batch_index}/{len(batches)}: "
                f"{len(batch_city_ids)} cities; IDs={batch_city_ids}."
            )
        export_name = f"{dataset_name}_{version}_{suffix}"
        collection = sample_fullcell_isa(
            fraction_image, selected_points, config, projection, scale
        )
        task = export_table_to_drive(
            collection=collection,
            description=export_name,
            folder=folder,
            file_name=export_name,
            selectors=selectors
        )
        tasks.append(task)
        record = {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "task_id": task.id,
            "description": export_name,
            "selection": selection,
            "batch_index": batch_index,
            "batch_count": len(batches),
            "batch_size": config["sample_points"]["city_batch_size"],
            "city_ids": batch_city_ids,
            "city_queue_sha256": queue_hash,
            "point_asset": config["export"]["asset_root"],
            "isa_asset": config["samplepoint_attributes"]["isa_asset"],
            "isa_band": isa_band,
            "isa_years": list(years),
            "denominator": "full_modis_cell_masked_as_zero"
        }
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(f"[ISA_TASK] batch={batch_index} task_id={task.id} description={export_name}")

    print(f"[INFO] Submitted {len(tasks)} ISA export tasks.")
    if selection == "batches" and offset + len(specs) < len(batches):
        print(f"[INFO] Next batch_offset: {offset + len(specs)}.")
    return tasks
