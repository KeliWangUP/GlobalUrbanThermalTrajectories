import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee

from src.export_utils import CONFIG_PATH, export_table_to_drive
from src.samplepoint_attributes import prepare_samplepoint_sources
from src.samplepoint_development import (
    build_fullcell_development_selectors,
    prepare_fullcell_development_images,
    sample_fullcell_development
)
from src.samplepoint_priority import filter_priority_sample_points, load_priority_city_ids_json
from src.samplepoint_utils import load_sample_points, samplepoint_base_fields


def export_samplepoint_development(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> list[ee.batch.Task]:
    """Submit one full-cell ISA/water and GHSL export per priority city."""
    settings = config["samplepoint_development_export"]
    selection = settings["selection"]
    offset = settings["city_offset"]
    maximum = settings["max_cities_per_run"]
    dataset_name = settings["dataset_name"]
    version = settings["version"]
    task_log = settings["task_log"]
    if selection not in ("test_city", "all"):
        raise ValueError("samplepoint_development_export.selection must be 'test_city' or 'all'.")
    if type(offset) is not int or offset < 0:
        raise ValueError("samplepoint_development_export.city_offset must be non-negative.")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("samplepoint_development_export.max_cities_per_run must be positive.")
    if selection == "test_city" and offset != 0:
        raise ValueError("city_offset must be 0 for test_city selection.")
    if any(not isinstance(value, str) or not value.strip() for value in (dataset_name, version, task_log)):
        raise ValueError("Development dataset_name, version and task_log must be non-empty strings.")
    log_path = Path(task_log)
    if not log_path.is_absolute():
        log_path = CONFIG_PATH.parent / log_path
    log_path = log_path.resolve()

    city_ids, priority_hash, priority_path = load_priority_city_ids_json(config)
    id_field = config["cities"]["id_field"]
    fields = samplepoint_base_fields(config)
    points, all_count = load_sample_points(config, fields)
    priority_points, priority_count = filter_priority_sample_points(
        points, id_field, city_ids
    )
    valid_count = priority_points.filter(ee.Filter.notNull(fields)).size().getInfo()
    if valid_count != priority_count:
        raise ValueError(
            f"{priority_count - valid_count} priority sample points have missing "
            f"required properties: {fields}."
        )
    queue_hash = hashlib.sha256(
        json.dumps(city_ids, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()

    if selection == "test_city":
        city_id = config["test"]["city_id"]
        if city_id not in city_ids:
            raise ValueError(f"Test city {city_id} is not in the priority-city list.")
        start = city_ids.index(city_id)
        selected_ids = [city_id]
    else:
        if offset >= len(city_ids):
            raise ValueError(f"city_offset {offset} is outside the {len(city_ids)} priority cities.")
        start = offset
        selected_ids = city_ids[offset:offset + maximum]

    sources = prepare_samplepoint_sources(config)
    images = prepare_fullcell_development_images(config, sources)
    selectors = build_fullcell_development_selectors(config)
    print(
        f"[INFO] Priority scope: {len(city_ids)} cities; {priority_count} of "
        f"{all_count} points; city list SHA-256={priority_hash}; source={priority_path}."
    )
    print(f"[INFO] City queue SHA-256={queue_hash}; selected IDs={selected_ids}.")
    print(f"[INFO] ISA asset={config['samplepoint_attributes']['isa_asset']}; band={sources.isa_band}.")

    tasks = []
    for queue_index, city_id in enumerate(selected_ids, start=start):
        city_points = points.filter(ee.Filter.eq(id_field, city_id))
        city_count = city_points.size().getInfo()
        if not city_count:
            raise ValueError(f"No sample points for priority city {city_id}.")
        print(f"[INFO] City {city_id}; offset={queue_index}; points={city_count}.")
        output = sample_fullcell_development(
            city_points, config, projection, scale, images
        )
        export_name = (
            f"{dataset_name}_{version}_priority_{priority_hash[:12]}_city_{city_id}"
        )
        task = export_table_to_drive(
            collection=output,
            description=export_name,
            folder=config["export"]["folder"],
            file_name=export_name,
            selectors=selectors
        )
        tasks.append(task)
        record = {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "task_id": task.id,
            "description": export_name,
            "priority_city_ids_json": str(priority_path),
            "priority_city_ids_sha256": priority_hash,
            "city_queue_sha256": queue_hash,
            "city_offset": queue_index,
            "city_id": city_id,
            "point_count": city_count,
            "point_asset": config["export"]["asset_root"],
            "isa_asset": config["samplepoint_attributes"]["isa_asset"],
            "isa_band": sources.isa_band,
            "isa_years": list(sources.isa_years),
            "ghsl_years": config["ghsl_built"]["years"],
            "isa_denominator": "full_modis_cell_masked_as_zero",
            "water_denominator": "full_modis_cell_masked_as_zero",
            "ghsl_statistic": "source_pixel_mean_and_count"
        }
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(f"[DEVELOPMENT_TASK] city_id={city_id} task_id={task.id} description={export_name}")

    print(f"[INFO] Submitted {len(tasks)} priority-city development tasks.")
    if selection == "all" and start + len(selected_ids) < len(city_ids):
        print(f"[INFO] Next city_offset: {start + len(selected_ids)}.")
    return tasks
