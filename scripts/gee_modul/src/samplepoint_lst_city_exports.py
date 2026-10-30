import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee

from src.export_utils import (
    build_samplepoint_lst_selectors,
    build_study_period_samplepoint_lst_statistics,
    export_table_to_drive
)
from src.samplepoint_priority import filter_priority_sample_points, load_priority_city_ids
from src.samplepoint_utils import load_sample_points


TASK_LOG_FILE = Path(__file__).resolve().parents[1] / "samplepoint_lst_city_task_log.jsonl"


def export_samplepoint_lst_city(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> list[ee.batch.Task]:
    """Submit one full-study-period sample-point export per selected city."""
    settings = config["samplepoint_lst_city_export"]
    priority_only = settings["priority_only"]
    selection = settings["selection"]
    offset = settings["city_offset"]
    max_cities = settings["max_cities_per_run"]
    if type(priority_only) is not bool:
        raise ValueError("samplepoint_lst_city_export.priority_only must be a boolean.")
    if selection not in ("test_city", "all"):
        raise ValueError("City selection must be 'test_city' or 'all'.")
    if type(offset) is not int or offset < 0:
        raise ValueError("samplepoint_lst_city_export.city_offset must be non-negative.")
    if type(max_cities) is not int or max_cities < 1:
        raise ValueError("samplepoint_lst_city_export.max_cities_per_run must be positive.")
    if selection == "test_city" and offset != 0:
        raise ValueError("city_offset must be 0 when selection is 'test_city'.")

    study = config["study"]
    start_year, end_year = study["start_year"], study["end_year"]
    if type(start_year) is not int or type(end_year) is not int or start_year > end_year:
        raise ValueError("Study years must be integers with start_year <= end_year.")

    id_field = config["cities"]["id_field"]
    points, point_count = load_sample_points(config, ["pixel_id", id_field])
    priority_hash = None
    priority_csv = None
    if priority_only:
        city_ids, priority_hash, priority_csv = load_priority_city_ids(config, id_field)
        _, selected_point_count = filter_priority_sample_points(points, id_field, city_ids)
        print(
            f"[INFO] Priority scope: {len(city_ids)} cities; "
            f"{selected_point_count} of {point_count} points; "
            f"city list SHA-256={priority_hash}; source={priority_csv}."
        )
    else:
        valid_count = points.filter(ee.Filter.notNull([id_field])).size().getInfo()
        if valid_count != point_count:
            raise ValueError(
                f"{point_count - valid_count} sample points have missing city IDs."
            )
        city_ids = points.aggregate_array(id_field).distinct().sort().getInfo()

    if (
        not city_ids
        or any(type(city_id) not in (int, str) for city_id in city_ids)
        or any(not re.fullmatch(r"[A-Za-z0-9_-]+", str(city_id)) for city_id in city_ids)
        or len({str(city_id) for city_id in city_ids}) != len(city_ids)
    ):
        raise ValueError("City IDs must be unique, valid task-name integers or strings.")
    queue_json = json.dumps(city_ids, ensure_ascii=True, separators=(",", ":"))
    queue_hash = hashlib.sha256(queue_json.encode("ascii")).hexdigest()

    if selection == "test_city":
        test_city_id = config["test"]["city_id"]
        if test_city_id not in city_ids:
            raise ValueError(
                f"Test city {test_city_id} is not present in the selected city scope."
            )
        start = city_ids.index(test_city_id)
        selected_ids = [test_city_id]
    else:
        if offset >= len(city_ids):
            raise ValueError(
                f"city_offset {offset} is outside the {len(city_ids)} selected cities."
            )
        start = offset
        selected_ids = city_ids[offset:offset + max_cities]
    print(
        f"[INFO] City queue: {len(city_ids)} cities; "
        f"queue SHA-256={queue_hash}; selected IDs={selected_ids}."
    )

    export_config = config["export"]
    dataset_name = export_config["samplepoint_lst_dataset_name"]
    version = config["project"]["version"]
    selectors = build_samplepoint_lst_selectors(id_field)
    scope = f"priority_{priority_hash[:12]}" if priority_hash else "all"
    tasks = []
    for queue_index, city_id in enumerate(selected_ids, start=start):
        city_points = points.filter(ee.Filter.eq(id_field, city_id))
        city_point_count = city_points.size().getInfo()
        if city_point_count == 0:
            raise ValueError(f"Selected city {city_id} has no sample points.")
        expected_rows = city_point_count * (end_year - start_year + 1) * 12
        print(
            f"[INFO] City {city_id}; queue offset {queue_index}; "
            f"{city_point_count} points; years {start_year}-{end_year}; "
            f"expected rows {expected_rows}."
        )
        collection = build_study_period_samplepoint_lst_statistics(
            points=city_points,
            config=config,
            projection=projection,
            scale=scale,
            id_field=id_field
        )
        export_name = (
            f"{dataset_name}_{version}_{scope}_city_{city_id}_"
            f"{start_year}_{end_year}"
        )
        task = export_table_to_drive(
            collection=collection,
            description=export_name,
            folder=export_config["folder"],
            file_name=export_name,
            selectors=selectors
        )
        tasks.append(task)
        record = {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "task_id": task.id,
            "description": export_name,
            "point_asset": export_config["asset_root"],
            "priority_only": priority_only,
            "priority_city_csv": str(priority_csv) if priority_csv else None,
            "priority_city_ids_sha256": priority_hash,
            "city_queue_sha256": queue_hash,
            "city_offset": queue_index,
            "city_id": city_id,
            "point_count": city_point_count,
            "start_year": start_year,
            "end_year": end_year
        }
        with TASK_LOG_FILE.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
        print(f"[CITY_TASK] city_id={city_id} task_id={task.id} description={export_name}")

    print(f"[INFO] Submitted {len(tasks)} city sample-point LST tasks.")
    if selection == "all" and start + len(selected_ids) < len(city_ids):
        print(f"[INFO] Next city_offset: {start + len(selected_ids)}.")
    return tasks
