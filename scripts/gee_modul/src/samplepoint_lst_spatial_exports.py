import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee

from src.export_utils import (
    build_samplepoint_lst_selectors,
    build_study_period_samplepoint_lst_statistics,
    export_table_to_drive
)
from src.samplepoint_spatial import (
    filter_tile,
    get_occupied_tiles,
    tile_for_test_city
)
from src.samplepoint_priority import filter_priority_sample_points, load_priority_city_ids
from src.samplepoint_utils import load_sample_points


TASK_LOG_FILE = Path(__file__).resolve().parents[1] / "samplepoint_lst_spatial_task_log.jsonl"


def _signed_tile_index(index: int) -> str:
    return f"m{abs(index)}" if index < 0 else f"p{index}"


def export_samplepoint_lst_spatial(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> list[ee.batch.Task]:
    """Submit one full-study-period export per selected occupied grid tile."""
    spatial = config["samplepoint_lst_spatial_export"]
    tile_pixels = spatial["tile_pixels"]
    offset = spatial["tile_offset"]
    max_tiles = spatial["max_tiles_per_run"]
    selection = spatial["selection"]
    city_scope = spatial["city_scope"]
    if type(tile_pixels) is not int or tile_pixels < 1:
        raise ValueError("samplepoint_lst_spatial_export.tile_pixels must be positive.")
    if type(offset) is not int or offset < 0:
        raise ValueError("samplepoint_lst_spatial_export.tile_offset must be non-negative.")
    if type(max_tiles) is not int or max_tiles < 1:
        raise ValueError("samplepoint_lst_spatial_export.max_tiles_per_run must be positive.")
    if selection not in ("test_city", "tile", "all"):
        raise ValueError("Spatial selection must be 'test_city', 'tile', or 'all'.")
    if city_scope not in ("priority", "all"):
        raise ValueError("Spatial city_scope must be 'priority' or 'all'.")
    if selection != "all" and offset != 0:
        raise ValueError("tile_offset must be 0 for test_city or tile selection.")

    study = config["study"]
    start_year, end_year = study["start_year"], study["end_year"]
    if type(start_year) is not int or type(end_year) is not int or start_year > end_year:
        raise ValueError("Study years must be integers with start_year <= end_year.")

    id_field = config["cities"]["id_field"]
    points, point_count = load_sample_points(
        config, ["pixel_id", id_field, "modis_x", "modis_y"]
    )
    priority_hash = None
    priority_csv = None
    if city_scope == "priority":
        priority_ids, priority_hash, priority_csv = load_priority_city_ids(
            config, id_field
        )
        total_point_count = point_count
        points, point_count = filter_priority_sample_points(
            points, id_field, priority_ids
        )
        print(
            f"[INFO] Priority scope: {len(priority_ids)} cities; "
            f"{point_count} of {total_point_count} points; "
            f"city list SHA-256={priority_hash}; source={priority_csv}."
        )
    if selection == "test_city":
        tiles = [tile_for_test_city(
            points, id_field, config["test"]["city_id"], tile_pixels
        )]
    elif selection == "tile":
        column = spatial["tile_column"]
        row = spatial["tile_row"]
        if type(column) is not int or type(row) is not int:
            raise ValueError("tile_column and tile_row must be integers for tile selection.")
        tiles = [(column, row)]
    else:
        tiles = get_occupied_tiles(points, point_count, tile_pixels)
        if offset >= len(tiles):
            raise ValueError(
                f"tile_offset {offset} is outside the {len(tiles)} occupied tiles."
            )
        print(f"[INFO] Occupied tiles: {len(tiles)}; starting offset: {offset}.")
        tiles = tiles[offset:offset + max_tiles]

    export_config = config["export"]
    dataset_name = export_config["samplepoint_lst_dataset_name"]
    version = config["project"]["version"]
    selectors = build_samplepoint_lst_selectors(id_field)
    tasks = []
    for column, row in tiles:
        tile_points = filter_tile(points, column, row, tile_pixels)
        tile_point_count = tile_points.size().getInfo()
        if tile_point_count == 0:
            raise ValueError(f"Selected tile ({column}, {row}) has no sample points.")
        print(
            f"[INFO] Tile ({column}, {row}); {tile_point_count} points; "
            f"years {start_year}-{end_year}; "
            f"expected rows {tile_point_count * (end_year - start_year + 1) * 12}."
        )
        collection = build_study_period_samplepoint_lst_statistics(
            points=tile_points,
            config=config,
            projection=projection,
            scale=scale,
            id_field=id_field
        )
        export_name = (
            f"{dataset_name}_{version}_tile_{tile_pixels}_"
            f"x{_signed_tile_index(column)}_y{_signed_tile_index(row)}_"
            f"{start_year}_{end_year}"
        )
        if priority_hash is not None:
            export_name = (
                f"{dataset_name}_{version}_priority_{priority_hash[:12]}_"
                f"tile_{tile_pixels}_x{_signed_tile_index(column)}_"
                f"y{_signed_tile_index(row)}_{start_year}_{end_year}"
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
            "city_scope": city_scope,
            "priority_city_csv": str(priority_csv) if priority_csv else None,
            "priority_city_ids_sha256": priority_hash,
            "tile_pixels": tile_pixels,
            "tile_column": column,
            "tile_row": row,
            "point_count": tile_point_count,
            "start_year": start_year,
            "end_year": end_year
        }
        with TASK_LOG_FILE.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
        print(
            f"[TILE_TASK] column={column} row={row} task_id={task.id} "
            f"description={export_name}"
        )
    print(f"[INFO] Submitted {len(tasks)} spatial sample-point LST tasks.")
    return tasks
