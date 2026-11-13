import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee

from src.city_polygon_isa import (
    CITY_POLYGON_AREA_FIELD,
    build_city_polygon_isa_area_image,
    build_city_polygon_isa_statistics,
    validate_city_polygon_isa_config
)
from src.export_utils import CONFIG_PATH, export_table_to_drive
from src.samplepoint_attributes import prepare_isa_source


def export_city_polygon_isa(config: dict[str, Any]) -> ee.batch.Task:
    years, selectors = validate_city_polygon_isa_config(config)
    settings = config["city_polygon_isa"]
    city_config = config["cities"]
    cities = ee.FeatureCollection(city_config["asset"])
    city_count = cities.size().getInfo()
    if city_count == 0:
        raise ValueError(
            f"City boundary asset is empty: {city_config['asset']}"
        )

    isa_image, isa_band, _ = prepare_isa_source(config)
    area_image = build_city_polygon_isa_area_image(isa_image, years)
    collection = build_city_polygon_isa_statistics(
        cities=cities,
        area_image=area_image,
        years=years,
        scale_m=settings["scale_m"],
        area_max_error_m=settings["area_max_error_m"],
        tile_scale=config["processing"]["tile_scale"]
    )

    description = f"{settings['dataset_name']}_{settings['version']}"
    task = export_table_to_drive(
        collection=collection,
        description=description,
        folder=config["export"]["folder"],
        file_name=description,
        selectors=selectors
    )

    log_path = Path(settings["task_log"])
    if not log_path.is_absolute():
        log_path = CONFIG_PATH.parent / log_path
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task.id,
        "description": description,
        "city_asset": city_config["asset"],
        "city_id_field": city_config["id_field"],
        "city_count": city_count,
        "isa_asset": config["samplepoint_attributes"]["isa_asset"],
        "isa_band": isa_band,
        "years": list(years),
        "scale_m": settings["scale_m"],
        "fraction_denominator": CITY_POLYGON_AREA_FIELD,
        "masked_gaia_pixels": "zero_contribution"
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    print(
        f"[CITY_ISA_TASK] cities={city_count} task_id={task.id} "
        f"description={description}"
    )
    return task
