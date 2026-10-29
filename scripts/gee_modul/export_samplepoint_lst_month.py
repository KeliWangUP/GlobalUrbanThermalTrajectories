from typing import Any

import ee

from src.export_utils import (
    build_monthly_samplepoint_lst_statistics,
    build_samplepoint_lst_selectors,
    export_table_to_drive,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.samplepoint_utils import load_sample_points


def export_samplepoint_lst_month(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> ee.batch.Task:
    """Submit one full-sample monthly MODIS LST CSV task."""
    export_config = config["export"]
    id_field = config["cities"]["id_field"]
    points, point_count = load_sample_points(config, ["pixel_id", id_field])

    month_config = config["samplepoint_lst_export"]
    year = month_config["year"]
    month = month_config["month"]
    if type(year) is not int or not (
        config["study"]["start_year"] <= year
        <= config["study"]["end_year"]
    ):
        raise ValueError(
            "samplepoint_lst_export.year must be an integer within the "
            "configured study period."
        )
    if type(month) is not int or not 1 <= month <= 12:
        raise ValueError(
            "samplepoint_lst_export.month must be an integer from 1 to 12."
        )

    collection = build_monthly_samplepoint_lst_statistics(
        year=year,
        month=month,
        points=points,
        config=config,
        projection=projection,
        scale=scale,
        id_field=id_field
    )
    dataset_name = export_config["samplepoint_lst_dataset_name"]
    export_name = (
        f"{dataset_name}_{config['project']['version']}_{year}_{month:02d}"
    )
    task = export_table_to_drive(
        collection=collection,
        description=export_name,
        folder=export_config["folder"],
        file_name=export_name,
        selectors=build_samplepoint_lst_selectors(id_field)
    )
    print(
        f"[INFO] Submitted {year}-{month:02d} LST export for "
        f"{point_count} sample points."
    )
    return task


def main() -> None:
    config = load_config()
    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)
    export_samplepoint_lst_month(config, projection, scale)


if __name__ == "__main__":
    main()
