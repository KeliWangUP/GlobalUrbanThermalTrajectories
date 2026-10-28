from typing import Any

import ee

from src.export_utils import (
    asset_exists,
    build_monthly_samplepoint_lst_statistics,
    build_samplepoint_lst_selectors,
    export_table_to_drive,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)


def export_samplepoint_lst_month(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> ee.batch.Task:
    """Submit one full-sample monthly MODIS LST CSV task."""
    export_config = config["export"]
    point_asset = export_config["asset_root"]
    if not asset_exists(point_asset):
        raise FileNotFoundError(
            f"Sample-point asset not found: {point_asset}. "
            "Build or upload the MODIS sample-point asset first."
        )

    points = ee.FeatureCollection(point_asset)
    point_count = points.size().getInfo()
    if point_count == 0:
        raise ValueError(f"Sample-point asset is empty: {point_asset}")

    id_field = config["cities"]["id_field"]
    required_fields = ["pixel_id", id_field]
    available_fields = ee.Feature(points.first()).propertyNames().getInfo()
    missing_fields = [
        field for field in required_fields
        if field not in available_fields
    ]
    if missing_fields:
        raise ValueError(
            "Sample-point asset is missing required properties: "
            f"{missing_fields}"
        )

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
