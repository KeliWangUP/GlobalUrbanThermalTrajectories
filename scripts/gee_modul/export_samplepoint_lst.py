from typing import Any

import ee

from src.export_utils import (
    build_samplepoint_lst_selectors,
    build_yearly_samplepoint_lst_statistics,
    export_table_to_drive,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.samplepoint_utils import load_sample_points


def export_samplepoint_lst(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> list[ee.batch.Task]:
    """Submit one monthly MODIS LST CSV task per study year."""
    export_config = config["export"]
    point_asset = export_config["asset_root"]
    id_field = config["cities"]["id_field"]
    points, point_count = load_sample_points(config, ["pixel_id", id_field])

    city_ids = points.aggregate_array(id_field).distinct().size().getInfo()
    if city_ids == 0:
        raise ValueError(
            f"Sample-point asset has no valid values in '{id_field}'."
        )

    dataset_name = export_config["samplepoint_lst_dataset_name"]
    version = config["project"]["version"]
    study = config["study"]
    selectors = build_samplepoint_lst_selectors(id_field)
    tasks = []

    print(
        f"[INFO] Sample-point asset: {point_asset}; "
        f"{point_count} points across {city_ids} cities."
    )
    for year in range(study["start_year"], study["end_year"] + 1):
        collection = build_yearly_samplepoint_lst_statistics(
            year=year,
            points=points,
            config=config,
            projection=projection,
            scale=scale,
            id_field=id_field
        )
        export_name = f"{dataset_name}_{version}_{year}"
        tasks.append(export_table_to_drive(
            collection=collection,
            description=export_name,
            folder=export_config["folder"],
            file_name=export_name,
            selectors=selectors
        ))

    print(f"[INFO] Submitted {len(tasks)} sample-point LST export tasks.")
    return tasks


def main() -> None:
    config = load_config()
    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)
    export_samplepoint_lst(config, projection, scale)


if __name__ == "__main__":
    main()
