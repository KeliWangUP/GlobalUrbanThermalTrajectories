from typing import Any

import ee

from src.export_utils import (
    asset_exists,
    export_table_to_drive,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.samplepoint_attributes import (
    build_samplepoint_attribute_collection,
    build_samplepoint_selectors,
    prepare_samplepoint_sources
)


def export_samplepoint_attributes(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float,
    selected_batch_index: int | None = None,
    expected_batch_count: int | None = None,
    selected_city_id: Any | None = None
) -> list[ee.batch.Task]:
    """Submit point-composition exports, optionally for one city batch."""
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
    required_fields = build_samplepoint_selectors(config)
    base_fields = [
        "pixel_id",
        id_field,
        "modis_x",
        "modis_y"
    ]
    if config["sample_points"]["include_lonlat"]:
        base_fields.extend(["lon", "lat"])
    available_fields = ee.Feature(points.first()).propertyNames().getInfo()
    missing_fields = [
        field for field in base_fields
        if field not in available_fields
    ]
    if missing_fields:
        raise ValueError(
            "Sample-point asset is missing required properties: "
            f"{missing_fields}"
        )

    city_ids = points.aggregate_array(id_field).distinct().sort().getInfo()
    if not city_ids or any(city_id is None for city_id in city_ids):
        raise ValueError(
            "Sample-point asset has no valid city IDs in "
            f"'{id_field}'."
        )
    city_batch_size = config["sample_points"]["city_batch_size"]
    if (
        isinstance(city_batch_size, bool)
        or not isinstance(city_batch_size, int)
        or city_batch_size < 1
    ):
        raise ValueError(
            "sample_points.city_batch_size must be a positive integer."
        )

    sources = prepare_samplepoint_sources(config)
    print(
        f"[INFO] Sample-point asset: {point_asset}; "
        f"{point_count} points across {len(city_ids)} cities."
    )
    print(f"[INFO] ISA band: {sources.isa_band}.")

    dataset_name = export_config["samplepoint_attributes_dataset_name"]
    version = config["project"]["version"]
    folder = export_config["folder"]
    tasks = []
    batch_count = (
        len(city_ids) + city_batch_size - 1
    ) // city_batch_size
    if selected_city_id is not None and selected_batch_index is not None:
        raise ValueError(
            "Select either a city or a batch, not both."
        )
    if (
        expected_batch_count is not None
        and (
            type(expected_batch_count) is not int
            or expected_batch_count != batch_count
        )
    ):
        raise ValueError(
            "The failed task's batch count does not match the current "
            f"sample-point asset/config ({expected_batch_count} vs "
            f"{batch_count}). Refusing to retry a potentially different "
            "set of cities."
        )
    if selected_city_id is not None and selected_city_id not in city_ids:
        raise ValueError(
            f"Selected city ID {selected_city_id} is not present in the "
            "sample-point asset."
        )
    if (
        selected_batch_index is not None
        and (
            type(selected_batch_index) is not int
            or not 1 <= selected_batch_index <= batch_count
        )
    ):
        raise ValueError(
            "selected_batch_index must be within the configured batch range "
            f"1-{batch_count}."
        )

    if selected_city_id is not None:
        batch_specs = [(None, [selected_city_id])]
    elif selected_batch_index is not None:
        start = (selected_batch_index - 1) * city_batch_size
        batch_specs = [(
            selected_batch_index,
            city_ids[start:start + city_batch_size]
        )]
    else:
        batch_specs = [
            (
                batch_index,
                city_ids[start:start + city_batch_size]
            )
            for batch_index, start in enumerate(
                range(0, len(city_ids), city_batch_size),
                start=1
            )
        ]

    for batch_index, batch_city_ids in batch_specs:
        batch_points = points.filter(
            ee.Filter.inList(id_field, batch_city_ids)
        )
        collection = build_samplepoint_attribute_collection(
            points=batch_points,
            config=config,
            projection=projection,
            modis_scale=scale,
            sources=sources
        )
        if selected_city_id is not None:
            export_name = (
                f"{dataset_name}_{version}_city_{selected_city_id}_probe"
            )
            print(f"[INFO] Submitting city probe: {selected_city_id}.")
        else:
            suffix = (
                f"city_batch_{batch_index:04d}_of_{batch_count:04d}"
            )
            export_name = f"{dataset_name}_{version}_{suffix}"
            if selected_batch_index is not None:
                export_name = f"{export_name}_retry"
            print(
                f"[INFO] Submitting batch {batch_index}/{batch_count}: "
                f"{len(batch_city_ids)} cities; IDs={batch_city_ids}"
            )
        tasks.append(export_table_to_drive(
            collection=collection,
            description=export_name,
            folder=folder,
            file_name=export_name,
            selectors=required_fields
        ))

    print(f"[INFO] Submitted {len(tasks)} sample-point export tasks.")
    return tasks


def main() -> None:
    config = load_config()
    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)
    export_samplepoint_attributes(config, projection, scale)


if __name__ == "__main__":
    main()
