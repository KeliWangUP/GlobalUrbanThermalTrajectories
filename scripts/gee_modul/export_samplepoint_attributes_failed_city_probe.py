from typing import Any

import ee

from export_samplepoint_attributes import export_samplepoint_attributes
from export_samplepoint_attributes_failed_batch import (
    FAILED_BATCH_FILE,
    read_failed_batch
)
from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)


def main() -> None:
    config = load_config()
    initialize_gee(config)
    batch_index, expected_batch_count, batch_name = read_failed_batch(
        FAILED_BATCH_FILE
    )
    point_asset = config["export"]["asset_root"]
    id_field = config["cities"]["id_field"]
    points = ee.FeatureCollection(point_asset)
    city_ids = points.aggregate_array(id_field).distinct().sort().getInfo()
    if not city_ids or any(city_id is None for city_id in city_ids):
        raise ValueError(
            "Sample-point asset has no valid city IDs."
        )
    batch_size = config["sample_points"]["city_batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError(
            "sample_points.city_batch_size must be a positive integer."
        )
    current_batch_count = (len(city_ids) + batch_size - 1) // batch_size
    if current_batch_count != expected_batch_count:
        raise ValueError(
            "The failed task's batch count does not match the current "
            f"sample-point asset/config ({expected_batch_count} vs "
            f"{current_batch_count})."
        )

    start = (batch_index - 1) * batch_size
    batch_city_ids = city_ids[start:start + batch_size]
    if not batch_city_ids:
        raise ValueError(f"No city IDs found for failed batch '{batch_name}'.")
    batch_points = points.filter(
        ee.Filter.inList(id_field, batch_city_ids)
    )
    point_counts: dict[str, Any] = batch_points.aggregate_histogram(
        id_field
    ).getInfo()
    selected_city_id = max(
        batch_city_ids,
        key=lambda city_id: int(point_counts.get(str(city_id), 0))
    )
    print(
        f"[INFO] Failed batch {batch_index:04d} of "
        f"{expected_batch_count:04d}; probing city {selected_city_id} "
        f"with {point_counts.get(str(selected_city_id), 0)} sample points."
    )

    projection, scale = get_modis_reference_grid(config)
    tasks = export_samplepoint_attributes(
        config=config,
        projection=projection,
        scale=scale,
        expected_batch_count=expected_batch_count,
        selected_city_id=selected_city_id
    )
    if len(tasks) != 1:
        raise RuntimeError(
            f"Expected one city probe task, got {len(tasks)}."
        )


if __name__ == "__main__":
    main()
