import ee

from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.rural_reference_utils import (
    build_all_rural_halos,
    build_rural_halos_with_metrics,
    get_rural_modis_chunk_ids
)


def main() -> None:
    config = load_config()
    initialize_gee(config)

    projection, scale = get_modis_reference_grid(config)
    city_config = config["cities"]
    rural_config = config["rural"]
    static_assets = rural_config["static_assets"]
    grid_config = config["reference_grid"]["tile_grid"]
    cities = ee.FeatureCollection(city_config["asset"])
    rural_halos = build_all_rural_halos(
        cities=cities,
        inner_buffer_m=rural_config["inner_buffer_m"],
        outer_buffer_m=rural_config["outer_buffer_m"],
        city_id_field=city_config["id_field"],
        max_error_m=rural_config["max_error_m"]
    )
    projection_info = projection.getInfo()

    expected_chunk_ids = set(
        get_rural_modis_chunk_ids(
            rural_halos=rural_halos,
            max_error_m=rural_config["max_error_m"],
            projection_crs=projection_info["crs"],
            scale=scale,
            tile_grid=grid_config
        )
    )
    fraction_collection = ee.ImageCollection(
        static_assets["fraction_collection_asset"]
    )
    actual_chunk_ids = set(
        fraction_collection
        .aggregate_array("chunk_id")
        .getInfo()
        or []
    )
    missing_chunk_ids = sorted(expected_chunk_ids - actual_chunk_ids)
    if missing_chunk_ids:
        sample = ", ".join(missing_chunk_ids[:10])
        raise ValueError(
            "Rural fraction chunks are incomplete; wait for all image "
            "exports to finish or rerun build_rural_fraction_asset.py. "
            f"Missing {len(missing_chunk_ids)} chunk(s): {sample}"
        )

    rural_fraction = fraction_collection.mosaic().select(
        "rural_fraction"
    )

    rural_halos = build_rural_halos_with_metrics(
        rural_halos=rural_halos,
        rural_fraction=rural_fraction,
        city_id_field=city_config["id_field"],
        max_error_m=rural_config["max_error_m"],
        minimum_rural_fraction=(
            rural_config["minimum_rural_fraction_per_modis_pixel"]
        ),
        projection=projection,
        scale=scale,
        tile_scale=config["processing"]["tile_scale"]
    )

    asset_id = static_assets["halos_asset"]
    task = ee.batch.Export.table.toAsset(
        collection=rural_halos,
        description=asset_id.rsplit("/", 1)[-1],
        assetId=asset_id
    )
    task.start()
    print(f"[EXPORT] Rural halo asset: {asset_id}")
    print(f"[TASK] {task.id}")


if __name__ == "__main__":
    main()
