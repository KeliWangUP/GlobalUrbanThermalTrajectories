import ee

from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.rural_reference_utils import build_rural_halos_with_metrics


def main() -> None:
    config = load_config()
    initialize_gee(config)

    projection, scale = get_modis_reference_grid(config)
    city_config = config["cities"]
    rural_config = config["rural"]
    static_assets = rural_config["static_assets"]
    rural_fraction = ee.Image(
        static_assets["fraction_asset"]
    ).select("rural_fraction")
    cities = ee.FeatureCollection(city_config["asset"])

    rural_halos = build_rural_halos_with_metrics(
        cities=cities,
        rural_fraction=rural_fraction,
        city_id_field=city_config["id_field"],
        inner_buffer_m=rural_config["inner_buffer_m"],
        outer_buffer_m=rural_config["outer_buffer_m"],
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
