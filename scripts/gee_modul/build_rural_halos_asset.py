import ee

from src.batch_utils import (
    assign_cities_to_modis_tiles,
    filter_nonempty_geometries
)
from src.export_utils import (
    asset_exists,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.rural_reference_utils import build_all_rural_halos


def main() -> None:
    config = load_config()
    initialize_gee(config)

    static_config = config["rural"]["static_assets"]
    halo_asset = static_config["halo_asset"]
    if asset_exists(halo_asset):
        print(f"[SKIP] Rural halo asset exists: {halo_asset}")
        return

    projection, _ = get_modis_reference_grid(config)
    cities = assign_cities_to_modis_tiles(config, projection)
    source_cities = ee.FeatureCollection(config["cities"]["asset"])
    city_count = source_cities.size().getInfo()
    assigned_count = cities.size().getInfo()
    city_ids = (
        source_cities
        .aggregate_array(config["cities"]["id_field"])
        .getInfo()
    )
    if any(city_id is None for city_id in city_ids):
        raise ValueError(
            f"Some cities have a null {config['cities']['id_field']}."
        )
    if len(set(city_ids)) != len(city_ids):
        raise ValueError(
            f"{config['cities']['id_field']} values are not unique."
        )
    if assigned_count != city_count:
        raise ValueError(
            f"Only {assigned_count} of {city_count} cities intersected "
            "the MODIS tile grid."
        )

    rural_config = config["rural"]
    halos = build_all_rural_halos(
        cities=cities,
        inner_buffer_m=rural_config["inner_buffer_m"],
        outer_buffer_m=rural_config["outer_buffer_m"],
        city_id_field=config["cities"]["id_field"],
        max_error_m=rural_config["max_error_m"]
    )
    halos = filter_nonempty_geometries(
        halos,
        rural_config["max_error_m"]
    )
    task = ee.batch.Export.table.toAsset(
        collection=halos,
        description=static_config["halo_asset_description"],
        assetId=halo_asset
    )
    task.start()
    print(f"[EXPORT] {halo_asset}")
    print(f"[TASK] {task.id}")
    print(
        "[NEXT] After the halo asset task completes, run "
        "build_rural_fraction_assets.py."
    )


if __name__ == "__main__":
    main()
