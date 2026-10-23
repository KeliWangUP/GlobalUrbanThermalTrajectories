import ee

from src.batch_utils import filter_nonempty_geometries
from src.export_utils import (
    asset_exists,
    get_modis_reference_grid,
    initialize_gee,
    list_image_asset_names,
    load_config,
    run_annual_exports
)
from src.rural_reference_utils import (
    RURAL_FRACTION_BAND,
    apply_rural_fraction_threshold,
    get_priority_city_ids,
    rural_city_fraction_asset_id
)


def main() -> None:
    config = load_config()
    initialize_gee(config)

    rural_config = config["rural"]
    static_config = rural_config["static_assets"]
    export_config = config["export"]
    halo_asset = static_config["halo_asset"]
    fraction_collection = static_config["city_fraction_collection_asset"]
    if not asset_exists(halo_asset):
        raise FileNotFoundError(
            f"Rural halo asset not found: {halo_asset}. Run "
            "build_rural_halos_asset.py first."
        )
    if not asset_exists(fraction_collection):
        raise FileNotFoundError(
            "City fraction ImageCollection not found: "
            f"{fraction_collection}. Run "
            "build_rural_city_fraction_assets.py first."
        )
    if not asset_exists(config["cities"]["priority_points_asset"]):
        raise FileNotFoundError(
            "Priority city point asset not found: "
            f"{config['cities']['priority_points_asset']}"
        )

    halos = filter_nonempty_geometries(
        ee.FeatureCollection(halo_asset),
        rural_config["max_error_m"]
    )
    priority_ids = get_priority_city_ids(config)
    halo_ids = set(
        halos
        .aggregate_array(config["cities"]["id_field"])
        .distinct()
        .getInfo()
    )
    selected_ids = [
        city_id
        for city_id in priority_ids
        if city_id in halo_ids
    ]
    no_halo_count = len(priority_ids) - len(selected_ids)
    if not selected_ids:
        raise ValueError(
            "No priority city IDs have a non-empty rural halo."
        )

    existing_names = list_image_asset_names(fraction_collection)
    missing_assets = [
        rural_city_fraction_asset_id(config, city_id)
        for city_id in selected_ids
        if rural_city_fraction_asset_id(config, city_id)
        .rsplit("/", 1)[-1] not in existing_names
    ]
    if missing_assets:
        raise FileNotFoundError(
            f"{len(missing_assets)} priority cities are missing fraction "
            "assets. Finish build_rural_city_fraction_assets.py first. "
            f"Examples: {', '.join(missing_assets[:5])}"
        )

    id_field = config["cities"]["id_field"]
    regions = halos.filter(ee.Filter.inList(id_field, selected_ids))
    city_fractions = (
        ee.ImageCollection(fraction_collection)
        .filter(ee.Filter.inList("city_id", selected_ids))
        .select(RURAL_FRACTION_BAND)
    )
    rural_mask = apply_rural_fraction_threshold(
        config,
        city_fractions.mosaic()
    )
    projection, scale = get_modis_reference_grid(config)

    print(
        f"[INFO] {len(priority_ids)} priority city polygons matched points; "
        f"{len(selected_ids)} have non-empty halos; "
        f"{no_halo_count} have no non-empty halo."
    )
    print(
        f"[INFO] Submitting one annual Drive CSV task per year for all "
        f"{len(selected_ids)} priority cities."
    )
    run_annual_exports(
        dataset_name=export_config["rural_priority_dataset_name"],
        regions=regions,
        config=config,
        projection=projection,
        scale=scale,
        spatial_mask=rural_mask
    )


if __name__ == "__main__":
    main()
