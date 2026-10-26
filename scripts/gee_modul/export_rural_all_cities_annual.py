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
    get_missing_rural_city_fraction_assets,
    resolve_rural_city_ids
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

    projection, scale = get_modis_reference_grid(config)
    halos = filter_nonempty_geometries(
        ee.FeatureCollection(halo_asset),
        rural_config["max_error_m"]
    )
    city_ids = resolve_rural_city_ids(config, halos)
    missing_assets = get_missing_rural_city_fraction_assets(
        config=config,
        city_ids=city_ids,
        halos=halos,
        projection=projection,
        existing_names=list_image_asset_names(fraction_collection)
    )
    if missing_assets:
        raise FileNotFoundError(
            f"{len(missing_assets)} city fraction assets are missing. "
            "Complete build_rural_city_fraction_assets.py first. Examples: "
            f"{', '.join(missing_assets[:5])}"
        )

    id_field = config["cities"]["id_field"]
    regions = halos.filter(ee.Filter.inList(id_field, city_ids))
    city_fractions = (
        ee.ImageCollection(fraction_collection)
        .filter(ee.Filter.inList("city_id", city_ids))
        .select(RURAL_FRACTION_BAND)
    )
    rural_mask = apply_rural_fraction_threshold(
        config,
        city_fractions.mosaic()
    )

    dataset_name = export_config["rural_all_cities_dataset_name"]
    print(
        f"[INFO] Exporting all {len(city_ids)} cities with non-empty rural "
        f"halos, years {config['study']['start_year']}-"
        f"{config['study']['end_year']}."
    )
    print(
        f"[INFO] All required city fraction assets exist in "
        f"{fraction_collection}."
    )
    run_annual_exports(
        dataset_name=dataset_name,
        regions=regions,
        config=config,
        projection=projection,
        scale=scale,
        spatial_mask=rural_mask
    )


if __name__ == "__main__":
    main()
