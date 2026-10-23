import ee

from src.batch_utils import (
    filter_nonempty_geometries,
    get_tile_city_counts,
    TILE_ID_PROPERTY
)
from src.export_utils import (
    LST_OUTPUT_FIELDS,
    QC_OUTPUT_FIELDS,
    asset_exists,
    build_yearly_lst_statistics,
    get_modis_reference_grid,
    initialize_gee,
    list_image_asset_names,
    load_config
)
from src.pipeline_checks import (
    check_outputs_against_diagnostics,
    print_monthly_results,
    select_single_city,
    validate_monthly_records
)
from src.rural_reference_utils import (
    apply_rural_fraction_threshold,
    get_missing_rural_city_fraction_assets,
    load_rural_city_fraction
)


def main() -> None:
    config = load_config()
    initialize_gee(config)

    city_id = config["test"]["city_id"]
    year = config["test"]["year"]
    id_field = config["cities"]["id_field"]
    tile_scale = config["processing"]["tile_scale"]
    projection, scale = get_modis_reference_grid(config)

    halo_asset = config["rural"]["static_assets"]["halo_asset"]
    if not asset_exists(halo_asset):
        raise FileNotFoundError(
            f"Rural halo asset not found: {halo_asset}. Run "
            "build_rural_halos_asset.py first."
        )
    all_halos = filter_nonempty_geometries(
        ee.FeatureCollection(halo_asset),
        config["rural"]["max_error_m"]
    )
    rural_halos = select_single_city(
        all_halos,
        id_field,
        city_id
    )
    city_tile_id = ee.Feature(
        rural_halos.first()
    ).get(TILE_ID_PROPERTY).getInfo()
    tile_ids = list(get_tile_city_counts(all_halos))
    if city_tile_id not in tile_ids:
        raise ValueError(f"City {city_id} is not assigned to a MODIS tile.")
    batch_index = tile_ids.index(city_tile_id)
    fraction_collection = config["rural"]["static_assets"][
        "city_fraction_collection_asset"
    ]
    if not asset_exists(fraction_collection):
        raise FileNotFoundError(
            f"City fraction collection not found: {fraction_collection}. Run "
            "build_rural_city_fraction_assets.py first."
        )
    missing_fraction_assets = get_missing_rural_city_fraction_assets(
        config=config,
        city_ids=[city_id],
        halos=rural_halos,
        projection=projection,
        existing_names=list_image_asset_names(fraction_collection)
    )
    if missing_fraction_assets:
        raise FileNotFoundError(
            "City fraction assets not found: "
            f"{', '.join(missing_fraction_assets)}. Run "
            "build_rural_city_fraction_assets.py first."
        )

    print(
        f"[CHECK] City ID: {city_id} in batch {batch_index:03d} "
        f"(MODIS tile {city_tile_id})"
    )

    projection_info = projection.getInfo()
    rural_fraction = load_rural_city_fraction(config, city_id)
    rural_mask = apply_rural_fraction_threshold(
        config,
        rural_fraction
    )

    mask_crs = rural_mask.projection().getInfo()["crs"]
    if mask_crs != projection_info["crs"]:
        raise AssertionError(
            "Rural mask projection does not match MODIS reference grid: "
            f"{mask_crs} != {projection_info['crs']}"
        )

    halo = ee.Feature(rural_halos.first())
    halo_geometry = halo.geometry()
    halo_area_km2 = (
        halo_geometry
        .area(maxError=config["rural"]["max_error_m"])
        .divide(1_000_000)
        .getInfo()
    )
    if halo_area_km2 <= 0:
        raise AssertionError(
            f"Rural halo area must be positive, got {halo_area_km2}."
        )

    rural_pixel_count = (
        rural_mask
        .selfMask()
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=halo_geometry,
            crs=projection,
            scale=scale,
            maxPixels=10_000_000,
            tileScale=tile_scale
        )
        .get("rural_modis_mask")
        .getInfo()
    )
    if not rural_pixel_count:
        raise AssertionError(
            f"No eligible MODIS rural pixels were found in city {city_id}'s "
            "rural halo."
        )

    monthly_stats = build_yearly_lst_statistics(
        year=year,
        regions=rural_halos,
        config=config,
        projection=projection,
        scale=scale,
        tile_scale=tile_scale,
        id_field=id_field,
        spatial_mask=rural_mask
    )
    properties = [
        feature["properties"]
        for feature in monthly_stats.getInfo()["features"]
    ]
    validate_monthly_records(properties, year, id_field, city_id)
    check_outputs_against_diagnostics(
        properties=properties,
        year=year,
        config=config,
        region=halo_geometry,
        projection=projection,
        scale=scale,
        tile_scale=tile_scale,
        spatial_mask=rural_mask
    )

    print(f"[PASS] City ID: {city_id}; year: {year}")
    print(f"[PASS] Monthly records: {len(properties)}")
    print(f"[PASS] LST fields: {len(LST_OUTPUT_FIELDS)}")
    print(f"[PASS] QC fields: {len(QC_OUTPUT_FIELDS)}")
    print(f"[PASS] Rural halo area: {halo_area_km2:.2f} km^2")
    print(f"[PASS] Eligible MODIS rural pixels in halo: {rural_pixel_count}")
    print(
        f"[PASS] Rural mask CRS: {mask_crs} "
        f"(MODIS CRS: {projection_info['crs']})"
    )
    print_monthly_results(properties)


if __name__ == "__main__":
    main()
