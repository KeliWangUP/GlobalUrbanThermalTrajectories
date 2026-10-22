import ee

from export_rural_monthly_lst import build_rural_halos, build_rural_mask
from src.batch_utils import (
    compute_batch_assignment,
    load_city_batch,
    verify_city_batches
)
from src.export_utils import (
    LST_OUTPUT_FIELDS,
    QC_OUTPUT_FIELDS,
    build_yearly_lst_statistics,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.pipeline_checks import (
    check_outputs_against_diagnostics,
    print_monthly_results,
    select_single_city,
    validate_monthly_records
)


def main() -> None:
    config = load_config()
    initialize_gee(config)

    city_id = config["test"]["city_id"]
    year = config["test"]["year"]
    id_field = config["cities"]["id_field"]
    tile_scale = config["processing"]["tile_scale"]

    # Use the stored batch asset that the production export reads.
    assignment = compute_batch_assignment(config)
    batch_index = next(
        (
            index
            for index, city_ids in enumerate(assignment)
            if city_id in city_ids
        ),
        None
    )
    if batch_index is None:
        raise ValueError(f"City {city_id} is not in any city batch.")
    verify_city_batches(config, [batch_index], assignment)
    city = select_single_city(
        load_city_batch(config, batch_index),
        id_field,
        city_id
    )
    print(f"[CHECK] City ID: {city_id} in batch {batch_index:02d}")

    projection, scale = get_modis_reference_grid(config)
    projection_info = projection.getInfo()
    rural_halos = build_rural_halos(config, city)
    rural_mask = build_rural_mask(config, projection)

    mask_crs = rural_mask.projection().getInfo()["crs"]
    if mask_crs != projection_info["crs"]:
        raise AssertionError(
            "Rural mask projection does not match MODIS reference grid: "
            f"{mask_crs} != {projection_info['crs']}"
        )

    halo = ee.Feature(select_single_city(
        rural_halos,
        id_field,
        city_id
    ).first())
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
