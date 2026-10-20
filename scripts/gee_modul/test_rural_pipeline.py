import ee

from export_rural_monthly_lst import build_rural_inputs
from src.export_utils import (
    LST_BANDS,
    LST_OUTPUT_FIELDS,
    QC_OUTPUT_FIELDS,
    build_yearly_lst_statistics,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)


def main() -> None:
    config = load_config()
    initialize_gee(config)

    test_config = config["test"]
    city_config = config["cities"]
    rural_config = config["rural"]
    city_id = test_config["city_id"]
    year = test_config["year"]
    id_field = city_config["id_field"]
    tile_scale = config["processing"]["tile_scale"]

    all_cities = ee.FeatureCollection(city_config["asset"])
    selected_city = all_cities.filter(
        ee.Filter.eq(id_field, city_id)
    )
    city_count = selected_city.size().getInfo()
    if city_count != 1:
        raise ValueError(
            f"Expected exactly one city with {id_field}={city_id}, "
            f"found {city_count}."
        )

    projection, scale = get_modis_reference_grid(config)
    projection_info = projection.getInfo()

    rural_halos, rural_mask = build_rural_inputs(
        config=config,
        projection=projection,
        city_id=city_id
    )
    halo_count = rural_halos.size().getInfo()
    if halo_count != 1:
        raise AssertionError(
            f"Expected one rural halo for city {city_id}, found {halo_count}."
        )

    halo = ee.Feature(rural_halos.first())
    halo_geometry = halo.geometry()
    halo_area_km2 = halo_geometry.area(
        maxError=rural_config["max_error_m"]
    ).divide(1_000_000).getInfo()
    if halo_area_km2 <= 0:
        raise AssertionError(
            f"Rural halo area must be positive, got {halo_area_km2}."
        )

    mask_projection = rural_mask.projection().getInfo()
    if mask_projection["crs"] != projection_info["crs"]:
        raise AssertionError(
            "Rural mask projection does not match MODIS reference grid: "
            f"{mask_projection['crs']} != {projection_info['crs']}"
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
    results = monthly_stats.getInfo()["features"]
    if len(results) != 12:
        raise AssertionError(
            f"Expected 12 monthly rural records, found {len(results)}."
        )

    properties = [feature["properties"] for feature in results]
    expected_months = list(range(1, 13))
    actual_months = sorted(row.get("month") for row in properties)
    if actual_months != expected_months:
        raise AssertionError(
            f"Expected months {expected_months}, got {actual_months}."
        )

    required_fields = set(LST_OUTPUT_FIELDS) | set(QC_OUTPUT_FIELDS)
    missing_fields_by_month = {
        row["month"]: required_fields - set(row)
        for row in properties
        if required_fields - set(row)
    }
    if missing_fields_by_month:
        raise AssertionError(
            "Output is missing LST/QC fields by month: "
            f"{missing_fields_by_month}"
        )
    if any(row.get("year") != year for row in properties):
        raise AssertionError(f"Output contains records outside year {year}.")
    if any(row.get(id_field) != city_id for row in properties):
        raise AssertionError(
            f"Output contains a city ID other than {city_id}."
        )

    lst_mean_fields = {f"{band}_mean" for band in LST_BANDS}
    if not any(
        row.get(field) is not None
        for row in properties
        for field in lst_mean_fields
    ):
        raise AssertionError(
            "All rural monthly LST means are null for this city and year."
        )

    if not any(
        row.get(field) is not None and row[field] > 0
        for row in properties
        for field in QC_OUTPUT_FIELDS
    ):
        raise AssertionError(
            "No positive rural QC summary was produced for this city/year."
        )

    print(f"[PASS] City ID: {city_id}")
    print(f"[PASS] Year: {year}; monthly records: {len(results)}")
    print(f"[PASS] Rural halo area: {halo_area_km2:.2f} km^2")
    print(f"[PASS] Eligible MODIS rural pixels in halo: {rural_pixel_count}")
    print(
        f"[PASS] Rural mask CRS: {mask_projection['crs']} "
        f"(MODIS CRS: {projection_info['crs']})"
    )
    for row in properties:
        lst_values = ", ".join(
            f"{band}_mean={row.get(f'{band}_mean')}"
            for band in LST_BANDS
        )
        qc_values = ", ".join(
            f"{field}={row.get(field)}"
            for field in sorted(QC_OUTPUT_FIELDS)
        )
        print(
            f"[RESULT] {row['year']}-{row['month']:02d}: "
            f"{lst_values}; QC: {qc_values}"
        )


if __name__ == "__main__":
    main()
