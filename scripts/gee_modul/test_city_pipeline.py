import ee

from src.export_utils import (
    build_yearly_lst_statistics,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)

LST_BANDS = [
    "terra_day",
    "terra_night",
    "aqua_day",
    "aqua_night"
]

LST_STATISTICS = [
    "mean",
    "median",
    "stdDev",
    "count"
]

EXPECTED_LST_FIELDS = {
    f"{band}_{stat}"
    for band in LST_BANDS
    for stat in LST_STATISTICS
}

EXPECTED_LST_MEAN_FIELDS = {
    f"{band}_mean"
    for band in LST_BANDS
}

QC_BANDS = [
    "terra_day_clear_days",
    "terra_day_n8day",
    "terra_night_clear_days",
    "terra_night_n8day",
    "aqua_day_clear_days",
    "aqua_day_n8day",
    "aqua_night_clear_days",
    "aqua_night_n8day"
]

EXPECTED_QC_FIELDS = {
    f"{band}_mean"
    for band in QC_BANDS
}


def main() -> None:
    config = load_config()
    initialize_gee(config)

    test_config = config["test"]
    city_config = config["cities"]
    city_id = test_config["city_id"]
    year = test_config["year"]
    id_field = city_config["id_field"]

    cities = ee.FeatureCollection(city_config["asset"])
    city = cities.filter(ee.Filter.eq(id_field, city_id))
    city_count = city.size().getInfo()
    if city_count != 1:
        raise ValueError(
            f"Expected exactly one city with {id_field}={city_id}, "
            f"found {city_count}."
        )

    projection, scale = get_modis_reference_grid(config)
    projection_info = projection.getInfo()
    print(f"[CHECK] City ID: {city_id}")
    print(f"[CHECK] Test year: {year}")
    print(f"[CHECK] MODIS CRS: {projection_info['crs']}")
    print(f"[CHECK] MODIS nominal scale: {scale:.3f} m")

    monthly_stats = build_yearly_lst_statistics(
        year=year,
        regions=city,
        config=config,
        projection=projection,
        scale=scale,
        tile_scale=config["processing"]["tile_scale"],
        id_field=id_field
    )
    results = monthly_stats.getInfo()["features"]

    if len(results) != 12:
        raise AssertionError(
            f"Expected 12 monthly records for one city, found {len(results)}."
        )

    expected_months = list(range(1, 13))
    actual_months = sorted(
        feature["properties"].get("month")
        for feature in results
    )
    if actual_months != expected_months:
        raise AssertionError(
            f"Expected months {expected_months}, got {actual_months}."
        )

    properties = [feature["properties"] for feature in results]
    if any(row.get("year") != year for row in properties):
        raise AssertionError(f"Output contains records outside year {year}.")
    if any(row.get(id_field) != city_id for row in properties):
        raise AssertionError(
            f"Output contains a city ID other than {city_id}."
        )

    required_fields = EXPECTED_LST_FIELDS | EXPECTED_QC_FIELDS
    missing_fields_by_month = {
        row["month"]: required_fields - set(row)
        for row in properties
        if required_fields - set(row)
    }
    if missing_fields_by_month:
        raise AssertionError(
            "Output is missing expected LST/QC fields by month: "
            f"{missing_fields_by_month}"
        )

    if not any(
        row.get(field) is not None
        for row in properties
        for field in EXPECTED_LST_MEAN_FIELDS
    ):
        raise AssertionError(
            "All monthly mean LST values are null for this city and year."
        )

    invalid_qc_values = []
    for row in properties:
        for field in EXPECTED_QC_FIELDS:
            value = row.get(field)
            if value is not None and value < 0:
                invalid_qc_values.append((row["month"], field, value))
    if invalid_qc_values:
        raise AssertionError(
            f"QC summaries contain negative values: {invalid_qc_values}"
        )

    if not any(
        row.get(field) is not None and row[field] > 0
        for row in properties
        for field in EXPECTED_QC_FIELDS
    ):
        raise AssertionError(
            "No positive QC summary was produced for this city and year."
        )

    print(f"[PASS] One city selected: {city_id}")
    print(f"[PASS] Monthly records: {len(results)}")
    print(f"[PASS] LST fields: {len(EXPECTED_LST_FIELDS)}")
    print(f"[PASS] QC fields: {len(EXPECTED_QC_FIELDS)}")
    for row in properties:
        lst_values = ", ".join(
            f"{band}_mean={row.get(f'{band}_mean')}"
            for band in LST_BANDS
        )
        qc_values = ", ".join(
            f"{field}={row.get(field)}"
            for field in sorted(EXPECTED_QC_FIELDS)
        )
        print(
            f"[RESULT] {row['year']}-{row['month']:02d}: "
            f"{lst_values}; QC: {qc_values}"
        )


if __name__ == "__main__":
    main()
