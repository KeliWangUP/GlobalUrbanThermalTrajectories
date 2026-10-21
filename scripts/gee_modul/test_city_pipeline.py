import ee

from src.modis_lst_utils import (
    compute_a2_validity,
    get_monthly_a2_collection
)
from src.export_utils import (
    LST_BANDS,
    LST_OUTPUT_FIELDS,
    QC_OUTPUT_FIELDS,
    build_yearly_lst_statistics,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)

EXPECTED_LST_MEAN_FIELDS = {
    f"{band}_mean"
    for band in LST_BANDS
}


def diagnose_a2_month(
    year: int,
    month: int,
    slot_name: str,
    config: dict,
    region: ee.Geometry,
    projection: ee.Projection,
    scale: float,
    tile_scale: int
) -> dict[str, float]:
    """Summarize per-composite production validity for one city/month/slot."""

    slot_config = config["modis"]["slots"][slot_name]
    collection = get_monthly_a2_collection(
        year=year,
        month=month,
        collection_id=slot_config["collection"]
    ).sort("system:time_start")

    count = collection.size().getInfo()
    print(f"[DIAG] {slot_name} {year}-{month:02d}: {count} A2 composites")

    image_list = collection.toList(count)
    total_final_valid = 0.0
    total_final_clear = 0.0
    qa_config = config["modis"]["qa"]

    for index in range(count):
        image = ee.Image(image_list.get(index))
        validity = compute_a2_validity(
            image=image,
            lst_band=slot_config["lst_band"],
            qc_band=slot_config["qc_band"],
            clear_band=slot_config["clear_band"],
            qa_cfg=qa_config
        )

        stats = validity.select([
            "qa_pass",
            "final_valid",
            "final_clear_count"
        ]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            crs=projection,
            scale=scale,
            maxPixels=10_000_000,
            tileScale=tile_scale
        ).getInfo()

        final_valid = stats.get("final_valid") or 0.0
        final_clear = stats.get("final_clear_count") or 0.0
        total_final_valid += final_valid
        total_final_clear += final_clear

        start_date = (
            ee.Date(image.get("system:time_start"))
            .format("YYYY-MM-dd")
            .getInfo()
        )
        print(
            f"[DIAG] {start_date} | "
            f"qaPass={stats.get('qa_pass', 0):.3f} | "
            f"final={final_valid:.3f} | "
            f"clear_final={final_clear:.2f}"
        )

    print(
        f"[DIAG-SUM] {slot_name} {year}-{month:02d} | "
        f"n8day_mean≈{total_final_valid:.3f} | "
        f"clear_days_mean≈{total_final_clear:.3f}"
    )
    return {
        "n8day_mean": total_final_valid,
        "clear_days_mean": total_final_clear
    }


def main() -> None:
    config = load_config()
    initialize_gee(config)

    test_config = config["test"]
    city_config = config["cities"]
    city_id = test_config["city_id"]
    year = test_config["year"]
    id_field = city_config["id_field"]
    qa_config = config["modis"]["qa"]

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
    city_geometry = ee.Feature(city.first()).geometry()
    print(f"[CHECK] City ID: {city_id}")
    print(f"[CHECK] Test year: {year}")
    print(f"[CHECK] MODIS CRS: {projection_info['crs']}")
    print(f"[CHECK] MODIS nominal scale: {scale:.3f} m")
    print(
        "[CHECK] QA config: "
        f"apply_qa_mask={qa_config.get('apply_qa_mask', True)}, "
        f"mandatory_qa_max={qa_config.get('mandatory_qa_max')}, "
        f"max_lst_error_code={qa_config.get('max_lst_error_code')}, "
        "require_good_data_quality="
        f"{qa_config.get('require_good_data_quality', False)}"
    )

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
    properties = [feature["properties"] for feature in results]

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

    if any(row.get("year") != year for row in properties):
        raise AssertionError(f"Output contains records outside year {year}.")
    if any(row.get(id_field) != city_id for row in properties):
        raise AssertionError(
            f"Output contains a city ID other than {city_id}."
        )

    required_fields = set(LST_OUTPUT_FIELDS) | set(QC_OUTPUT_FIELDS)
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
        for field in QC_OUTPUT_FIELDS:
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
        for field in QC_OUTPUT_FIELDS
    ):
        raise AssertionError(
            "No positive QC summary was produced for this city and year."
        )

    output_by_month = {
        row["month"]: row
        for row in properties
    }
    diagnostic_cases = [
        (5, "aqua_night"),
        (7, "aqua_night"),
        (7, "aqua_day")
    ]
    tolerance = 1e-5
    for diagnostic_month, slot_name in diagnostic_cases:
        diagnostic = diagnose_a2_month(
            year=year,
            month=diagnostic_month,
            slot_name=slot_name,
            config=config,
            region=city_geometry,
            projection=projection,
            scale=scale,
            tile_scale=config["processing"]["tile_scale"]
        )
        output = output_by_month[diagnostic_month]
        for diagnostic_key, output_suffix in (
            ("n8day_mean", "n8day"),
            ("clear_days_mean", "clear_days")
        ):
            output_field = f"{slot_name}_{output_suffix}_mean"
            output_value = output.get(output_field)
            diagnostic_value = diagnostic[diagnostic_key]
            if (
                output_value is None
                or abs(output_value - diagnostic_value) > tolerance
            ):
                raise AssertionError(
                    f"{output_field} for {year}-{diagnostic_month:02d} "
                    "does not match per-composite production QA validity: "
                    f"monthly={output_value}, diagnostic={diagnostic_value}"
                )

    print(f"[PASS] One city selected: {city_id}")
    print(f"[PASS] Monthly records: {len(results)}")
    print(f"[PASS] LST fields: {len(LST_OUTPUT_FIELDS)}")
    print(f"[PASS] QC fields: {len(QC_OUTPUT_FIELDS)}")
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
