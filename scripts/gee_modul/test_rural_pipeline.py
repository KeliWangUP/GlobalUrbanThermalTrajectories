import ee

from src.modis_lst_utils import (
    add_composite_midpoint,
    count_clear_sky_bits
)

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

def diagnose_a2_image(
    image: ee.Image,
    region: ee.Geometry,
    rural_mask: ee.Image,
    lst_band: str,
    qc_band: str,
    clear_band: str,
    qa_config: dict,
    projection: ee.Projection,
    scale: float,
    tile_scale: int
) -> dict:
    """
    Diagnose observation attrition for one MODIS A2 composite.

    All fractions use the same denominator:
    all eligible rural MODIS pixels inside the rural halo.

    Returns
    -------
    dict
        Client-side dictionary containing diagnostic statistics.
    """

    lst = image.select(lst_band)
    qc = image.select(qc_band)

    # --------------------------------------------------
    # LST availability
    # --------------------------------------------------

    lst_available = (
        lst
        .mask()
        .unmask(0)
        .gt(0)
        .rename("lst_available")
    )

    # --------------------------------------------------
    # Clear-sky availability
    # --------------------------------------------------

    clear_count = (
        count_clear_sky_bits(
            image=image,
            clear_band=clear_band
        )
        .unmask(0)
        .toFloat()
    )

    has_clear_obs = (
        clear_count
        .gt(0)
        .rename("has_clear_obs")
    )

    # --------------------------------------------------
    # QA bits
    #
    # Use 255 for masked QC values:
    # bits 0-1 -> 3
    # bits 6-7 -> 3
    # therefore masked QC fails quality tests.
    # --------------------------------------------------
    qc_available = (
        qc
        .mask()
        .unmask(0)
        .gt(0)
        .rename("qc_available")
    )
    
    qc_filled = qc.unmask(255)

    mandatory_qa = (
        qc_filled
        .bitwiseAnd(3)
    )

    mandatory_qa_0 = (
        mandatory_qa
        .eq(0)
        .rename("mandatory_qa_0")
    )

    mandatory_qa_le1 = (
        mandatory_qa
        .lte(1)
        .rename("mandatory_qa_le1")
    )

    data_quality = (
        qc_filled
        .rightShift(2)
        .bitwiseAnd(3)
    )

    good_data_quality = (
        data_quality
        .eq(0)
        .rename("good_data_quality")
    )

    lst_error = (
        qc_filled
        .rightShift(6)
        .bitwiseAnd(3)
    )

    lst_error_le1 = (
        lst_error
        .lte(1)
        .rename("lst_error_le1")
    )

    # --------------------------------------------------
    # Build QA mask using CURRENT config
    # --------------------------------------------------

    current_qa_mask = mandatory_qa.lte(
        qa_config["mandatory_qa_max"]
    )

    max_error_code = qa_config.get(
        "max_lst_error_code"
    )

    if max_error_code is not None:
        current_qa_mask = current_qa_mask.And(
            lst_error.lte(max_error_code)
        )

    if qa_config.get(
        "require_good_data_quality",
        False
    ):
        current_qa_mask = current_qa_mask.And(
            data_quality.eq(0)
        )

    current_qa_mask = current_qa_mask.rename(
        "current_qa_pass"
    )

    # --------------------------------------------------
    # Final validity under current production settings
    # --------------------------------------------------

    final_valid = (
        lst_available
        .And(has_clear_obs)
        .And(current_qa_mask)
        .rename("final_valid")
    )

    # Clear-sky observations that actually survive
    # all production filters.
    final_clear_count = (
        clear_count
        .multiply(final_valid)
        .rename("final_clear_count")
    )

    # --------------------------------------------------
    # Diagnostic image
    #
    # All diagnostic bands are unmasked before applying
    # the same static rural mask. Therefore the spatial
    # denominator is identical for every metric.
    # --------------------------------------------------

    diagnostic_image = (
        ee.Image.cat([
            lst_available,
            has_clear_obs,
            qc_available,
            mandatory_qa_0,
            mandatory_qa_le1,
            good_data_quality,
            lst_error_le1,
            current_qa_mask,
            final_valid,
            clear_count.rename("raw_clear_count"),
            final_clear_count
        ])
        .updateMask(rural_mask)
    )

    stats = diagnostic_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        crs=projection,
        scale=scale,
        maxPixels=10_000_000,
        tileScale=tile_scale
    )

    return stats.getInfo()


def diagnose_a2_month(
    year: int,
    month: int,
    slot_name: str,
    config: dict,
    region: ee.Geometry,
    rural_mask: ee.Image,
    projection: ee.Projection,
    scale: float,
    tile_scale: int,
    rural_pixel_count: int
) -> dict[str, float]:
    """
    Diagnose all A2 composites assigned to one month and slot.

    The temporal assignment exactly follows the production
    midpoint-based monthly aggregation.

    Returns
    -------
    dict
        Monthly diagnostic totals that can be compared with
        *_n8day_mean and *_clear_days_mean.
    """

    slot_config = config["modis"]["slots"][
        slot_name
    ]

    qa_config = config["modis"]["qa"]

    month_start = ee.Date.fromYMD(
        year,
        month,
        1
    )

    month_end = month_start.advance(
        1,
        "month"
    )

    # Must match build_monthly_a2_slot().
    source_start = month_start.advance(
        -4,
        "day"
    )

    source_end = month_end.advance(
        -4,
        "day"
    )

    collection = (
        ee.ImageCollection(
            slot_config["collection"]
        )
        .filterDate(
            source_start,
            source_end
        )
        .map(add_composite_midpoint)
        .filter(
            ee.Filter.gte(
                "composite_mid_time",
                month_start.millis()
            )
        )
        .filter(
            ee.Filter.lt(
                "composite_mid_time",
                month_end.millis()
            )
        )
        .sort("system:time_start")
    )

    n_images = collection.size().getInfo()

    print()
    print(
        f"[DIAG] {slot_name} "
        f"{year}-{month:02d}"
    )
    print(
        f"[DIAG] A2 composites: {n_images}; "
        f"eligible rural pixels: {rural_pixel_count}"
    )

    image_list = collection.toList(
        n_images
    )

    total_final_valid = 0.0
    total_final_clear = 0.0

    for index in range(n_images):

        image = ee.Image(
            image_list.get(index)
        )

        start_date = (
            ee.Date(
                image.get("system:time_start")
            )
            .format("YYYY-MM-dd")
            .getInfo()
        )

        stats = diagnose_a2_image(
            image=image,
            region=region,
            rural_mask=rural_mask,
            lst_band=slot_config["lst_band"],
            qc_band=slot_config["qc_band"],
            clear_band=slot_config["clear_band"],
            qa_config=qa_config,
            projection=projection,
            scale=scale,
            tile_scale=tile_scale
        )

        final_valid = (
            stats.get("final_valid")
            or 0.0
        )

        final_clear = (
            stats.get("final_clear_count")
            or 0.0
        )

        total_final_valid += final_valid
        total_final_clear += final_clear

        print(
            f"[DIAG] {start_date} | "
            f"LST={stats.get('lst_available', 0):.3f} | "
            f"clear={stats.get('has_clear_obs', 0):.3f} | "
            f"QA0={stats.get('mandatory_qa_0', 0):.3f} | "
            f"QA<=1={stats.get('mandatory_qa_le1', 0):.3f} | "
            f"dataQ0={stats.get('good_data_quality', 0):.3f} | "
            f"err<=2K={stats.get('lst_error_le1', 0):.3f} | "
            f"currentQA={stats.get('current_qa_pass', 0):.3f} | "
            f"final={final_valid:.3f} | "
            f"clear_raw={stats.get('raw_clear_count', 0):.2f} | "
            f"clear_final={final_clear:.2f}"
            f"QC={stats.get('qc_available', 0):.3f} | "
        )

    print(
        f"[DIAG-SUM] {slot_name} "
        f"{year}-{month:02d} | "
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

        # --------------------------------------------------
    # A2 observation diagnostics
    # --------------------------------------------------

    diagnostic_cases = [
        (5, "aqua_night"),
        (7, "aqua_night"),
        (7, "aqua_day")
    ]

    for diagnostic_month, slot_name in diagnostic_cases:

        diagnose_a2_month(
            year=year,
            month=diagnostic_month,
            slot_name=slot_name,
            config=config,
            region=halo_geometry,
            rural_mask=rural_mask,
            projection=projection,
            scale=scale,
            tile_scale=tile_scale,
            rural_pixel_count=rural_pixel_count
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
