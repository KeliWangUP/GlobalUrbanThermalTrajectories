from typing import Any

import ee

from src.export_utils import (
    LST_BANDS,
    LST_OUTPUT_FIELDS,
    QC_OUTPUT_FIELDS
)
from src.modis_lst_utils import (
    compute_a2_validity,
    get_monthly_a2_collection
)

DIAGNOSTIC_CASES = [
    (5, "aqua_night"),
    (7, "aqua_night"),
    (7, "aqua_day")
]


def _number(stats: dict, key: str) -> float:
    return stats.get(key) or 0.0


def select_single_city(
    cities: ee.FeatureCollection,
    id_field: str,
    city_id: Any
) -> ee.FeatureCollection:
    """Select exactly one city and fail when it is missing or duplicated."""

    selected = cities.filter(ee.Filter.eq(id_field, city_id))
    count = selected.size().getInfo()
    if count != 1:
        raise ValueError(
            f"Expected exactly one feature with {id_field}={city_id}, "
            f"found {count}."
        )
    return selected


def diagnose_a2_image(
    image: ee.Image,
    region: ee.Geometry,
    slot_config: dict[str, Any],
    qa_config: dict[str, Any],
    projection: ee.Projection,
    scale: float,
    tile_scale: int,
    spatial_mask: ee.Image | None = None
) -> dict:
    """
    Diagnose observation attrition for one MODIS A2 composite.

    Validity comes from the production compute_a2_validity(). Every metric
    uses the same denominator: all pixels inside the region, restricted to
    spatial_mask when one is given.
    """

    validity = compute_a2_validity(
        image=image,
        lst_band=slot_config["lst_band"],
        qc_band=slot_config["qc_band"],
        clear_band=slot_config["clear_band"],
        qa_cfg=qa_config
    )

    # Informational raw QC-bit pass rates; not used for validity.
    qc_band = slot_config["qc_band"]
    qc_filled = image.select(qc_band).unmask(255)
    mandatory_qa = qc_filled.bitwiseAnd(3)
    data_quality = qc_filled.rightShift(2).bitwiseAnd(3)
    lst_error = qc_filled.rightShift(6).bitwiseAnd(3)

    diagnostic_image = ee.Image.cat([
        validity.select([
            "lst_available",
            "has_clear_obs",
            "qa_pass",
            "final_valid",
            "final_clear_count"
        ]),
        validity.select("clear_count").rename("raw_clear_count"),
        image.select(qc_band).mask().unmask(0).gt(0).rename("qc_available"),
        mandatory_qa.eq(0).rename("mandatory_qa_0"),
        mandatory_qa.lte(1).rename("mandatory_qa_le1"),
        data_quality.eq(0).rename("good_data_quality"),
        lst_error.lte(1).rename("lst_error_le1")
    ])
    if spatial_mask is not None:
        diagnostic_image = diagnostic_image.updateMask(spatial_mask)

    return diagnostic_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        crs=projection,
        scale=scale,
        maxPixels=10_000_000,
        tileScale=tile_scale
    ).getInfo()


def diagnose_a2_month(
    year: int,
    month: int,
    slot_name: str,
    config: dict[str, Any],
    region: ee.Geometry,
    projection: ee.Projection,
    scale: float,
    tile_scale: int,
    spatial_mask: ee.Image | None = None
) -> dict[str, float]:
    """
    Diagnose all A2 composites assigned to one month and slot.

    The monthly assignment is the production midpoint assignment. The
    returned totals are comparable with *_n8day_mean and *_clear_days_mean.
    """

    slot_config = config["modis"]["slots"][slot_name]
    collection = get_monthly_a2_collection(
        year=year,
        month=month,
        collection_id=slot_config["collection"]
    ).sort("system:time_start")

    count = collection.size().getInfo()
    print(f"\n[DIAG] {slot_name} {year}-{month:02d}: {count} A2 composites")

    image_list = collection.toList(count)
    total_final_valid = 0.0
    total_final_clear = 0.0

    for index in range(count):
        image = ee.Image(image_list.get(index))
        stats = diagnose_a2_image(
            image=image,
            region=region,
            slot_config=slot_config,
            qa_config=config["modis"]["qa"],
            projection=projection,
            scale=scale,
            tile_scale=tile_scale,
            spatial_mask=spatial_mask
        )
        final_valid = _number(stats, "final_valid")
        final_clear = _number(stats, "final_clear_count")
        total_final_valid += final_valid
        total_final_clear += final_clear

        start_date = (
            ee.Date(image.get("system:time_start"))
            .format("YYYY-MM-dd")
            .getInfo()
        )
        print(
            f"[DIAG] {start_date} | "
            f"LST={_number(stats, 'lst_available'):.3f} | "
            f"clear={_number(stats, 'has_clear_obs'):.3f} | "
            f"QA0={_number(stats, 'mandatory_qa_0'):.3f} | "
            f"QA<=1={_number(stats, 'mandatory_qa_le1'):.3f} | "
            f"dataQ0={_number(stats, 'good_data_quality'):.3f} | "
            f"err<=2K={_number(stats, 'lst_error_le1'):.3f} | "
            f"qaPass={_number(stats, 'qa_pass'):.3f} | "
            f"final={final_valid:.3f} | "
            f"clear_raw={_number(stats, 'raw_clear_count'):.2f} | "
            f"clear_final={final_clear:.2f} | "
            f"QC={_number(stats, 'qc_available'):.3f}"
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


def validate_monthly_records(
    properties: list[dict[str, Any]],
    year: int,
    id_field: str,
    expected_id: Any
) -> None:
    """Validate the schema and basic content of one city-year output."""

    if len(properties) != 12:
        raise AssertionError(
            f"Expected 12 monthly records, found {len(properties)}."
        )

    actual_months = sorted(row.get("month") for row in properties)
    if actual_months != list(range(1, 13)):
        raise AssertionError(
            f"Expected months 1-12, got {actual_months}."
        )
    if any(row.get("year") != year for row in properties):
        raise AssertionError(f"Output contains records outside year {year}.")
    if any(row.get(id_field) != expected_id for row in properties):
        raise AssertionError(
            f"Output contains a city ID other than {expected_id}."
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

    if not any(
        row.get(f"{band}_mean") is not None
        for row in properties
        for band in LST_BANDS
    ):
        raise AssertionError(
            "All monthly mean LST values are null for this city and year."
        )

    negative_qc_values = [
        (row["month"], field, row[field])
        for row in properties
        for field in QC_OUTPUT_FIELDS
        if row.get(field) is not None and row[field] < 0
    ]
    if negative_qc_values:
        raise AssertionError(
            f"QC summaries contain negative values: {negative_qc_values}"
        )
    if not any(
        row.get(field) is not None and row[field] > 0
        for row in properties
        for field in QC_OUTPUT_FIELDS
    ):
        raise AssertionError(
            "No positive QC summary was produced for this city and year."
        )


def check_outputs_against_diagnostics(
    properties: list[dict[str, Any]],
    year: int,
    config: dict[str, Any],
    region: ee.Geometry,
    projection: ee.Projection,
    scale: float,
    tile_scale: int,
    spatial_mask: ee.Image | None = None,
    tolerance: float = 1e-5
) -> None:
    """Require monthly outputs to equal per-composite production validity."""

    output_by_month = {row["month"]: row for row in properties}
    for month, slot_name in DIAGNOSTIC_CASES:
        diagnostic = diagnose_a2_month(
            year=year,
            month=month,
            slot_name=slot_name,
            config=config,
            region=region,
            projection=projection,
            scale=scale,
            tile_scale=tile_scale,
            spatial_mask=spatial_mask
        )
        for diagnostic_key, output_suffix in (
            ("n8day_mean", "n8day"),
            ("clear_days_mean", "clear_days")
        ):
            output_field = f"{slot_name}_{output_suffix}_mean"
            output_value = output_by_month[month].get(output_field)
            diagnostic_value = diagnostic[diagnostic_key]
            if (
                output_value is None
                or abs(output_value - diagnostic_value) > tolerance
            ):
                raise AssertionError(
                    f"{output_field} for {year}-{month:02d} does not match "
                    "per-composite production QA validity: "
                    f"monthly={output_value}, diagnostic={diagnostic_value}"
                )


def print_monthly_results(properties: list[dict[str, Any]]) -> None:
    """Print the monthly LST means and QC summaries."""

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
