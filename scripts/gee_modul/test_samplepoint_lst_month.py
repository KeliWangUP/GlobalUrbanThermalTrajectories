import ee

from src.export_utils import (
    LST_BANDS,
    build_monthly_samplepoint_lst_statistics,
    build_samplepoint_lst_selectors,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)


def main() -> None:
    config = load_config()
    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)

    month_config = config["samplepoint_lst_export"]
    year = month_config["year"]
    month = month_config["month"]
    id_field = config["cities"]["id_field"]
    city_id = config["test"]["city_id"]
    points = (
        ee.FeatureCollection(config["export"]["asset_root"])
        .filter(ee.Filter.eq(id_field, city_id))
        .limit(1)
    )
    if points.size().getInfo() == 0:
        raise ValueError(f"No sample points found for city {city_id}.")

    monthly = build_monthly_samplepoint_lst_statistics(
        year=year,
        month=month,
        points=points,
        config=config,
        projection=projection,
        scale=scale,
        id_field=id_field
    )
    features = monthly.getInfo()["features"]
    if len(features) != 1:
        raise ValueError(
            f"Expected one sample record, got {len(features)}."
        )

    properties = features[0]["properties"]
    required_fields = set(build_samplepoint_lst_selectors(id_field))
    if not required_fields.issubset(properties):
        raise ValueError("The monthly output is missing expected fields.")
    if (
        properties.get(id_field) != city_id
        or properties.get("year") != year
        or properties.get("month") != month
    ):
        raise ValueError(
            "The monthly output contains an unexpected city or date."
        )
    if not any(
        properties.get(f"{band}_mean") is not None
        for band in LST_BANDS
    ):
        raise ValueError("The sample point has no valid LST for this month.")

    values = {
        band: properties.get(f"{band}_mean")
        for band in LST_BANDS
    }
    print(
        f"[PASS] City {city_id}; pixel {properties['pixel_id']}; "
        f"month {year}-{month:02d}: {values}"
    )


if __name__ == "__main__":
    main()
