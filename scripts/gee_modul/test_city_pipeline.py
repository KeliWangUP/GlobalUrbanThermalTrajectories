import ee

from src.export_utils import (
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

    mean_fields = [
        key for key in properties[0]
        if key.endswith("_mean")
    ]
    if not mean_fields:
        raise AssertionError(
            "No mean statistic fields were produced by reduceRegions."
        )

    if not any(
        row.get(field) is not None
        for row in properties
        for field in mean_fields
    ):
        raise AssertionError(
            "All monthly mean LST values are null for this city and year."
        )

    print(f"[PASS] One city selected: {city_id}")
    print(f"[PASS] Monthly records: {len(results)}")
    print(f"[PASS] Mean statistic fields: {', '.join(mean_fields)}")
    for row in properties:
        values = ", ".join(
            f"{field}={row.get(field)}"
            for field in mean_fields
        )
        print(f"[RESULT] {row['year']}-{row['month']:02d}: {values}")


if __name__ == "__main__":
    main()
