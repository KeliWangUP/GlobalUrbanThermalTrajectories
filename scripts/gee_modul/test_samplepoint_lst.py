import ee

from src.export_utils import (
    LST_BANDS,
    build_samplepoint_lst_selectors,
    build_yearly_samplepoint_lst_statistics,
    get_modis_reference_grid,
    initialize_gee,
    load_config
)


def main() -> None:
    config = load_config()
    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)

    id_field = config["cities"]["id_field"]
    city_id = config["test"]["city_id"]
    points = (
        ee.FeatureCollection(config["export"]["asset_root"])
        .filter(ee.Filter.eq(id_field, city_id))
        .limit(1)
    )
    if points.size().getInfo() == 0:
        raise ValueError(
            f"No sample points found for city {city_id}."
        )

    year = config["test"]["year"]
    monthly = build_yearly_samplepoint_lst_statistics(
        year=year,
        points=points,
        config=config,
        projection=projection,
        scale=scale,
        id_field=id_field
    )
    features = monthly.getInfo()["features"]
    if len(features) != 12:
        raise ValueError(
            f"Expected 12 monthly sample records, got {len(features)}."
        )

    properties = [feature["properties"] for feature in features]
    required_fields = set(build_samplepoint_lst_selectors(id_field))
    for row in properties:
        if not required_fields.issubset(row):
            raise ValueError(
                "A sample-point LST record is missing expected output fields."
            )
        if row.get(id_field) != city_id or row.get("year") != year:
            raise ValueError(
                "A sample-point LST record has an unexpected city or year."
            )
    if sorted(row["month"] for row in properties) != list(range(1, 13)):
        raise ValueError("Sample-point output does not contain months 1-12.")
    if not any(
        row.get(f"{band}_mean") is not None
        for row in properties
        for band in LST_BANDS
    ):
        raise ValueError(
            "The test point has no valid monthly LST observations."
        )

    print(
        f"[PASS] City {city_id}; pixel "
        f"{properties[0]['pixel_id']}; monthly records: {len(properties)}"
    )
    for row in properties:
        values = {
            band: row.get(f"{band}_mean")
            for band in LST_BANDS
        }
        print(f"[RESULT] {year}-{row['month']:02d}: {values}")


if __name__ == "__main__":
    main()
