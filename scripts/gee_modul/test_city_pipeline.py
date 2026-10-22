import ee

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

    city_config = config["cities"]
    city_id = config["test"]["city_id"]
    year = config["test"]["year"]
    id_field = city_config["id_field"]
    tile_scale = config["processing"]["tile_scale"]
    qa_config = config["modis"]["qa"]

    city = select_single_city(
        ee.FeatureCollection(city_config["asset"]),
        id_field,
        city_id
    )
    projection, scale = get_modis_reference_grid(config)
    print(f"[CHECK] City ID: {city_id}; test year: {year}")
    print(f"[CHECK] QA config: {qa_config}")

    monthly_stats = build_yearly_lst_statistics(
        year=year,
        regions=city,
        config=config,
        projection=projection,
        scale=scale,
        tile_scale=tile_scale,
        id_field=id_field
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
        region=ee.Feature(city.first()).geometry(),
        projection=projection,
        scale=scale,
        tile_scale=tile_scale
    )

    print(f"[PASS] City ID: {city_id}; monthly records: {len(properties)}")
    print(f"[PASS] LST fields: {len(LST_OUTPUT_FIELDS)}")
    print(f"[PASS] QC fields: {len(QC_OUTPUT_FIELDS)}")
    print_monthly_results(properties)


if __name__ == "__main__":
    main()
