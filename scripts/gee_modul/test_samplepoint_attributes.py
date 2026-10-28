import ee

from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from src.samplepoint_attributes import (
    build_samplepoint_attribute_collection,
    prepare_samplepoint_sources
)


def main() -> None:
    config = load_config()
    initialize_gee(config)
    projection, scale = get_modis_reference_grid(config)

    point_asset = config["export"]["asset_root"]
    id_field = config["cities"]["id_field"]
    city_id = config["test"]["city_id"]
    city_points = ee.FeatureCollection(point_asset).filter(
        ee.Filter.eq(id_field, city_id)
    )
    city_point_count = city_points.size().getInfo()
    if city_point_count == 0:
        raise ValueError(
            f"No sample points for city {city_id} in {point_asset}."
        )

    sources = prepare_samplepoint_sources(config)
    sample = build_samplepoint_attribute_collection(
        points=city_points.limit(1),
        config=config,
        projection=projection,
        modis_scale=scale,
        sources=sources
    )
    result = ee.Feature(sample.first()).toDictionary().getInfo()
    isa_fractions = {
        year: result.get(f"isa_urban_fraction_{year}")
        for year in sources.isa_years
    }
    if any(fraction is None for fraction in isa_fractions.values()):
        raise ValueError(
            "The test sample point is missing one or more annual ISA fractions."
        )
    if any(
        not 0 <= fraction <= 1
        for fraction in isa_fractions.values()
    ):
        raise ValueError(
            "An annual ISA urban fraction is outside the [0, 1] range."
        )
    fractions_by_year = [
        isa_fractions[year]
        for year in sources.isa_years
    ]
    if any(
        earlier > later
        for earlier, later in zip(
            fractions_by_year,
            fractions_by_year[1:]
        )
    ):
        raise ValueError(
            "Annual ISA urban fractions must not decrease over time."
        )

    water_fraction = result.get("water_fraction")
    if water_fraction is not None and not 0 <= water_fraction <= 1:
        raise ValueError(
            f"Water fraction is outside [0, 1]: {water_fraction}"
        )

    ghsl_years = config["ghsl_built"]["years"]
    valid_ghsl_counts = [
        result.get(f"built_surface_{year}_m2_count", 0)
        for year in ghsl_years
    ] + [
        result.get(f"built_volume_{year}_m3_count", 0)
        for year in ghsl_years
    ]
    if not any(count and count > 0 for count in valid_ghsl_counts):
        raise ValueError(
            "No valid GHSL pixels were returned for any configured year."
        )

    print(f"[PASS] City {city_id}; sample points available: {city_point_count}")
    print(f"[PASS] Sample pixel: {result.get('pixel_id')}")
    print(f"[PASS] ISA annual urban fractions: {isa_fractions}")
    print(f"[PASS] Water fraction: {water_fraction}")
    for year in ghsl_years:
        print(
            f"[GHSL] {year}: surface mean="
            f"{result.get(f'built_surface_{year}_m2_mean')}; volume mean="
            f"{result.get(f'built_volume_{year}_m3_mean')}; "
            f"valid pixels="
            f"{result.get(f'built_surface_{year}_m2_count')}/"
            f"{result.get(f'built_volume_{year}_m3_count')}"
        )


if __name__ == "__main__":
    main()
