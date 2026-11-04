import math
from typing import Any

import ee

from src.samplepoint_attributes import prepare_isa_source
from src.samplepoint_isa import build_fullcell_isa_image, sample_fullcell_isa
from src.samplepoint_utils import load_sample_points, samplepoint_base_fields


SYNTHETIC_CRS = "EPSG:3857"


def _validate_fractions(
    properties: dict[str, Any],
    years: tuple[int, ...],
    location: str
) -> None:
    fractions = []
    for year in years:
        field = f"isa_urban_fraction_{year}"
        value = properties.get(field)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{location}: missing or non-finite {field}: {value!r}.")
        if not -1e-6 <= value <= 1 + 1e-6:
            raise ValueError(f"{location}: {field} is outside [0, 1]: {value}.")
        fractions.append(value)
    if any(first > second + 1e-6 for first, second in zip(fractions, fractions[1:])):
        raise ValueError(f"{location}: annual ISA fractions decrease.")


def synthetic_fullcell_grid(
    config: dict[str, Any]
) -> tuple[ee.Projection, ee.Image, ee.Image, ee.FeatureCollection]:
    """Create one aligned 60 m cell containing four 30 m source pixels."""
    native = ee.Projection(SYNTHETIC_CRS, [30, 0, 0, 0, -30, 0])
    target = ee.Projection(SYNTHETIC_CRS, [60, 0, 0, 0, -60, 0])
    coords = ee.Image.pixelCoordinates(native)
    x = coords.select("x").mod(2)
    y = coords.select("y").mod(2)
    fields = {
        "pixel_id": "SYNTHETIC",
        config["cities"]["id_field"]: 0,
        "modis_x": 0,
        "modis_y": 0
    }
    if config["sample_points"]["include_lonlat"]:
        fields.update({"lon": 0, "lat": 0})
    point = ee.Feature(
        ee.Geometry.Point([30, -30], proj=ee.Projection(SYNTHETIC_CRS)),
        fields
    )
    points = ee.FeatureCollection([point])
    return target, x, y, points


def synthetic_isa_classes(x: ee.Image, y: ee.Image) -> ee.Image:
    return (
        x.multiply(0).add(38)
        .where(x.eq(1).And(y.eq(0)), 3)
        .where(x.eq(0).And(y.eq(1)), 2)
        .updateMask(x.neq(1).Or(y.neq(1)))
        .rename("isa")
    )


def check_synthetic_fullcell(config: dict[str, Any]) -> None:
    """Check masked-zero and fractional-overlap behavior on a tiny GEE grid."""
    print(f"[INFO] Synthetic full-cell checks use {SYNTHETIC_CRS}.")
    target, x, y, points = synthetic_fullcell_grid(config)
    classes = synthetic_isa_classes(x, y)
    years = (1985, 2020, 2021)
    for label, image, expected in (
        ("mixed", classes, (0.25, 0.5, 0.75)),
        ("all_masked", classes.updateMask(ee.Image.constant(0)), (0, 0, 0))
    ):
        fractions = build_fullcell_isa_image(image, years, max_pixels=16)
        sample = sample_fullcell_isa(
            fractions, points, config, target, 60, years
        ).getInfo()
        features = sample.get("features", [])
        if len(features) != 1:
            raise ValueError(f"Synthetic {label}: expected one sample, got {len(features)}.")
        values = features[0]["properties"]
        _validate_fractions(values, years, f"Synthetic {label}")
        for year, target_value in zip(years, expected):
            actual = values[f"isa_urban_fraction_{year}"]
            if not math.isclose(actual, target_value, abs_tol=1e-6):
                raise ValueError(
                    f"Synthetic {label} {year}: expected {target_value}, got {actual}."
                )
    print("[PASS] Synthetic full-cell test: area weights, masked zero and annual thresholds.")


def check_samplepoint_isa_city(
    config: dict[str, Any],
    projection: ee.Projection,
    scale: float
) -> None:
    settings = config["samplepoint_isa_export"]
    limit = settings["test_point_limit"]
    pixel_id = settings["test_pixel_id"]
    if type(limit) is not int or limit < 1:
        raise ValueError("samplepoint_isa_export.test_point_limit must be positive.")
    if pixel_id is not None and (not isinstance(pixel_id, str) or not pixel_id.strip()):
        raise ValueError("samplepoint_isa_export.test_pixel_id must be null or a non-empty string.")

    check_synthetic_fullcell(config)
    id_field = config["cities"]["id_field"]
    city_id = config["test"]["city_id"]
    points, _ = load_sample_points(config, samplepoint_base_fields(config))
    city_points = points.filter(ee.Filter.eq(id_field, city_id))
    city_count = city_points.size().getInfo()
    if not city_count:
        raise ValueError(f"No sample points for test city {city_id}.")
    selected = city_points
    if pixel_id is not None:
        selected = selected.filter(ee.Filter.eq("pixel_id", pixel_id))
    selected = selected.sort("pixel_id").limit(limit)
    expected_ids = selected.aggregate_array("pixel_id").getInfo()
    if not expected_ids:
        raise ValueError(f"No selected sample points for city {city_id}, pixel={pixel_id}.")

    image, band, years = prepare_isa_source(config)
    fractions = build_fullcell_isa_image(
        image, years, config["samplepoint_attributes"]["max_pixels_per_cell"]
    )
    info = sample_fullcell_isa(
        fractions, selected, config, projection, scale, years
    ).getInfo()
    features = info.get("features", [])
    received_ids = [feature["properties"].get("pixel_id") for feature in features]
    if (
        any(not isinstance(value, str) for value in expected_ids + received_ids)
        or len(received_ids) != len(expected_ids)
        or sorted(received_ids) != sorted(expected_ids)
    ):
        raise ValueError(
            f"Sample output did not preserve the selected point IDs: "
            f"expected={expected_ids}, received={received_ids}."
        )
    for feature in features:
        properties = feature["properties"]
        if properties.get(id_field) != city_id:
            raise ValueError(f"Unexpected city ID in sample output: {properties.get(id_field)}.")
        _validate_fractions(properties, years, f"Pixel {properties['pixel_id']}")
        summary = {
            year: properties[f"isa_urban_fraction_{year}"]
            for year in (1985, 2000, 2020, 2021)
        }
        print(f"[RESULT] pixel={properties['pixel_id']} ISA={summary}")
    print(
        f"[PASS] City {city_id}: {len(features)} sample points checked "
        f"out of {city_count}; ISA band={band}."
    )
