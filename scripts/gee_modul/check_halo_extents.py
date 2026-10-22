import ee

from src.batch_utils import (
    load_city_batch,
    resolve_batch_indices,
    verify_city_batches
)
from src.export_utils import (
    get_modis_reference_grid,
    initialize_gee,
    load_config
)
from export_rural_monthly_lst import build_rural_halos


def add_halo_extent(
    feature: ee.Feature,
    crs: str,
    scale: float,
    max_error_m: int
) -> ee.Feature:
    """Add the MODIS-pixel width and height of the halo bounding box."""

    bounds = (
        feature.geometry()
        .transform(crs, max_error_m)
        .bounds(max_error_m)
    )
    ring = ee.List(ee.List(bounds.coordinates()).get(0))
    xs = ring.map(lambda point: ee.List(point).get(0))
    ys = ring.map(lambda point: ee.List(point).get(1))

    def _span(values: ee.List) -> ee.Number:
        return (
            ee.Number(values.reduce(ee.Reducer.max()))
            .subtract(ee.Number(values.reduce(ee.Reducer.min())))
            .divide(scale)
        )

    return feature.set({
        "width_px": _span(xs),
        "height_px": _span(ys)
    })


def main() -> None:
    config = load_config()
    initialize_gee(config)

    id_field = config["cities"]["id_field"]
    max_error_m = config["rural"]["max_error_m"]
    limit = config["test"]["max_halo_extent_pixels"]
    projection, scale = get_modis_reference_grid(config)
    crs = projection.getInfo()["crs"]

    batch_indices = resolve_batch_indices(config)
    verify_city_batches(config, batch_indices)

    flagged_total = 0
    for batch_index in batch_indices:
        halos = build_rural_halos(
            config,
            load_city_batch(config, batch_index)
        ).map(
            lambda feature: add_halo_extent(
                feature,
                crs,
                scale,
                max_error_m
            )
        )
        flagged = (
            halos
            .filter(ee.Filter.Or(
                ee.Filter.gt("width_px", limit),
                ee.Filter.gt("height_px", limit)
            ))
            .select([id_field, "width_px", "height_px"], None, False)
            .getInfo()["features"]
        )
        print(
            f"[BATCH {batch_index:02d}] halos with a bounding box above "
            f"{limit} MODIS pixels: {len(flagged)}"
        )
        for feature in flagged:
            properties = feature["properties"]
            print(
                f"[FLAG] {id_field}={properties[id_field]} "
                f"width_px={properties['width_px']:.0f} "
                f"height_px={properties['height_px']:.0f}"
            )
        flagged_total += len(flagged)

    print(f"[DONE] Flagged halos in {len(batch_indices)} batches: "
          f"{flagged_total}")


if __name__ == "__main__":
    main()
