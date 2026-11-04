import math

import ee


def validate_max_pixels(max_pixels: int) -> None:
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError("samplepoint_attributes.max_pixels_per_cell must be positive.")


def build_modis_cell_regions(
    points: ee.FeatureCollection,
    projection: ee.Projection,
    scale: float
) -> ee.FeatureCollection:
    """Replace each MODIS center point with its square pixel footprint."""
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("MODIS pixel scale must be a positive finite number.")

    def to_cell(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        cell = (
            feature.geometry()
            .buffer(scale / 2, proj=projection)
            .bounds(proj=projection)
        )
        return feature.setGeometry(cell).set(
            "_modis_area_m2",
            cell.area(proj=projection)
        )

    return points.map(to_cell)


def reduce_over_modis_cells(
    image: ee.Image,
    regions: ee.FeatureCollection,
    reducer: ee.Reducer,
    max_pixels: int,
    tile_scale: int
) -> ee.FeatureCollection:
    """Reduce regions on the image's native grid, without rescaling its projection."""
    validate_max_pixels(max_pixels)
    return image.reduceRegions(
        collection=regions,
        reducer=reducer,
        tileScale=tile_scale,
        maxPixelsPerRegion=max_pixels
    )
