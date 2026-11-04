import ee


def validate_max_pixels(max_pixels: int) -> None:
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError("samplepoint_attributes.max_pixels_per_cell must be positive.")


def sample_modis_pixel_area(
    points: ee.FeatureCollection,
    properties: list[str],
    projection: ee.Projection,
    scale: float,
    tile_scale: int
) -> ee.FeatureCollection:
    """Sample pixelArea directly on the existing MODIS reference grid."""
    return ee.Image.pixelArea().rename("modis_area_m2").sampleRegions(
        collection=points,
        properties=properties,
        projection=projection,
        scale=scale,
        tileScale=tile_scale,
        geometries=False
    )
