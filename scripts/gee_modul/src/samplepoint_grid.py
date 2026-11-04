import ee


def validate_max_pixels(max_pixels: int) -> None:
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError("samplepoint_attributes.max_pixels_per_cell must be positive.")


def mean_fullcell_zero_mask(image: ee.Image, max_pixels: int) -> ee.Image:
    """Area-weighted full-cell mean with masked source pixels contributing zero."""
    validate_max_pixels(max_pixels)
    return image.unmask(0, sameFootprint=False).reduceResolution(
        reducer=ee.Reducer.mean(),
        bestEffort=False,
        maxPixels=max_pixels
    )
