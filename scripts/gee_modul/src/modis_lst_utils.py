import ee

def count_clear_sky_bits(
    image: ee.Image,
    clear_band: str
) -> ee.Image:
    """
    Count clear-sky days encoded in an 8-bit MODIS A2 bitmask.

    Parameters
    ----------
    image : ee.Image
        MOD11A2 or MYD11A2 image.

    clear_band : str
        Clear_sky_days or Clear_sky_nights.

    Returns
    -------
    ee.Image
        Number of clear-sky observations within the 8-day period.
    """

    bitmask = image.select(clear_band)

    count = ee.Image.constant(0)

    for bit in range(8):
        clear = (
            bitmask
            .rightShift(bit)
            .bitwiseAnd(1)
        )

        count = count.add(clear)

    return count.rename("clear_count")

def build_a2_qa_mask(
    image: ee.Image,
    qc_band: str,
    mandatory_qa_max: int = 1,
    max_lst_error_code: int = 1
) -> ee.Image:
    """
    Build a quality mask for MODIS A2 LST composites.
    """

    qc = image.select(qc_band)

    # Bits 0-1: mandatory QA
    mandatory_qa = qc.bitwiseAnd(3)

    # Bits 6-7: estimated LST error
    lst_error = (
        qc
        .rightShift(6)
        .bitwiseAnd(3)
    )

    mask = (
        mandatory_qa.lte(mandatory_qa_max)
        .And(
            lst_error.lte(max_lst_error_code)
        )
    )

    return mask

def preprocess_a2_slot(
    image: ee.Image,
    lst_band: str,
    qc_band: str,
    clear_band: str,
    slot_name: str,
    scale_factor: float = 0.02,
    kelvin_offset: float = -273.15,
    mandatory_qa_max: int = 1,
    max_lst_error_code: int = 1
) -> ee.Image:
    """
    Prepare one MODIS A2 observation slot.

    Output bands
    ------------
    <slot>_weighted
    <slot>_weight
    <slot>_ncomp
    """

    lst = (
        image
        .select(lst_band)
        .multiply(scale_factor)
        .add(kelvin_offset)
    )

    qa_mask = build_a2_qa_mask(
        image=image,
        qc_band=qc_band,
        mandatory_qa_max=mandatory_qa_max,
        max_lst_error_code=max_lst_error_code
    )

    clear_count = count_clear_sky_bits(
        image=image,
        clear_band=clear_band
    )

    valid = (
        qa_mask
        .And(clear_count.gt(0))
        .And(lst.mask().gt(0))
    )

    valid_numeric = valid.toByte()

    weight = (
        clear_count
        .multiply(valid_numeric)
        .rename(f"{slot_name}_weight")
    )

    weighted_lst = (
        lst
        .unmask(0)
        .multiply(weight)
        .rename(f"{slot_name}_weighted")
    )

    ncomp = (
        valid_numeric
        .rename(f"{slot_name}_ncomp")
    )

    return (
        weighted_lst
        .addBands(weight)
        .addBands(ncomp)
    )

def add_composite_midpoint(
    image: ee.Image
) -> ee.Image:
    """
    Add the approximate midpoint time of an 8-day composite.
    """

    start = ee.Date(
        image.get("system:time_start")
    )

    midpoint = start.advance(4, "day")

    return image.set(
        "composite_mid_time",
        midpoint.millis()
    )

def build_monthly_a2_slot(
    year: int,
    month: int,
    slot_name: str,
    slot_cfg: dict,
    modis_cfg: dict,
    reference_projection: ee.Projection
) -> ee.Image:
    """
    Build one clear-sky-weighted monthly MODIS A2 LST band.

    Composite assignment is based on the approximate
    midpoint of each 8-day period.

    Returns
    -------
    ee.Image
        Three bands:
        <slot>
        <slot>_clear_days
        <slot>_n8day
    """

    month_start = ee.Date.fromYMD(
        year, month, 1
    )

    month_end = month_start.advance(
        1, "month"
    )

    collection = (
        ee.ImageCollection(
            slot_cfg["collection"]
        )
        .map(add_composite_midpoint)
        .filter(
            ee.Filter.gte(
                "composite_mid_time",
                month_start.millis()
            )
        )
        .filter(
            ee.Filter.lt(
                "composite_mid_time",
                month_end.millis()
            )
        )
    )

    qa_cfg = modis_cfg["qa"]

    def _process(image):
        return preprocess_a2_slot(
            image=image,
            lst_band=slot_cfg["lst_band"],
            qc_band=slot_cfg["qc_band"],
            clear_band=slot_cfg["clear_band"],
            slot_name=slot_name,
            scale_factor=modis_cfg["scale_factor"],
            kelvin_offset=modis_cfg["kelvin_offset"],
            mandatory_qa_max=qa_cfg[
                "mandatory_qa_max"
            ],
            max_lst_error_code=qa_cfg[
                "max_lst_error_code"
            ]
        )

    processed = collection.map(_process)

    def _build_month():

        weighted_sum = (
            processed
            .select(f"{slot_name}_weighted")
            .sum()
        )

        clear_days = (
            processed
            .select(f"{slot_name}_weight")
            .sum()
            .rename(
                f"{slot_name}_clear_days"
            )
        )

        n8day = (
            processed
            .select(f"{slot_name}_ncomp")
            .sum()
            .rename(
                f"{slot_name}_n8day"
            )
        )

        monthly_lst = (
            weighted_sum
            .divide(clear_days)
            .updateMask(
                clear_days.gt(0)
            )
            .rename(slot_name)
        )

        return (
            monthly_lst
            .addBands(clear_days)
            .addBands(n8day)
        )

    # Needed for the early period before Aqua exists.
    empty = (
        ee.Image.constant([
            0, 0, 0
        ])
        .rename([
            slot_name,
            f"{slot_name}_clear_days",
            f"{slot_name}_n8day"
        ])
        .updateMask(
            ee.Image.constant(0)
        )
        .setDefaultProjection(
            reference_projection
        )
    )

    result = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            _build_month(),
            empty
        )
    )

    return result

def build_monthly_four_slot_image(
    year: int,
    month: int,
    config: dict,
    reference_projection: ee.Projection
) -> ee.Image:
    """
    Build the complete monthly MODIS A2 thermal image.
    """

    slot_images = []

    for slot_name, slot_cfg in (
        config["modis"]["slots"].items()
    ):

        slot_image = build_monthly_a2_slot(
            year=year,
            month=month,
            slot_name=slot_name,
            slot_cfg=slot_cfg,
            modis_cfg=config["modis"],
            reference_projection=reference_projection
        )

        slot_images.append(slot_image)

    monthly = slot_images[0]

    for image in slot_images[1:]:
        monthly = monthly.addBands(image)

    return monthly.set({
        "year": year,
        "month": month
    })

