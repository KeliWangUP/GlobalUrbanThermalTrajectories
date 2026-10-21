import ee


def count_clear_sky_bits(
    image: ee.Image,
    clear_band: str
) -> ee.Image:
    """
    Count clear-sky observations encoded in the MODIS A2 8-bit mask.

    Each bit corresponds to one day/night in the 8-day composite.

    Returns
    -------
    ee.Image
        Number of clear-sky observations, ranging from 0 to 8.
    """

    bitmask = image.select(clear_band)

    count = ee.Image.constant(0)

    for bit in range(8):
        count = count.add(
            bitmask
            .rightShift(bit)
            .bitwiseAnd(1)
        )

    return count.rename("clear_count")


def build_a2_qa_mask(
    image: ee.Image,
    qc_band: str,
    mandatory_qa_max: int = 1,
    max_lst_error_code: int = 1,
    require_good_data_quality: bool = False
) -> ee.Image:
    """
    Build a quality mask for MODIS A2 LST composites.

    Parameters
    ----------
    mandatory_qa_max : int
        Maximum accepted mandatory QA code.
        0 = good quality only.
        1 = also allow other/unquantifiable quality.

    max_lst_error_code : int
        Maximum accepted LST error code.
        0 = <= 1 K
        1 = <= 2 K
        2 = <= 3 K
        3 = > 3 K

    require_good_data_quality : bool
        If True, require bits 2-3 to equal 0.
    """

    qc = image.select(qc_band)

    # Bits 0-1: mandatory QA
    mandatory_qa = qc.bitwiseAnd(3)

    # Bits 2-3: detailed data quality
    data_quality = (
        qc
        .rightShift(2)
        .bitwiseAnd(3)
    )

    # Bits 6-7: estimated LST error
    lst_error = (
        qc
        .rightShift(6)
        .bitwiseAnd(3)
    )

    mask = (
        mandatory_qa
        .lte(mandatory_qa_max)
        .And(
            lst_error.lte(max_lst_error_code)
        )
    )

    if require_good_data_quality:
        mask = mask.And(
            data_quality.eq(0)
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
    max_lst_error_code: int = 1,
    require_good_data_quality: bool = False
) -> ee.Image:
    """
    Prepare one MODIS A2 observation slot.

    Output bands
    ------------
    <slot>_weighted
        LST multiplied by clear-sky observation count.

    <slot>_weight
        Number of clear-sky observations.

    <slot>_ncomp
        1 if this A2 composite is valid for the pixel, otherwise 0/masked.
    """

    lst = (
        image
        .select(lst_band)
        .multiply(scale_factor)
        .add(kelvin_offset)
        .toFloat()
    )

    qa_mask = build_a2_qa_mask(
        image=image,
        qc_band=qc_band,
        mandatory_qa_max=mandatory_qa_max,
        max_lst_error_code=max_lst_error_code,
        require_good_data_quality=require_good_data_quality
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
    Add the approximate temporal midpoint of an 8-day composite.
    """

    start = ee.Date(
        image.get("system:time_start")
    )

    midpoint = start.advance(4, "day")

    return image.set({
        "composite_mid_time": midpoint.millis()
    })


def create_empty_processed_slot(
    slot_name: str,
    reference_projection: ee.Projection
) -> ee.Image:
    """
    Create a fully masked placeholder for months without observations.

    Primarily needed for Aqua before its observation period.
    """

    empty = (
        ee.Image.constant([0, 0, 0])
        .rename([
            f"{slot_name}_weighted",
            f"{slot_name}_weight",
            f"{slot_name}_ncomp"
        ])
        .updateMask(
            ee.Image.constant(0)
        )
        .setDefaultProjection(
            reference_projection
        )
    )

    return empty


def build_monthly_a2_slot(
    year: int,
    month: int,
    slot_name: str,
    slot_cfg: dict,
    modis_cfg: dict,
    reference_projection: ee.Projection
) -> ee.Image:
    """
    Build one clear-sky-weighted monthly MODIS A2 LST product.

    The 8-day composites are assigned to calendar months according
    to their approximate temporal midpoint.

    Returns
    -------
    ee.Image
        Three bands:

        <slot>
            Monthly weighted mean LST.
            Masked where no valid observation exists.

        <slot>_clear_days
            Total number of valid clear-sky observations represented
            by A2 composites assigned to the month.

            Pixels with no valid observations are explicitly set to 0
            when the sensor/month exists.

        <slot>_n8day
            Number of valid A2 composites contributing at each pixel.

            Pixels with no valid composites are explicitly set to 0
            when the sensor/month exists.
    """

    month_start = ee.Date.fromYMD(
        year,
        month,
        1
    )

    month_end = month_start.advance(
        1,
        "month"
    )

    # Midpoint = start + 4 days.
    #
    # Therefore an A2 composite belongs to this month if its
    # start date falls approximately within:
    #
    # [month_start - 4 days, month_end - 4 days)
    source_start = month_start.advance(
        -4,
        "day"
    )

    source_end = month_end.advance(
        -4,
        "day"
    )

    collection = (
        ee.ImageCollection(
            slot_cfg["collection"]
        )
        .filterDate(
            source_start,
            source_end
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
            ],
            require_good_data_quality=qa_cfg.get(
                "require_good_data_quality",
                False
            )
        )

    processed = collection.map(
        _process
    )

    # --------------------------------------------------
    # Safe dummy for empty source collections
    # --------------------------------------------------

    empty_processed = create_empty_processed_slot(
        slot_name=slot_name,
        reference_projection=reference_projection
    )

    safe_processed = ee.ImageCollection(
        ee.Algorithms.If(
            processed.size().gt(0),
            processed,
            ee.ImageCollection([
                empty_processed
            ])
        )
    )

    # --------------------------------------------------
    # Monthly aggregation
    # --------------------------------------------------
    #
    # Important:
    #
    # QC bands are explicitly zero-filled after summation.
    # This ensures later spatial means use ALL analysis pixels
    # as the denominator, including pixels with zero valid
    # observations.
    # --------------------------------------------------

    weighted_sum = (
        safe_processed
        .select(
            f"{slot_name}_weighted"
        )
        .sum()
        .unmask(0)
    )

    clear_days = (
        safe_processed
        .select(
            f"{slot_name}_weight"
        )
        .sum()
        .unmask(0)
        .rename(
            f"{slot_name}_clear_days"
        )
    )

    n8day = (
        safe_processed
        .select(
            f"{slot_name}_ncomp"
        )
        .sum()
        .unmask(0)
        .rename(
            f"{slot_name}_n8day"
        )
    )

    # LST remains masked where there are no valid observations.
    monthly_lst = (
        weighted_sum
        .divide(clear_days)
        .updateMask(
            clear_days.gt(0)
        )
        .rename(slot_name)
    )

    monthly_result = (
        monthly_lst
        .addBands(clear_days)
        .addBands(n8day)
    )

    # --------------------------------------------------
    # Truly unavailable sensor/month
    #
    # Example:
    # Aqua in 2001 should remain null/masked rather than
    # being interpreted as zero observations from an
    # operational sensor.
    # --------------------------------------------------

    empty_result = (
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
            monthly_result,
            empty_result
        )
    )

    return result.set({
        "year": year,
        "month": month,
        "slot": slot_name,
        "n_source_composites": collection.size()
    })


def build_monthly_four_slot_image(
    year: int,
    month: int,
    config: dict,
    reference_projection: ee.Projection
) -> ee.Image:
    """
    Build the complete four-slot monthly MODIS A2 thermal image.

    Slots
    -----
    terra_day
    terra_night
    aqua_day
    aqua_night
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

        slot_images.append(
            slot_image
        )

    monthly = slot_images[0]

    for image in slot_images[1:]:
        monthly = monthly.addBands(
            image
        )

    return monthly.set({
        "year": year,
        "month": month,
        "monthly_assignment": "a2_composite_midpoint",
        "monthly_weighting": "clear_sky_count"
    })