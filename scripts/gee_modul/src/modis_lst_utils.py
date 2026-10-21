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

    # Masked QC fails every test (bits 0-1 and 6-7 become 3).
    qc = image.select(qc_band).unmask(255)

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

    mask = mandatory_qa.lte(mandatory_qa_max)

    if max_lst_error_code is not None:
        mask = mask.And(
            lst_error.lte(max_lst_error_code)
        )

    if require_good_data_quality:
        mask = mask.And(
            data_quality.eq(0)
        )

    return mask.rename("qa_pass")


def compute_a2_validity(
    image: ee.Image,
    lst_band: str,
    qc_band: str,
    clear_band: str,
    qa_cfg: dict
) -> ee.Image:
    """
    Single source of truth for A2 observation validity.

    Used by production processing and by diagnostics.

    Returns bands: lst_available, clear_count, has_clear_obs,
    qa_pass, final_valid, final_clear_count.

    qa_pass is always 1 when qa_cfg["apply_qa_mask"] is false.
    """

    lst_available = (
        image
        .select(lst_band)
        .mask()
        .unmask(0)
        .gt(0)
        .rename("lst_available")
    )

    clear_count = (
        count_clear_sky_bits(
            image=image,
            clear_band=clear_band
        )
        .unmask(0)
        .toFloat()
        .rename("clear_count")
    )

    has_clear_obs = clear_count.gt(0).rename("has_clear_obs")

    if qa_cfg.get("apply_qa_mask", True):
        qa_pass = build_a2_qa_mask(
            image=image,
            qc_band=qc_band,
            mandatory_qa_max=qa_cfg["mandatory_qa_max"],
            max_lst_error_code=qa_cfg.get("max_lst_error_code"),
            require_good_data_quality=qa_cfg.get(
                "require_good_data_quality",
                False
            )
        )
    else:
        qa_pass = ee.Image.constant(1).rename("qa_pass")

    final_valid = (
        lst_available
        .And(has_clear_obs)
        .And(qa_pass)
        .rename("final_valid")
    )

    final_clear_count = (
        clear_count
        .multiply(final_valid)
        .rename("final_clear_count")
    )

    return ee.Image.cat([
        lst_available,
        clear_count,
        has_clear_obs,
        qa_pass,
        final_valid,
        final_clear_count
    ])


def preprocess_a2_slot(
    image: ee.Image,
    lst_band: str,
    qc_band: str,
    clear_band: str,
    slot_name: str,
    qa_cfg: dict,
    scale_factor: float = 0.02,
    kelvin_offset: float = -273.15
) -> ee.Image:
    """
    Prepare one MODIS A2 observation slot.

    Validity comes from compute_a2_validity() and follows qa_cfg.
    """

    lst = (
        image
        .select(lst_band)
        .multiply(scale_factor)
        .add(kelvin_offset)
    )

    validity = compute_a2_validity(
        image=image,
        lst_band=lst_band,
        qc_band=qc_band,
        clear_band=clear_band,
        qa_cfg=qa_cfg
    )

    valid_numeric = (
        validity
        .select("final_valid")
        .toByte()
    )

    weight = (
        validity
        .select("final_clear_count")
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


def get_monthly_a2_collection(
    year: int,
    month: int,
    collection_id: str
) -> ee.ImageCollection:
    """
    A2 composites assigned to a calendar month by composite midpoint
    (start + 4 days). Shared by production and diagnostics.
    """

    month_start = ee.Date.fromYMD(year, month, 1)
    month_end = month_start.advance(1, "month")

    return (
        ee.ImageCollection(collection_id)
        .filterDate(
            month_start.advance(-4, "day"),
            month_end.advance(-4, "day")
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

    collection = get_monthly_a2_collection(
        year=year,
        month=month,
        collection_id=slot_cfg["collection"]
    )

    qa_cfg = modis_cfg["qa"]

    def _process(image):

        return preprocess_a2_slot(
            image=image,
            lst_band=slot_cfg["lst_band"],
            qc_band=slot_cfg["qc_band"],
            clear_band=slot_cfg["clear_band"],
            slot_name=slot_name,
            qa_cfg=qa_cfg,
            scale_factor=modis_cfg["scale_factor"],
            kelvin_offset=modis_cfg["kelvin_offset"]
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