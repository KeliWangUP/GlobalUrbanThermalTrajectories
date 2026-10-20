import ee
from src.export_utils import get_modis_reference_grid, initialize_gee, load_config

def build_modis_coordinate_image(
    projection: ee.Projection,
    include_lonlat: bool = True
) -> ee.Image:
    """
    Create an image containing MODIS grid coordinates.

    Parameters
    ----------
    projection : ee.Projection
        Native MODIS projection.
    include_lonlat : bool, default=True
        Whether to include longitude and latitude bands.

    Returns
    -------
    ee.Image
        Coordinate image.
    """

    coords = (
        ee.Image.pixelCoordinates(projection)
        .rename(["modis_x", "modis_y"])
    )

    if include_lonlat:
        lonlat = (
            ee.Image.pixelLonLat()
            .rename(["lon", "lat"])
        )

        coords = coords.addBands(lonlat)

    print("[INFO] MODIS coordinate image created.")

    return coords

def create_candidate_sample_points(
    coordinate_image: ee.Image,
    city_bounds: ee.FeatureCollection,
    projection: ee.Projection,
    native_scale: float,
    city_id_field: str,
    tile_scale: int
) -> ee.FeatureCollection:
    """
    Generate MODIS pixel-center sample points within city boundaries.

    Parameters
    ----------
    coordinate_image : ee.Image
        Image containing MODIS pixel coordinates and optional lon/lat.
    city_bounds : ee.FeatureCollection
        Urban boundary polygons.
    projection : ee.Projection
        Native MODIS projection.
    native_scale : float
        Native MODIS pixel scale.
    city_id_field : str
        City identifier field.
    tile_scale : int, default=4
        Earth Engine tileScale used to reduce memory pressure.

    Returns
    -------
    ee.FeatureCollection
        MODIS pixel-center points associated with cities.
    """

    points = coordinate_image.sampleRegions(
        collection=city_bounds,
        properties=[city_id_field],
        projection=projection,
        scale=native_scale,
        geometries=True,
        tileScale=tile_scale
    )

    print("[INFO] Candidate MODIS sample points created.")

    return points

def add_global_pixel_id(
    points: ee.FeatureCollection,
    prefix: str = "MODIS"
) -> ee.FeatureCollection:
    """
    Add a globally stable pixel ID based on MODIS grid coordinates.

    Parameters
    ----------
    points : ee.FeatureCollection
        MODIS sample points.
    prefix : str, default="MODIS"
        Prefix used for pixel IDs.

    Returns
    -------
    ee.FeatureCollection
        FeatureCollection with pixel_id.
    """

    def _add_id(feature):

        x = ee.Number(feature.get("modis_x")).toInt()
        y = ee.Number(feature.get("modis_y")).toInt()

        pixel_id = (
            ee.String(prefix)
            .cat("_")
            .cat(x.format("%d"))
            .cat("_")
            .cat(y.format("%d"))
        )

        return feature.set({
            "pixel_id": pixel_id,
            "modis_x": x,
            "modis_y": y
        })

    points = points.map(_add_id)

    print("[INFO] Global MODIS pixel IDs assigned.")

    return points

def build_modis_city_sample_points(config: dict):
    """
    Build globally consistent MODIS pixel-center sample points.

    Parameters
    ----------
    config : dict
        Project configuration.

    Returns
    -------
    ee.FeatureCollection
        Sample points containing pixel_id and city_id.
    """

    grid_cfg = config["reference_grid"]
    city_cfg = config["cities"]
    point_cfg = config["sample_points"]

    print("[INFO] Starting MODIS sample-point generation.")

    # --------------------------------------------------
    # Load city boundaries
    # --------------------------------------------------

    cities = ee.FeatureCollection(city_cfg["asset"])

    print(f"[INFO] City boundary asset: {city_cfg['asset']}")

    # --------------------------------------------------
    # Get reference MODIS projection
    # --------------------------------------------------

    modis_proj, native_scale = get_modis_reference_grid(config)

    # --------------------------------------------------
    # Build coordinate raster
    # --------------------------------------------------

    coord_image = build_modis_coordinate_image(
        projection=modis_proj,
        include_lonlat=point_cfg["include_lonlat"]
    )

    # --------------------------------------------------
    # Sample MODIS centers inside city polygons
    # --------------------------------------------------

    points = create_candidate_sample_points(
        coordinate_image=coord_image,
        city_bounds=cities,
        projection=modis_proj,
        native_scale=native_scale,
        city_id_field=city_cfg["id_field"],
        tile_scale=config["processing"]["tile_scale"]
    )

    # --------------------------------------------------
    # Add stable global IDs
    # --------------------------------------------------

    points = add_global_pixel_id(
        points=points,
        prefix=grid_cfg["pixel_id_prefix"]
    )

    print("[INFO] Sample-point generation completed.")

    return points

"""Single city sample-point extraction example."""

# city_id = 5249

# munich_points = points.filter(
#     ee.Filter.eq(
#         config["cities"]["id_field"],
#         city_id
#     )
# )

# print(
#     "[DEBUG] Munich sample-point count:",
#     munich_points.size().getInfo()
# )

# print(
#     "[DEBUG] First sample point:",
#     munich_points.first().toDictionary().getInfo()
# )

"""Global sample-point validation functions."""

# def validate_global_sample_points(
#     points: ee.FeatureCollection,
#     cities: ee.FeatureCollection,
#     city_id_field: str = "ID_UC_G0"
# ) -> None:
#     """
#     Validate the global MODIS reference sample points.

#     Checks
#     ------
#     1. Total number of sampled points.
#     2. Number of unique MODIS pixel IDs.
#     3. Potential duplicate MODIS pixels.
#     4. Number of represented cities.
#     5. Number of source cities.
#     6. Missing required properties.
#     7. Example records.

#     Parameters
#     ----------
#     points : ee.FeatureCollection
#         Generated MODIS center points.

#     cities : ee.FeatureCollection
#         Source city polygons.

#     city_id_field : str
#         Unique city identifier field.
#     """

#     print("\n========== GLOBAL SAMPLE POINT QC ==========")

#     total_points = points.size().getInfo()

#     unique_pixels = (
#         points
#         .aggregate_array("pixel_id")
#         .distinct()
#         .size()
#         .getInfo()
#     )

#     represented_cities = (
#         points
#         .aggregate_array(city_id_field)
#         .distinct()
#         .size()
#         .getInfo()
#     )

#     total_cities = cities.size().getInfo()

#     missing_required = (
#         points
#         .filter(
#             ee.Filter.notNull([
#                 "pixel_id",
#                 city_id_field,
#                 "modis_x",
#                 "modis_y",
#                 "lon",
#                 "lat"
#             ]).Not()
#         )
#         .size()
#         .getInfo()
#     )

#     duplicate_memberships = total_points - unique_pixels

#     print(f"[QC] Total sample points:      {total_points}")
#     print(f"[QC] Unique MODIS pixels:      {unique_pixels}")
#     print(f"[QC] Duplicate memberships:    {duplicate_memberships}")
#     print(f"[QC] Source cities:            {total_cities}")
#     print(f"[QC] Represented cities:       {represented_cities}")
#     print(f"[QC] Missing required fields:  {missing_required}")

#     if missing_required > 0:
#         raise ValueError(
#             "Some sample points have missing required properties."
#         )

#     if duplicate_memberships > 0:
#         print(
#             "[WARNING] Some MODIS pixels occur more than once. "
#             "This may indicate overlapping city polygons."
#         )
#     else:
#         print("[QC PASS] All MODIS pixel IDs are globally unique.")

#     if represented_cities < total_cities:
#         print(
#             f"[WARNING] {total_cities - represented_cities} cities "
#             "have no sampled MODIS center point."
#         )
#     else:
#         print("[QC PASS] All cities are represented.")

#     print("\n[DEBUG] First five sample points:")

#     preview = points.limit(5).getInfo()

#     for feature in preview["features"]:
#         print(feature["properties"])

# cities = ee.FeatureCollection(
#     "users/keliwang/urban_expansion/GHSL_urban_bounds_2024"
# )

# validate_global_sample_points(
#     points=points,
#     cities=cities,
#     city_id_field="ID_UC_G0"
# )


"""Export MODIS sample points to Earth Engine asset."""

def prepare_sample_points_for_export(
    points: ee.FeatureCollection,
    city_id_field: str,
    include_lonlat: bool
) -> ee.FeatureCollection:
    """
    Keep only the fields required by the permanent MODIS reference-point asset.

    Parameters
    ----------
    points : ee.FeatureCollection
        Generated MODIS center points.

    city_id_field : str, default="ID_UC_G0"
        Unique GHSL urban-centre identifier.

    Returns
    -------
    ee.FeatureCollection
        Cleaned reference-point collection.
    """

    required_fields = ["pixel_id", city_id_field, "modis_x", "modis_y"]
    if include_lonlat:
        required_fields.extend(["lon", "lat"])

    cleaned = points.select(
        propertySelectors=required_fields,
        retainGeometry=True
    )

    print("[INFO] Sample points prepared for export.")
    print(f"[INFO] Fields retained: {required_fields}")
    print("[INFO] Geometry retained: True")

    return cleaned

def asset_exists(asset_id: str) -> bool:
    """
    Check whether an Earth Engine asset already exists.

    Parameters
    ----------
    asset_id : str
        Earth Engine asset ID.

    Returns
    -------
    bool
        True if the asset exists.
    """

    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False

def export_sample_points_to_asset(
    points: ee.FeatureCollection,
    output_asset: str,
    description: str = "modis_sample_points",
    overwrite: bool = False
) -> ee.batch.Task:
    """
    Export the permanent MODIS sample-point collection to a GEE asset.

    Parameters
    ----------
    points : ee.FeatureCollection
        Clean MODIS reference points.

    output_asset : str
        Destination Earth Engine asset ID.

    description : str
        Earth Engine task description.

    overwrite : bool, default=False
        Delete the existing asset before export.
        Keep False unless replacement is intentional.

    Returns
    -------
    ee.batch.Task
        Started Earth Engine export task.
    """

    if asset_exists(output_asset):

        if not overwrite:
            raise FileExistsError(
                f"Output asset already exists:\n{output_asset}\n"
                "Use a new version or explicitly set overwrite=True."
            )

        print(f"[WARNING] Removing existing asset: {output_asset}")
        ee.data.deleteAsset(output_asset)

    print("\n========== STARTING EXPORT ==========")
    print(f"[INFO] Output asset: {output_asset}")
    print(f"[INFO] Description: {description}")

    task = ee.batch.Export.table.toAsset(
        collection=points,
        description=description,
        assetId=output_asset
    )

    task.start()

    print(f"[INFO] Task ID: {task.id}")
    print("[INFO] Export task submitted.")

    return task

def main() -> None:
    config = load_config()
    initialize_gee(config)
    points = build_modis_city_sample_points(config)
    export_config = config["export"]
    export_points = prepare_sample_points_for_export(
        points=points,
        city_id_field=config["cities"]["id_field"],
        include_lonlat=config["sample_points"]["include_lonlat"]
    )
    export_sample_points_to_asset(
        points=export_points,
        output_asset=export_config["asset_root"],
        description=export_config["sample_points_description"],
        overwrite=export_config["overwrite_sample_points"]
    )


if __name__ == "__main__":
    main()