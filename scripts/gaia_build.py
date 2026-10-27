#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from osgeo import gdal

# Configure global GDAL runtime parameters
gdal.UseExceptions()
gdal.SetConfigOption('GDAL_CACHEMAX', '4096')        # 4GB RAM cache for fast tile I/O
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS') # Enable multi-core parallel compression


def check_and_fix_tile(filepath):
    """
    Check if a raster has an inverted (positive) North-South resolution.
    If corrupted, vertically flip the tile and reproject using GDAL warp.
    Returns: (status: str, filepath: str)
    """
    try:
        ds = gdal.Open(filepath, gdal.GA_ReadOnly)
        if ds is None:
            return "FAILED_OPEN", filepath

        gt = ds.GetGeoTransform()
        pixel_height = gt[5]  # North-South resolution step

        if pixel_height > 0:
            # Positive pixel height detected: must flip vertically
            temp_fixed = filepath.replace(".tif", "_fixed_temp.tif")
            warp_opts = gdal.WarpOptions(
                resampleAlg='near', # Strictly preserve discrete integer classes
                creationOptions=['COMPRESS=DEFLATE', 'TILED=YES'],
                multithread=False   # Avoid thread contention inside sub-process
            )
            gdal.Warp(temp_fixed, ds, options=warp_opts)
            ds = None  # Release read handle
            
            # Atomic replacement of the original corrupt tile
            shutil.move(temp_fixed, filepath)
            return "FIXED", filepath

        ds = None
        return "CLEAN", filepath

    except Exception as e:
        return f"ERROR: {str(e)}", filepath


def batch_validate_and_fix(tile_list, max_workers=None):
    """Scan and repair invalid GeoTIFF tiles in parallel using multi-processing."""
    print(f"\n[PHASE 1/4] Scanning {len(tile_list)} tiles for NS resolution anomalies...")
    t0 = time.time()
    
    fixed_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_and_fix_tile, f): f for f in tile_list}
        
        for future in as_completed(futures):
            status, path = future.result()
            if status == "FIXED":
                fixed_count += 1
                filename = os.path.basename(path)
                print(f"  -> Fixed inverted orientation: {filename}")
            elif status.startswith("ERROR") or status == "FAILED_OPEN":
                error_count += 1
                print(f"  -> [WARNING] Processing failed for: {path} ({status})")

    elapsed = time.time() - t0
    print(f"[PHASE 1/4 COMPLETE] Done in {elapsed:.1f}s. Fixed: {fixed_count}, Errors: {error_count}, Valid: {len(tile_list) - fixed_count - error_count}")


def build_vrt_index(tile_list, vrt_path):
    """Construct an in-memory or XML virtual dataset index."""
    print(f"\n[PHASE 2/4] Indexing tiles into VRT: {vrt_path}...")
    t0 = time.time()
    
    vrt_options = gdal.BuildVRTOptions(
        srcNodata=0,
        VRTNodata=0,
        resampleAlg='near'
    )
    gdal.BuildVRT(vrt_path, tile_list, options=vrt_options)
    
    elapsed = time.time() - t0
    print(f"[PHASE 2/4 COMPLETE] VRT generated in {elapsed:.1f}s.")


def convert_vrt_to_cog(vrt_path, cog_path):
    """
    Stream VRT blocks to a single sparse, compressed Cloud-Optimized GeoTIFF (COG).
    """
    print(f"\n[PHASE 3/4] Streaming VRT to sparse COG: {cog_path}...")
    t0 = time.time()

    # Pre-remove existing output file if present
    if os.path.exists(cog_path):
        os.remove(cog_path)

    translate_opts = gdal.TranslateOptions(
        format='GTiff',
        creationOptions=[
            'TILED=YES',              # 512x512 tile blocks required by GEE
            'BLOCKXSIZE=512',
            'BLOCKYSIZE=512',
            'COMPRESS=DEFLATE',       # Lossless high-ratio compression
            'PREDICTOR=2',            # Horizontal differencing for integer arrays
            'ZLEVEL=9',               # High compression ratio
            'SPARSE_OK=TRUE',         # Omit empty ocean/desert tiles to save disk space
            'BIGTIFF=YES',            # Handle global coverage without hitting the 4GB cap
            'NUM_THREADS=ALL_CPUS'    # Parallel tile compression
        ],
        callback=gdal.TermProgress_nocb
    )
    
    gdal.Translate(cog_path, vrt_path, options=translate_opts)
    
    elapsed = time.time() - t0
    size_gb = os.path.getsize(cog_path) / (1024 ** 3)
    print(f"\n[PHASE 3/4 COMPLETE] COG generated in {elapsed:.1f}s. File size: {size_gb:.2f} GB")


def build_discrete_overviews(cog_path):
    """
    Build internal overview pyramids using NEAREST resampling.
    Ensures discrete integer year classes remain strictly unblurred.
    """
    print(f"\n[PHASE 4/4] Building internal NEAREST overview pyramids...")
    t0 = time.time()

    ds = gdal.Open(cog_path, gdal.GA_Update)
    if ds is None:
        raise RuntimeError(f"Failed to open generated COG for overviews: {cog_path}")

    # Set compression for overviews
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

    overview_levels = [2, 4, 8, 16, 32, 64]
    ds.BuildOverviews('NEAREST', overview_levels, callback=gdal.TermProgress_nocb)
    
    ds = None  # Flush buffers and close
    
    elapsed = time.time() - t0
    print(f"\n[PHASE 4/4 COMPLETE] Pyramids built in {elapsed:.1f}s.")


def main():
    total_start_time = time.time()

    # Configure paths
    ROOT_DIR = "/home/GlobalUrbanThermalTrajectories/data/raw/"
    DATA_DIR = "/home/GlobalUrbanThermalTrajectories/data/raw/gaia_new"
    OUTPUT_VRT = os.path.join(ROOT_DIR, "gaia_global_mosaic.vrt")
    OUTPUT_COG = os.path.join(ROOT_DIR, "gaia_global_isa_cog.tif")

    # Discover tiles
    tile_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif")))
    # Filter out previously generated outputs from input list
    tile_files = [f for f in tile_files if not f.endswith("_cog.tif") and not f.endswith("_temp.tif")]

    if not tile_files:
        print(f"[ERROR] No input GeoTIFF tiles found in {DATA_DIR}")
        sys.exit(1)

    print(f"=== Starting Integrated GAIA Tile Consolidation Pipeline ===")
    print(f"Directory:    {DATA_DIR}")
    print(f"Tile Count:   {len(tile_files)}")
    print(f"Target COG:   {OUTPUT_COG}")

    # Execute pipeline stages
    batch_validate_and_fix(tile_files)
    build_vrt_index(tile_files, OUTPUT_VRT)
    convert_vrt_to_cog(OUTPUT_VRT, OUTPUT_COG)
    build_discrete_overviews(OUTPUT_COG)

    # Optional: clean up intermediate VRT
    if os.path.exists(OUTPUT_VRT):
        os.remove(OUTPUT_VRT)

    total_elapsed = time.time() - total_start_time
    print(f"\n========================================================")
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_elapsed / 60:.2f} minutes.")
    print(f"Generated Asset Ready for GEE: {OUTPUT_COG}")
    print(f"========================================================")


if __name__ == "__main__":
    main()