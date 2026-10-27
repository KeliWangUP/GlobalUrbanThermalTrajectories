import glob
import os
from osgeo import gdal

data_dir = "/home/GlobalUrbanThermalTrajectories/data/raw/gaia_new"
tif_files = glob.glob(os.path.join(data_dir, "*.tif"))

fixed_count = 0

for f in tif_files:
    ds = gdal.Open(f, gdal.GA_ReadOnly)
    if ds is None:
        continue
    
    gt = ds.GetGeoTransform()
    # gt[5] is the North-South pixel resolution (pixel height)
    if gt[5] > 0:
        print(f"[FOUND] Positive NS resolution: {f} (pixel_height = {gt[5]})")
        
        # Calculate new corrected origin and negative resolution
        # New top-left Y = old bottom-left Y + (rows * pixel_height)
        new_top_left_y = gt[3] + (ds.RasterYSize * gt[5])
        new_pixel_height = -gt[5]
        corrected_gt = (gt[0], gt[1], gt[2], new_top_left_y, gt[4], new_pixel_height)
        
        # Method: Use gdal.Warp to re-align coordinates and flip data correctly
        temp_out = f.replace(".tif", "_fixed.tif")
        gdal.Warp(
            temp_out,
            ds,
            options=gdal.WarpOptions(
                resampleAlg='near', # Preserve discrete integer years
                creationOptions=['COMPRESS=DEFLATE']
            )
        )
        ds = None  # Release file handle
        
        # Replace original with the fixed one
        os.replace(temp_out, f)
        fixed_count += 1
    else:
        ds = None

print(f"\n[DONE] Checked {len(tif_files)} files. Fixed {fixed_count} inverted tiles.")