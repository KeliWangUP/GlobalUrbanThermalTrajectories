# Rural LST exports for all cities

The complete source-to-analysis workflow is documented in
[`../../DATA_PROCESSING_WORKFLOW.md`](../../DATA_PROCESSING_WORKFLOW.md).

To create one annual CSV containing all cities with a non-empty rural halo,
run from this directory:

```bash
python export_rural_all_cities_annual.py
```

The exporter reads `study.start_year` and `study.end_year` from `config.json`.
Set `export.rural_city_ids` to `null` to select all non-empty city halos.
Before submitting any task, it verifies that every selected city has its
direct or chunked city-fraction assets. Cities without a non-empty rural halo
cannot produce rural LST and are excluded.

The output dataset name is `export.rural_all_cities_dataset_name`; the current
configuration uses `rural_lst_all_cities`, keeping the previous
`rural_lst_over_300k` exports separate. The exporter submits one Drive CSV
task per study year, with all selected cities in each CSV.

The earlier successful 1,700-city annual flow remains available as
`export_rural_priority_cities_annual.py`. The `rural` target in
`run_export.py` now points to the all-city annual exporter.

Large-city fraction computation may still split that city's fraction image
into MODIS-aligned subtiles. This is per-city fraction chunking, not the
retired tile-batch temperature export path.

Download these annual CSVs into `data/raw/rural_all` for the analysis
pipeline. Keep the earlier priority-only CSVs in their existing
`data/raw/rural` folder; the analysis config reads only the all-city folder.

For the 2000-2024 analysis series, `study.start_year` and `study.end_year`
should be set to 2000 and 2024. Set
`export.rural_city_fraction_priority_only` to `false` before rebuilding city
fractions so the fraction builder includes non-priority cities as well.

## GHSL built-surface and volume statistics

To export built-up surface and built volume totals for each city polygon, run
from the repository root:

```bash
python scripts/gee_modul/export_ghsl_built_city_stats.py
```

The export uses the configured `cities.asset` and `cities.id_field`. For each
configured GHSL year (currently 2000, 2005, 2010, 2015, and 2020), it sums
`built_surface` from `JRC/GHSL/P2023A/GHS_BUILT_S` and
`built_volume_total` from `JRC/GHSL/P2023A/GHS_BUILT_V` over the city polygon.
It also exports the valid, unmasked pixel count for each band/year and the
polygon area in square metres. Sum units are square metres for built surface
and cubic metres for built volume. The reduction scale, area error tolerance,
source collections, bands, and years are configured in `config.json`.

The script submits one CSV task to the configured Earth Engine Drive folder.
Its configured basename is
`export.ghsl_built_dataset_name` with the project version appended. Download
the completed CSV locally when the Earth Engine task succeeds. This export
does not modify the source GHSL assets or the existing SUHI/metadata Parquets.

## MODIS sample-point ISA, GHSL and water attributes

The sample-point composition workflow summarizes the existing MODIS grid
points in `export.asset_root`. Each raster is aggregated to the MODIS
reference grid with `reduceResolution`, then sampled at the existing MODIS
pixel-center points. This avoids constructing per-point vector polygons while
keeping each output value associated with the same 1 km MODIS cell.

To check one sample point from the city configured at `test.city_id`, run:

```bash
python scripts/gee_modul/test_samplepoint_attributes.py
```

The check evaluates one point in that city and validates the annual ISA urban
fractions, the water fraction range, and availability of
GHSL pixels. It does not submit an export task.

To submit exports for all sample points, run:

```bash
python scripts/gee_modul/export_samplepoint_attributes.py
```

The workflow submits one Drive CSV task per `sample_points.city_batch_size`
cities (currently 50), with one output row per sample point. It also supports
the `samplepoint_attributes` target in `run_export.py`. It is currently the
only enabled `export.targets` entry, so running `run_export.py` submits the
full sample-point export without starting city or rural exports.

For each configured year from 1985 through 2021, the export contains
`isa_urban_fraction_YEAR`. ISA values `0` and `2`-`38` are treated as valid;
`0` means non-urban and `2`-`38` encode the year of urban expansion, with
`2` = 2021, `3` = 2020, ..., `38` = 1985 or earlier. The fraction for year
`Y` is the number of valid ISA pixels that are already urban by the end of
year `Y`, divided by all valid ISA pixels in that MODIS cell. In code terms,
urban pixels satisfy `ISA >= 2023 - Y`; therefore the fraction for a later
year cannot be lower than the fraction for an earlier year. Fractions use
unweighted native ISA pixel counts, not area weighting. No-data and values
outside the defined valid classes are excluded. The band is selected from
`samplepoint_attributes.isa_band`; when that field is `null`, the asset must
contain exactly one band.

For GHSL, the export contains the mean and valid 100 m pixel count for
`built_surface` (m² per source pixel) and `built_volume_total` (m³ per source
pixel) for every configured five-year interval. `water_fraction` is an
area-weighted fraction from the same GLO30 `WBM` mask as the rural workflow:
`WBM == 0` is non-water there, so water is `WBM != 0`. The CSV also includes
water and valid WBM areas in m². Results use the configured Drive folder and
dataset name; they do not modify the source point, ISA, GHSL, or DEM assets.

## MODIS sample-point monthly LST

The sample-point LST exporter uses the same monthly MODIS composites, QA
settings, city-LST output fields, and CSV folder as the city exporter, but
samples each monthly image at every point in `export.asset_root`. Run it
directly:

```bash
python scripts/gee_modul/export_samplepoint_lst.py
```

Or add `samplepoint_lst` to `export.targets` and run `run_export.py`. The
export submits one Drive CSV task per study year and includes all sample
points. The output has the city-LST columns plus `pixel_id`, which identifies
each sampled MODIS cell and allows joining these records to the other
sample-point exports.

Each record represents one sample point and one calendar month. Because the
spatial unit is a single MODIS pixel, each available LST value is repeated in
the `mean` and `median` columns, `stdDev` is 0, and `count` is 1. Missing
observations remain null. QC `_mean` fields contain the sampled pixel's
monthly QC values. A single-city, single-point check is available via:

```bash
python scripts/gee_modul/test_samplepoint_lst.py
```
