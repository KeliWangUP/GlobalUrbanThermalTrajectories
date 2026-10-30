# Rural LST exports for all cities

The complete source-to-analysis workflow is documented in
[`../../DATA_PROCESSING_WORKFLOW.md`](../../DATA_PROCESSING_WORKFLOW.md).

The root scripts are operational entry points, `src/` contains reusable
processing and export helpers, and `archive/` holds superseded
troubleshooting scripts. Existing `test_*.py` scripts are live Earth Engine
checks, not offline unit tests; they are retained as operational diagnostics.
Offline regression checks for the sample-point exports can be run from the
repository root with:

```bash
python -m unittest discover -s scripts/gee_modul/tests -p "test_samplepoint*.py"
```

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

Earlier troubleshooting scripts are kept under `archive/` and remain
executable from the repository root. The single-batch retry helper accepts
either a legacy `batch` string or a one-entry `batches` list in `failed.json`:

```bash
python scripts/gee_modul/archive/export_samplepoint_attributes_failed_batch.py
```

The retry script checks that the current sample-point asset and batch size
still produce the same total batch count before submitting anything. It
exports only the selected batch and appends `_retry` to the Drive task/file
name so the retry does not collide with the original output. For a list of
multiple failed batches, use the per-city recovery script instead.

To diagnose a persistent reprojection failure before retrying a complete
batch, run:

```bash
python scripts/gee_modul/archive/export_samplepoint_attributes_failed_city_probe.py
```

This requires exactly one failed batch in `failed.json`, selects the city
with the most sample points in that batch, and submits one city-only CSV
probe.

To map all failed batches to their cities and submit city-only retries in
controlled waves, run:

```bash
python scripts/gee_modul/export_samplepoint_attributes_failed_cities.py
```

The script validates that every failed task has the same total batch count as
the current sample-point asset, then creates one Drive CSV task per city in
the selected wave. The wave is controlled by
`failed_samplepoint_retry.city_offset` and
`failed_samplepoint_retry.max_cities_per_run` in `config.json`. If the limit
is at least the number of failed cities, one run submits all of them; the
current configuration selects up to 2,750 cities from offset 0. Each task
name includes the city ID and `_retry`; the script prints the Earth Engine
task ID next to each city ID and appends
the submission time, city ID, and task ID to
`failed_city_retry_task_log.jsonl`. After a wave, increase `city_offset` by
the number of submitted cities before starting the next wave. Keep all
successful batch exports; this script maps only the batch names listed in
`failed.json`. Matching the total batch count alone cannot prove that the
asset's sorted city IDs or original batch size are unchanged: verify both
before submitting. If submission stops partway through, check the task log
and Earth Engine task list before resuming; rerunning the same offset can
create duplicate tasks.

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
samples each monthly image at every point in `export.asset_root`.

The global annual and global single-month exports described below have failed
at full sample size. To test a spatially partitioned export, run:

```bash
python scripts/gee_modul/export_samplepoint_lst_spatial.py
```

In `samplepoint_lst_spatial_export` in `config.json`, the test tile side
length is 640 native MODIS pixels (about 593 km in the MODIS projected grid
at the reference scale). One 640-pixel tile and one 1280-pixel tile for 2020
completed successfully; the full study period and denser regions still need
to be tested.
`selection: "test_city"` submits exactly one task for the tile containing
the first `pixel_id` in `test.city_id`. The tile includes **all** sample
points in the selected city scope within that spatial block, not just points
belonging to the test city.
Each task covers every month from `study.start_year` to `study.end_year`,
inclusive; the current study configuration is 2000-2024. The `study` years
are shared with other exporters. The script prints
the tile indices, point count, expected row count, task ID, and name.

`samplepoint_lst_spatial_export.city_scope` is currently `"priority"`:
city IDs are read from `priority_city_csv`, relative to `config.json` unless
an absolute path is provided. The configured historical rural 2020 CSV
contains 1,700 distinct IDs; the 2000-2024 historical CSVs were checked to
contain the same IDs. The exporter checks that every requested city exists
in the sample-point asset and stops with missing IDs if not. It filters
sample points to these IDs **before** choosing tiles. A SHA-256 digest of the
sorted city IDs is recorded in the task log and its first 12 characters
appear in priority task/file names.

Set `city_scope` to `"all"` to retain the original global behavior; the
priority CSV is then not used. Global outputs overlap with priority outputs,
so do not concatenate them without deduplication. This setting affects only
the spatial sample-point LST exporter, not other export targets. The
historical CSV is a local input; it must exist on the machine submitting the
tasks. `test_city` must belong to the selected scope.

The tile indices are `floor(modis_x / tile_pixels)` and
`floor(modis_y / tile_pixels)`, including for negative coordinates. Each
sample point belongs to one half-open coordinate block. Only occupied tiles
are submitted, and the CSV fields, LST/QC computation and missing-value
handling are shared with the existing monthly exporter. A task may still
exceed Earth Engine's limits because of point density or time span; the tile
size is a starting hypothesis, not a proven capacity.

To select one known tile, set `selection` to `"tile"` and set integer
`tile_column` and `tile_row`. To process all occupied tiles, set `selection`
to `"all"`; `tile_offset` and `max_tiles_per_run` control which sorted tiles
are submitted. The configured maximum is initially 1. Keep the point asset,
tile size, and study period unchanged between waves, or offsets and output
names may refer to different work.

Submitted tasks are appended to `samplepoint_lst_spatial_task_log.jsonl` with
their city scope and list digest, tile, point count, time range and task ID.
This is a submission log, not
an automatic completion tracker. If submission stops or a task fails, check
the log and Earth Engine task status before submitting again; rerunning the
same tile can create duplicates. Changing tile size or time range to retry
failed work changes its coverage: retain a manifest and merge only successful,
non-overlapping outputs, checking `(pixel_id, city ID, year, month)` keys.
Matching output row counts should be checked against the printed expectation.

The previous unpartitioned annual exporter remains available:

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

To test one month for one sample point, set
`samplepoint_lst_export.year` and `.month` in `config.json` and run:

```bash
python scripts/gee_modul/test_samplepoint_lst_month.py
```

To submit a **full-sample, single-month** CSV task using those same
configuration values, run:

```bash
python scripts/gee_modul/export_samplepoint_lst_month.py
```

This creates one task named
`samplepoint_lst_monthly_<version>_<year>_<month>` and exports every point
for that month. Alternatively, add `samplepoint_lst_month` to
`export.targets` and run `run_export.py`. The default targets are unchanged.
