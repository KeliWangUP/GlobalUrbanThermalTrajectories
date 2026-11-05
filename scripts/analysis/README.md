# Analysis data preparation

For the complete source-to-analysis workflow and future analysis roadmap, see
[`../../DATA_PROCESSING_WORKFLOW.md`](../../DATA_PROCESSING_WORKFLOW.md).

## Pixel-level Silver dataset

The city-partitioned pixel workflow is configured separately in
[`pixel_silver_config.json`](./pixel_silver_config.json). From the repository
root, run:

```powershell
python scripts\analysis\build_pixel_silver.py
```

It reads the configured v2 LST TAR, development Parquet, 25 rural CSVs and
priority-city ID JSON. It produces:

```text
data\processed\silver\
├── pixel_lst\city_id=<ID>\data.parquet
├── pixel_development\data.parquet
├── rural_lst\rural_<year>.parquet
└── manifest.json
```

One LST file contains all months for one city. The city folder structure is
preserved, but the TAR is not simply extracted: LST and QC columns are
strictly converted to consistent numeric types while `pixel_id` remains a
string, and nulls are retained. The existing development Parquet is validated
and copied unchanged; rural CSVs are converted with the shared city-LST CSV
validator. No ISA, water or other scientific fields are corrected or filtered
by this conversion.

Before publication the workflow checks the 1,700-city list, per-city pixel
keys, all configured months, development matches, input hashes, numeric
conversion, output schemas, null counts and DuckDB-readable row counts.
Invalid inputs fail explicitly. Output is first written to a uniquely named
staging directory next to `silver`; only a completed build is renamed to
`silver`. An existing `silver` directory is never overwritten. On failure,
the printed staging directory remains for inspection; it is not published.
Raw inputs are never modified.

With Hive partitioning enabled, `city_id` comes from the directory name and
`ID_UC_G0` remains the data column. Filtering on `city_id` enables city
partition pruning; queries spanning all cities may touch many small files:

```python
from pathlib import Path
import duckdb

pattern = str(Path(r"data\processed\silver\pixel_lst") / "city_id=*" / "data.parquet")
con = duckdb.connect()
rows = con.execute(
    "SELECT * FROM read_parquet(?, hive_partitioning=true) "
    "WHERE city_id=? AND year=?",
    [pattern, 5249, 2020]
).fetchall()
```

## City-month Gold trend dataset

The Gold builder streams the Silver pixel LST, development, and rural Parquets
through DuckDB. It applies the configured water-pixel and per-slot QC rules,
joins urban and rural observations on city/year/month, estimates monthly
urban, rural, and SUHI trends, and attaches city climate metadata by
`ID_UC_G0`.

The configuration maps to the current Silver schemas and the existing city
metadata lookup:
[`city_month_gold_config.json`](./city_month_gold_config.json). Paths are
resolved relative to that config, so the same setup works after moving the
project to Linux. Build from the repository root with:

```bash
python scripts/analysis/build_city_month_gold.py
```

By default the builder uses years 2000–2024 and publishes two trend windows,
2003–2019 and 2003–2024, under
`data/processed/gold/city_month_trends`. It creates:

- `city_month_paired_qc.parquet`: matched city/month/sensor-slot observations,
  QC flags, urban/rural means, and SUHI.
- `city_month_slot_trends_<start>_<end>.parquet`: long-form slopes and QC
  diagnostics for each city/month/slot.
- `city_month_trend_wide_<start>_<end>.parquet`: one row per priority city,
  including climate metadata and month/slot trend columns.
- `manifest.json`: configuration, upstream Silver-manifest hash, metadata
  hash, output schemas, row counts, and sizes.

The builder validates the 1,700-city list, unique pixel and rural keys,
complete city/month coverage, development-to-LST pixel counts, climate
metadata coverage, and the final one-row-per-city output. Existing Gold
output is never overwritten; choose a new `output_dir` in the config to create
another build.

### Gold completeness and trend diagnostics

After building the Gold trend tables, run:

```bash
python scripts/analysis/diagnose_city_month_gold.py
```

The diagnostics use the wide table as the QC denominator, so city/month/slot
combinations with zero valid years remain counted as failures. They write
monthly pass counts and fractions by Köppen major zone and full class, with a
separate Af/Am/Aw table and heatmap. For each A/B/C/D major zone they also
plot QC-passing monthly distributions of the urban, rural, and SUHI slopes,
with Terra/Aqua separated and day/night shown in separate panels. The plots
use calendar months; no cold/warm seasons or global wet-season months are
imposed. Monthly precipitation is not currently an input, so Af/Am/Aw results
are reported month by month rather than labeling months as wet or dry.

The diagnostics config also supports `"hemisphere": "north"`, `"south"`, or
`"all"`. North/south selection uses `centroid_lat` in the configured city
metadata and filters both the wide QC denominator and long trend records;
cities exactly on the equator are excluded from north and south. The configured
`expected_city_count` checks the unfiltered Gold wide table. Input and output
paths are set in
[`city_month_gold_diagnostics_config.json`](./city_month_gold_diagnostics_config.json).
The current config selects the northern hemisphere and writes to
`data/processed/result/city_month_diagnostics_north`; an existing output
directory is never overwritten.

The v2 mean-LST fields are already `float32`; writing them as `float64`
preserves those stored values but does not restore precision previously lost
when the v2 inputs were created. The manifest includes source SHA-256 hashes
and row counts. Data under `data/` is not tracked by Git.

### City/month/slot trend Gold table

Build `city_month_slot_trends.parquet` from the monthly master SUHI table with:

```bash
python scripts/analysis/build_city_month_slot_trends.py
```

The settings are in
[`city_month_slot_trends_config.json`](./city_month_slot_trends_config.json).
The default period is 2003–2019, and the default output is
`data/processed/gold/city_month_slot_trends_2003_2019/`. Each output row is
one city, calendar month, and Terra/Aqua day/night slot. It includes the
Köppen-Geiger class as `climate`, centroid latitude, hemisphere derived from
latitude (`north`, `south`, `equator`, or null for missing latitude), OLS
slopes for SUHI/urban/rural LST, their standard errors, and `n_valid_years`.

Only years with that slot's `qc_pass_* == true` and finite urban, rural, and
SUHI means are used. A row is emitted only with at least the configured
`minimum_valid_years` (15 by default); this does not require every year in the
period to pass QC. Slopes and standard errors are scaled to degrees Celsius
per decade. Standard errors are the conventional OLS slope standard errors,
calculated from the residual sum of squares with `n - 2` degrees of freedom.
The output directory is never overwritten; the Parquet and its provenance
manifest are published together.

### City/month/slot trend figures

After building the city-level trend Parquet, run:

```bash
python scripts/analysis/plot_city_month_slot_trends.py
```

The input, output, Köppen major groups, and figure resolution are configured in
[`city_month_slot_trend_figures_config.json`](./city_month_slot_trend_figures_config.json).
By default, the script groups full Köppen classes by their first letter and
plots A/B/C/D; add E to `climate_groups` to include polar cities.

Figure 1 shows each group's monthly city-level SUHI slope median, IQR, and
eligible-city count, with Terra/Aqua and day/night in separate panels.
Figure 2 shows the monthly median and IQR for βU, βR, and βSUHI in the same
climate/slot panels. Before aggregation, the script verifies for every city
that βSUHI equals βU − βR within the configured numerical tolerance. Climate
medians and IQRs for the three metrics are calculated independently, so the
displayed group medians are not expected to satisfy that subtraction identity.
Rows without a valid Köppen class are excluded from climate-group summaries
and figures, but retained in the city-level trend input; their row and city
counts are printed and recorded in the output manifest.
The output directory includes both figures, the city-level summary CSV, and
a manifest recording provenance and the decomposition check; it is never
overwritten.

## City-level analysis workflow

The preprocessing pipeline keeps the downloaded CSVs unchanged and creates:

```text
data/
├── raw/
│   ├── urban/
│   ├── rural/
│   ├── rural_all/
│   ├── GHSL_bounds_shp/
│   └── Beck_KG_V1_present_0p0083.tif
├── parquet/
│   ├── urban/
│   └── rural/
└── processed/
│   ├── city_metadata.parquet
│   ├── suhi_monthly.parquet
│   ├── suhi_seasonal.parquet
│   └── suhi_annual.parquet
```

The existing `data/city` and `data/rural_1700` CSVs were copied byte-for-byte
into the canonical raw folders; the original directories and ZIP archives are
left untouched.

The old priority-only rural CSVs remain in `data/raw/rural`. The analysis
config now reads the replacement all-city exports from
`data/raw/rural_all`, so the two datasets are not mixed or mistaken for
duplicate years.

Input directories, output file paths, and Parquet compression are set in
[`config.json`](./config.json). Relative paths in that file are resolved from
the config file's directory, so they can be changed without editing the
Python scripts. It also contains the quality thresholds used by the analysis:
each urban and rural slot must have at least 5 valid pixels, a spatial-mean
clear-day count of 4, and a spatial-mean valid 8-day composite count of 1.
Both urban and rural values must meet all three thresholds for that slot.
Failed monthly values are set to null in the analysis panel, while the original
QC fields and per-slot pass flags are retained. These are provisional starting
thresholds and are applied together.

Run the steps from the repository root:

```bash
python scripts/analysis/01_convert_exports.py
python scripts/analysis/02_build_monthly_suhi.py
python scripts/analysis/03_eda.py
```

The first step validates one row per city/year/month, converts numeric columns,
preserves empty measurements as nulls, and writes one compressed Parquet file
per source year. The second step joins rural rows to urban rows on
`ID_UC_G0`, `year`, and `month`, then calculates each monthly SUHI as urban
mean LST minus rural mean LST for Terra/Aqua day/night.
It also reads the configured local city-boundary shapefile, repairs invalid
geometries when necessary, calculates polygon centroids in the source
projected CRS, then transforms those points to EPSG:4326. It samples
`Beck_KG_V1_present_0p0083.tif` at each centroid and writes the lookup fields
`ID_UC_G0`, `centroid_lat`, `koppen_geiger_code`, `koppen_geiger_class`, and
`koppen_geiger_macro_group` to `data/processed/city_metadata.parquet`. The
climate raster is the Beck et al.
present-day Köppen-Geiger map at approximately 1-km resolution
(1980–2016; EPSG:4326). Raster code 0 is an unclassified/no-data pixel and is
stored as null in all climate columns. The lookup is merged by `ID_UC_G0`
into the monthly, seasonal, and annual analysis Parquets; class coverage is
reported. Raw CSVs, the source shapefile, and the climate raster are not
modified.

The configured `cz_macro_groups` maps all 30 raster class codes to one
macroclimate group. Its derived field, `koppen_geiger_macro_group`, is stored
alongside the original class code and label; a code-0/no-data centroid has null
values in all three climate fields.

If the three SUHI Parquets already exist, run `04_merge_city_metadata.py` to
rebuild the configured metadata lookup and refresh all city metadata columns
in the monthly, seasonal, and annual files, without rebuilding yearly source
Parquets. This merge is repeatable and replaces old centroid or climate values
from the authoritative shapefile and raster.

To create or refresh the city lookup without rebuilding SUHI data, run:

```bash
python scripts/analysis/build_city_metadata.py
```

To refresh the metadata and also merge it into existing SUHI files, run:

```bash
python scripts/analysis/04_merge_city_metadata.py
```

`suhi_monthly.parquet` is the master analysis table. In addition to the four
slot-level SUHI columns (`suhi_terra_day_mean`, `suhi_terra_night_mean`,
`suhi_aqua_day_mean`, and `suhi_aqua_night_mean`), it includes:

```text
A_T = suhi_terra_day_mean - suhi_terra_night_mean
A_A = suhi_aqua_day_mean - suhi_aqua_night_mean
M_T = (suhi_terra_day_mean + suhi_terra_night_mean) / 2
M_A = (suhi_aqua_day_mean + suhi_aqua_night_mean) / 2
```

Each derived value requires both corresponding day and night SUHI values to
pass QC. Seasonal and annual versions are equal-month averages of the derived
monthly values, so each uses months with valid paired day/night observations.

Seasonal groups use fixed meteorological months: DJF, MAM, JJA, and SON.
December belongs to the following season year. Seasonal and annual means give
each available monthly value equal weight. Outputs include valid-SUHI-month
counts for each slot and derived metric, along with completeness flags. A
season is marked QC-pass per metric with at least 2 valid months; a year
requires at least 9. Boundary DJF groups remain present and are separately
marked incomplete when fewer than three calendar months are available.
Seasonal and annual values are null when their per-metric minimum number of
QC-passing months is not reached.
LST mean columns are already in degrees Celsius in the GEE exports, so no
additional unit conversion is applied here.

The first EDA in `03_eda.py` plots daytime SUHI against nighttime SUHI for
Terra and Aqua in separate 12-month panel figures. Each point is one city;
the x/y values are medians across QC-passing observations for that city and
calendar month from 2000 through 2020. Terra uses `qc_pass_A_T`, Aqua uses
`qc_pass_A_A`. The script prints monthly city and city-month coverage counts
and saves PNGs under `results/figures/eda`. The year range, aggregation
statistic, output directory, and figure resolution are configured in
`config.json`. Both axes are centered on zero, with the x=y reference line
retained to make the quadrants and day/night differences clear.

The same command also creates a second view of yearly variation: for each
calendar month it saves one figure per sensor, with one panel per year in the
configured range (2000–2020 inclusive). Each point is that year's QC-passing
city observation; there is no cross-year aggregation in these panels. Yearly
figures are grouped under `results/figures/eda/by_year/terra` and
`results/figures/eda/by_year/aqua`. Axes are shared across months, years, and
sensors, centered at zero, and include the x=y reference line. The script
prints the passing city counts and coverage for every sensor/month/year.
All `03_eda.py` output is also written to a new timestamped log file in the
repository-root `logs/` directory. The directory is created automatically;
`01_convert_exports.py` and `02_build_monthly_suhi.py` still log to the
terminal only.

`03_eda.py` runs all seven EDA modules. Each can also be run independently:

```bash
python scripts/analysis/eda_calendar_months.py
python scripts/analysis/eda_yearly_panels.py
python scripts/analysis/eda_local_season.py
python scripts/analysis/eda_yearly_trends.py
python scripts/analysis/eda_city_slopes.py
python scripts/analysis/eda_climate_month_trends.py
python scripts/analysis/eda_seasonal_contrasts.py
```

The local-season module uses `centroid_lat` to group cities into Northern
Hemisphere (`> 10°`), Southern Hemisphere (`< -10°`), and Equatorial Belt
(`-10°` through `10°`, inclusive). The band width is set by
`eda.equatorial_latitude_band_degrees` in `config.json`. Equatorial and
northern cities retain their calendar months; southern cities shift by six
months. The aligned year advances when this shift crosses December, so July–
December in a southern source year map to January–June of the next aligned
year.

Local-season plots are separated by latitude group instead of mixing groups
with colors: 6 monthly summary figures (two sensors × three groups) are saved
under `results/figures/eda/local_season/<latitude_group>/`. These summarize
each city's configured mean or median across the source-year range. Another 72
year-panel figures (two sensors × three groups × 12 months) use only complete
aligned years; for a 2000–2020 source range, those are 2001–2020. They are
saved under `results/figures/eda/local_season/by_year/<latitude_group>/`.
Groups and sensors use common axis limits, with axes centered at zero and the
`day = night` reference line retained. Per-group city counts are logged for
the month summaries, and per-year QC counts are logged for year panels.

The yearly trend EDA focuses on compact annual summaries rather than city
scatter clouds. It uses the stored city-month `A_T`/`A_A` and `M_T`/`M_A`
alongside daytime and nighttime SUHI, and filters each sensor with its paired
day/night QC flag. For each calendar month, year, sensor, and latitude group,
it plots the cross-city median D, N, A, and M. A and M are read directly from
the stored columns. Bootstrap resampling keeps each city's D/N/A/M values
together; the shaded bands show the configured confidence interval. These are
annual cross-sectional intervals, not confidence
intervals for a fitted temporal slope. City sample-count ranges are shown in
each month panel and logged.

Each trend figure is a 3×4 month grid with year on the x-axis. Six figures
(two sensors × three latitude groups) are written under
`results/figures/eda/yearly_trends/<sensor>/<latitude_group>/`. Axes share one
symmetric range across the six figures. Bootstrap repetitions, confidence
level, and random seed are configured by `bootstrap_replicates`,
`bootstrap_confidence_level`, and `bootstrap_random_seed` in `config.json`.

The city-slope EDA fits a separate ordinary least-squares line over years for
each city, calendar month, and sensor, using only that sensor's paired
day/night QC-passing observations. Its slopes are plotted as (βD, βN) in
°C/decade. A city-month is included only when it has at least
`min_city_slope_years` valid years (currently 15). It writes six figures
(two sensors × three latitude groups); each figure has one panel per calendar
month. Figures remain separated into Northern Hemisphere, Southern
Hemisphere, and Equatorial Belt, while points are colored by the configured
Köppen-Geiger macroclimate group. Cities without a classified climate are
shown in gray. The panels share symmetric axes across all six figures, include
zero crosshairs and a βD=βN reference. City counts and exclusions for
insufficient valid years are logged. The input SUHI table must include
`koppen_geiger_macro_group`; run `04_merge_city_metadata.py` to add or refresh
the metadata.

For less crowded comparisons, the same run also writes a climate-separated
version: each figure contains one climate group and one latitude group, with
the same 12 monthly panels and shared axis limits. These additional figures
are saved under
`results/figures/eda/city_slope_scatter/by_climate/<sensor>/<climate_group>/<latitude_group>/`.
The climate group has one consistent color in this version.
Run this module independently with:

```bash
python scripts/analysis/eda_city_slopes.py
```

The climate × month trend heatmaps use the configured Köppen-Geiger macro
groups and the same minimum-year rule as the city-slope plots. For each city,
sensor, and month, OLS slopes for D and N are fitted from paired-QC annual
observations; the slope of stored A (`A_T`/`A_A`) is fitted on the same paired
observations. Southern cities shift by six months and advance the aligned year
when crossing December. Northern and Equatorial Belt cities keep calendar
months. All cities are then combined by macroclimate group and aligned
local-season month.

For every climate group × local month cell, each of βD, βN, and βA is
summarized as the median of eligible city slopes, with a percentile bootstrap
CI across cities. Eligibility requires `min_city_slope_years` valid years.
The asterisk is shown only when at least two cities contribute and the
configured CI excludes zero.
There are six heatmaps (two sensors × three slope metrics); each shows the
median slope, city count, and the asterisk rule above. Bootstrap settings are
shared with the annual trend EDA. A long-form CSV containing cell medians, CI
bounds, city counts, confidence level, and bootstrap replicate count is saved
alongside the figures under
`results/figures/eda/climate_month_trend_heatmaps/`.
Run the module independently with:

```bash
python scripts/analysis/eda_climate_month_trends.py
```

The warm/cold seasonal-contrast EDA addresses two related questions: whether
the seasonal contrast of daytime/nighttime SUHI is changing, and how much of
that change accompanies seasonal changes in urban versus rural LST. Northern
summer and winter are configured as June–August and December–February;
Southern Hemisphere assignments are reversed. December is assigned to the
following season year. Each city-month gets derived integer `is_summer` and
`is_winter` flags; shoulder months have both flags set to 0. The flags are
written to a separate Parquet sidecar and do not modify the source monthly
SUHI file. Cities in the configured equatorial band (currently ±10°) are
retained in that sidecar with both flags 0, but excluded from the warm/cold
season analysis.

The seasonal calculation uses paired day/night QC-passing observations and
requires all three months in each season by default. It writes one row per
city × season year × season containing Terra/Aqua daytime, nighttime, and
day–night contrast values for urban LST, rural LST, and SUHI. It then
calculates each city's annual warm-minus-cold contrast and fits a city-level
OLS trend (γ, °C/decade) when at least `min_city_slope_years` complete paired
years are available. For each city and metric, the decomposition is
`γ_SUHI = γ_urban − γ_rural`, using the same months and years.

An additional figure estimates the urban and rural LST trends separately for
warm and cold seasons. It uses the same complete paired years for both seasons
within each city, so the two slopes can be compared directly to identify
whether a changing seasonal contrast is associated with warm-season warming,
cold-season cooling, or both. Points show climate-group median city slopes and
city-bootstrap confidence intervals; the six panels separate Terra/Aqua and
D/N/A.

Climate-group summaries include median γ and a city-bootstrap confidence
interval, along with the shares of eligible cities having positive, negative,
or zero γ. A median direction is labeled increasing/decreasing only when
there are at least two cities and its configured confidence interval excludes
zero; the city shares are descriptive and are not counts of individually
significant city trends. `All cities` includes eligible classified and
unclassified cities outside the equatorial band. This decomposition describes
associations and is not by itself a causal attribution.

Tables and figures are saved under
`results/figures/eda/warm_cold_seasonal_contrasts/`: city-month season labels
(Parquet), city-year-season values, annual warm-minus-cold contrasts,
city-level γ estimates, climate-group summaries (CSV), median-trend CI
figures, positive-city-share figures, and the separate warm/cold urban/rural
LST trend figure.
