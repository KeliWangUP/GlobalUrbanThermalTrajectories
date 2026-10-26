# Analysis data preparation

For the complete source-to-analysis workflow and future analysis roadmap, see
[`../../DATA_PROCESSING_WORKFLOW.md`](../../DATA_PROCESSING_WORKFLOW.md).

The preprocessing pipeline keeps the downloaded CSVs unchanged and creates:

```text
data/
├── raw/
│   ├── urban/
│   ├── rural/
│   ├── rural_all_cities/
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
`data/raw/rural_all_cities`, so the two datasets are not mixed or mistaken for
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

`03_eda.py` runs all six EDA modules. Each can also be run independently:

```bash
python scripts/analysis/eda_calendar_months.py
python scripts/analysis/eda_yearly_panels.py
python scripts/analysis/eda_local_season.py
python scripts/analysis/eda_yearly_trends.py
python scripts/analysis/eda_city_slopes.py
python scripts/analysis/eda_climate_month_trends.py
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
the metadata. Run this module independently with:

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
