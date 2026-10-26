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
│   └── GHSL_bounds_shp/
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
It also reads the configured local city-boundary shapefile, calculates each
polygon centroid in the source projected CRS, transforms that point to
EPSG:4326, and writes an `ID_UC_G0`/`centroid_lat` lookup table to
`data/processed/city_metadata.parquet`. The lookup is merged by `ID_UC_G0`
into the monthly, seasonal, and annual analysis Parquets. Raw CSVs and the
source shapefile are not modified. Invalid polygon geometries are repaired
with `make_valid` for centroid calculation, and the repaired count is printed.
If the three SUHI Parquets already exist without `centroid_lat`, run
`04_merge_city_metadata.py` to enrich them in place without rebuilding from
yearly source files. This merge is repeatable and replaces any old
`centroid_lat` value with the configured metadata lookup.

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

`03_eda.py` runs all four EDA modules. Each can also be run independently:

```bash
python scripts/analysis/eda_calendar_months.py
python scripts/analysis/eda_yearly_panels.py
python scripts/analysis/eda_local_season.py
python scripts/analysis/eda_yearly_trends.py
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
scatter clouds. It uses the already stored city-month `A_T` and `A_A` values
alongside daytime and nighttime SUHI, and filters each sensor with its paired
day/night QC flag. For each calendar month, year, sensor, and latitude group,
it plots the cross-city median D, median N, and median A. A percentile
bootstrap resamples cities within each year/month/group, keeping each city's
D/N/A values together; the shaded bands show the configured confidence
interval. These are annual cross-sectional intervals, not confidence
intervals for a fitted temporal slope. City sample-count ranges are shown in
each month panel and logged.

Each trend figure is a 3×4 month grid with year on the x-axis. Six figures
(two sensors × three latitude groups) are written under
`results/figures/eda/yearly_trends/<sensor>/<latitude_group>/`. Axes share one
symmetric range across the six figures. Bootstrap repetitions, confidence
level, and random seed are configured by `bootstrap_replicates`,
`bootstrap_confidence_level`, and `bootstrap_random_seed` in `config.json`.
