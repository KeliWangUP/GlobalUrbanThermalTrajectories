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

## City metadata

Build the metadata table from the configured GHSL urban-boundary shapefile and
Köppen-Geiger raster with:

```bash
python scripts/analysis/build_city_metadata.py
```

In addition to `ID_UC_G0` and Köppen fields, the output contains `centroid_lon`
and `centroid_lat` in EPSG:4326, `city_area_km2` from GHSL's `GC_UCA_KM2`,
`country` from `GC_CNT_GAD`, and the original `GC_DEV_USR` region value. The
area is the GHSL urban-centre area in km²; country and region values are
carried from the source attributes. It also joins
`ghsl_built_city_development_2000_2020_v0.3.csv` by `ID_UC_G0`, storing the
2000 built-surface and built-volume sums plus their OLS slopes over the five
years 2000, 2005, 2010, 2015, and 2020. The output contains only the 2000
baseline and annual slopes (`m²/year` and `m³/year`), not the intermediate
year values: `built_surface_2000_m2`, `built_surface_slope_m2_per_year`,
`built_volume_total_2000_m3`, and
`built_volume_total_slope_m3_per_year`. Built surface is kept distinct from
ISA; this CSV does not contain a separate ISA field.

To refresh the metadata join in the existing monthly, seasonal, and annual
SUHI Parquets after rebuilding the metadata table, run:

```bash
python scripts/analysis/04_merge_city_metadata.py
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
`data/processed/gold/city_month_trends_stable_development_2000_isa_0p7`. It
creates:

- `city_month_paired_qc.parquet`: matched city/month/sensor-slot observations,
  QC flags, urban/rural means, and SUHI.
- `city_pixel_classification.parquet`: each pixel's initial ISA fraction,
  stable/development label, and water-eligibility flag.
- `city_month_zone_paired_qc.parquet`: matched monthly records for stable,
  development, and overall urban zones, with rural observations and SUHI.
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

The zone classification uses the configured initial ISA fraction column
(`isa_urban_fraction_2000`) and `urban_classification.stable_isa_threshold`
(default `0.7`); pixels below the threshold are labeled development. The
water-eligible mask is applied before the urban-zone monthly aggregates are
computed. Stable, development, and overall zone outputs are built in this
same Gold run; there is no separate zone-generation command.

### Seasonal trends and γ by urban pixel zone

After building the Gold dataset, calculate seasonal means and trends for the
stable, development, and overall urban zones, along with rural and SUHI trends:

```bash
python scripts/analysis/build_city_season_zone_trends.py
```

[`city_season_zone_trends_config.json`](./city_season_zone_trends_config.json)
reads `city_month_zone_paired_qc.parquet` and the manifest from the Gold output
directory above. It verifies the ISA classification settings against the Gold
manifest. December is assigned to the following year's DJF season; valid
monthly city means are equally weighted, with at least two valid months by
default. Trends are in °C per decade. γ is summer-minus-winter: JJA−DJF in the
Northern Hemisphere and DJF−JJA in the Southern Hemisphere; equatorial cities
are excluded.

To compare γ across urban zones and Köppen subclasses:

```bash
python scripts/analysis/plot_city_season_zone_gamma.py
```

[`city_season_zone_gamma_figures_config.json`](./city_season_zone_gamma_figures_config.json)
controls the Köppen major group, optional subclasses, hemisphere (`all`,
`north`, or `south`), and output directory. The figures compare zone-level
γSUHI distributions and plot γR versus γU with y=x as the synchrony reference.

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
plots A/B/C/D; add E to `climate_groups` to include polar cities. The
`hemisphere` setting can be `all`, `north`, or `south`. It defaults to `all`
to preserve the pooled analysis; choosing north or south filters the city-level
trend rows before climate summaries and plotting. Equatorial and missing-
hemisphere rows are excluded from north/south runs. The selected hemisphere,
source and selected city/row counts, and exclusions are recorded in the
manifest and run log.

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

### City/season/slot trend Gold table and figures

Build city-level seasonal trends from `suhi_seasonal.parquet` with:

```bash
python scripts/analysis/build_city_season_slot_trends.py
```

The source path, year window, minimum number of valid seasons, and output
directory are set in
[`city_season_slot_trends_config.json`](./city_season_slot_trends_config.json).
By default, the script fits 2003–2019 OLS slopes separately for each city,
season (DJF/MAM/JJA/SON), and Terra/Aqua day/night slot. It requires the
season-specific QC flag and finite urban, rural, and SUHI means; the minimum
valid-year rule defaults to 15. The trend table includes β and standard error
for urban, rural, and SUHI LST, plus centroid longitude/latitude, GHSL area,
country, `GC_DEV_USR` region, climate, hemisphere, and the number of valid
seasonal observations. It also retains GHSL's 2000 built-surface and built-
volume baselines and their 2000–2020 OLS slopes. Run
`04_merge_city_metadata.py` first so those GHSL features are present in
`suhi_seasonal.parquet`; the default output directory is versioned with
`_with_ghsl` to preserve the existing trend product.

Generate the matching climate summaries and figures with:

```bash
python scripts/analysis/plot_city_season_slot_trends.py
```

[`city_season_slot_trend_figures_config.json`](./city_season_slot_trend_figures_config.json)
controls the trend input, figure output, climate major groups, and
`hemisphere` (`all`, `north`, or `south`). Figure 1 displays each climate
group's seasonal SUHI slope median, IQR, and city count; Figure 2 displays the
urban, rural, and SUHI slope distributions. Both separate Terra/Aqua and
day/night. The city-level βSUHI = βU − βR identity is checked before
aggregation, while climate-group medians are reported independently. These
are the source table's fixed DJF/MAM/JJA/SON seasons: selecting a hemisphere
filters cities but does not shift seasons to local winter or summer. Missing
climate rows are excluded from climate-group plots and counted in the
manifest. Existing output directories are never overwritten.

### Hemisphere-aware summer-minus-winter trend contrasts

After building the city/season/slot trend table with
`build_city_season_slot_trends.py`, calculate city-level seasonal contrasts
with:

```bash
python scripts/analysis/build_city_seasonal_trend_contrasts.py
```

The input trend Parquet and output directory are configured in
[`city_seasonal_trend_contrasts_config.json`](./city_seasonal_trend_contrasts_config.json).
For each city and Terra/Aqua day/night slot, the script computes summer β
minus winter β. It uses JJA−DJF in the Northern Hemisphere and DJF−JJA in the
Southern Hemisphere, then computes `gamma_SUHI`, `gamma_urban`, and
`gamma_rural`. The city-level identity
`gamma_SUHI = gamma_urban − gamma_rural` is checked before publishing.
Equatorial cities are excluded because this hemisphere-based summer/winter
definition does not apply to them. City/slot records missing either required
seasonal trend are excluded and counted in the manifest. The output is one
row per complete city/slot pair, with the selected seasons, each seasonal β,
and each γ; no output directory is overwritten.

### Seasonal γ contrasts by Köppen subclass

Visualize the city-level summer-minus-winter contrast table with:

```bash
python scripts/analysis/plot_city_seasonal_gamma_subclimate_trends.py
```

[`city_seasonal_gamma_subclimate_figures_config.json`](./city_seasonal_gamma_subclimate_figures_config.json)
selects the γ Parquet, Köppen major group, optional subclasses, and hemisphere
(`all`, `north`, or `south`). Figure 1 shows each subclass and Terra/Aqua
day/night slot's city-level γSUHI median, IQR, and city count. Figure 2 shows
the corresponding γUrban, γRural, and γSUHI distributions. The city-level
identity γSUHI = γUrban − γRural is checked before summarization; subclass
medians are calculated independently. Figures 3 and 4 show city-level
γUrban-versus-γRural scatterplots, colored by Köppen subclass and GHSL
`GC_DEV_USR` region, respectively. Both retain the configured hemisphere and
subclass selection and use y=x as the synchrony reference; vertical distance
from the line is γSUHI. Figures 5 and 6 relate each city/slot's γUrban, γRural,
and γSUHI to the GHSL built-surface and built-volume OLS slopes, respectively;
points are colored by GHSL `GC_DEV_USR` region. The
`building_growth_axis_limits` setting in the figure config sets optional
display bounds independently for surface (km²/year) and volume (million
m³/year). Its default surface range is 0–0.5 and volume is unbounded; this only
clips the displayed x-axis, so cities outside the bounds remain in the data
and are counted. The figures use the same configured climate and hemisphere
filters. The output directory includes all six figures, a summary CSV, and a
provenance manifest. Re-running replaces files generated by this script while
preserving other files in the output directory. On POSIX
servers, generated files use group-readable/writable permissions (files `664`,
directories `2775`); the output directory's group must be shared with the SFTP
account.

### Seasonal trends by Köppen subclass within one major climate group

To compare the smaller Köppen classes within one selected major climate group,
run:

```bash
python scripts/analysis/plot_city_season_subclimate_trends.py
```

Configure `major_climate_group` as A–E in
[`city_season_subclimate_trend_figures_config.json`](./city_season_subclimate_trend_figures_config.json).
By default, an empty `subclimate_classes` list automatically includes all
subclasses present in the trend input for that major group; set it to a list
such as `["Cfa", "Cfb"]` to focus on selected subclasses. The same
`hemisphere` switch (`all`, `north`, or `south`) is applied before climate
subclass summaries. The generated pair of figures uses the seasonal SUHI
median/IQR/city-count and urban–rural decomposition views described above,
with each subplot row representing one Köppen subclass. A CSV summary and
manifest record the selected climate group, subclasses, hemisphere, and
city-level decomposition check. This plots an existing seasonal trend table;
it does not re-fit the trends.

### City/slot annual trend Gold table and figures

Build city-level annual trends from `suhi_annual.parquet` with:

```bash
python scripts/analysis/build_city_annual_slot_trends.py
```

The source path, year window, minimum number of valid years, and output
directory are set in
[`city_annual_slot_trends_config.json`](./city_annual_slot_trends_config.json).
By default, the script fits 2003–2019 OLS slopes separately for each city and
Terra/Aqua day/night slot. It requires the annual slot-specific QC flag and
finite urban, rural, and SUHI means. The output includes β, standard error,
climate, hemisphere, latitude, and valid-year count.

Generate annual figures grouped by Köppen subclass with:

```bash
python scripts/analysis/plot_city_annual_subclimate_trends.py
```

[`city_annual_subclimate_trend_figures_config.json`](./city_annual_subclimate_trend_figures_config.json)
selects the trend table, output directory, major climate group (`A`–`E`),
optional subclasses (an empty list includes all available subclasses), and
hemisphere (`all`, `north`, or `south`). Figure 1 shows annual SUHI slope
medians, IQRs, and eligible-city counts; Figure 2 shows the urban, rural, and
SUHI slopes separately for each Terra/Aqua day/night slot. City-level
βSUHI = βU − βR is checked before aggregation; climate-subclass medians are
computed independently. The output directory includes both figures, a CSV
summary, and a provenance manifest, and is never overwritten.

### Paired-month city-level SUHI trend comparisons

Run the requested D-February/July and C-April/July paired comparisons with:

```bash
python scripts/analysis/paired_month_city_trends.py
```

The source trend Parquet and output directory are configured in
[`paired_month_city_trends_config.json`](./paired_month_city_trends_config.json).
For each comparison and Terra Day/Aqua Day separately, only cities with
city-level trend rows in both compared months and whose Köppen class starts
with the requested group are paired. The delta is defined as
`beta_suhi_later_month - beta_suhi_earlier_month` (July minus February for D;
July minus April for C). The summary reports paired-city count, median delta,
positive/zero/negative counts, and the positive fraction. Exact-zero deltas
remain in the denominator of the positive fraction. A second CSV contains
the city-level paired slopes and deltas for auditing; a manifest records input
provenance and definitions.

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
