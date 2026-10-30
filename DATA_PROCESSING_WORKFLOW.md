# DNA-SUHI：Day–Night Asymmetry of Surface Urban Heat Islands

**DNA-SUHI** 是本项目的研究名称：DNA 代表 **D**ay–**N**ight
**A**symmetry（日夜不对称），SUHI 则指地表城市热岛。名称强调项目同时研究
城市热岛的白天与夜间特征，以及两者之间的不对称差异；它也对应主分析表中的
`A_T`、`A_A` 日夜差异指标和 `M_T`、`M_A` 日夜平均指标。

本文记录 DNA-SUHI 从源数据、处理步骤到后续分析的主线。

> 本文是项目端到端数据流程的主记录。已经实现或运行过的步骤与建议中的后续分析分开标记；配置或处理逻辑变化时，应同步更新本文。  
> 最近核对：2026-10-26

## 1. 研究数据与核心对象

研究期由 GEE 配置控制；当前配置为 2000–2024 年。
城市边界使用
`ID_UC_G0` 标识，并固定采用 2024 年边界。热红外数据来自 MODIS 8 日合成：

- Terra：`MODIS/061/MOD11A2`
- Aqua：`MODIS/061/MYD11A2`
- 四个观测槽：Terra day/night、Aqua day/night

处理网格的投影和像元尺度从
`MODIS/061/MOD11A1` 的 `LST_Day_1km` 波段读取；实际月温度统计仍使用上述
MOD11A2/MYD11A2 8 日产品。

主分析对象是城市—月份面板。城乡温度使用相同 MODIS 网格与观测时段，
先分别汇总城市区和对应乡村参考区的温度，再计算：

```text
SUHI = T_urban - T_rural
A_T  = SUHI_TerraDay - SUHI_TerraNight
A_A  = SUHI_AquaDay  - SUHI_AquaNight
M_T  = (SUHI_TerraDay + SUHI_TerraNight) / 2
M_A  = (SUHI_AquaDay  + SUHI_AquaNight) / 2
```

其中 `A_*` 表示日夜 SUHI 差异，`M_*` 表示日夜 SUHI 平均值。
这些派生量只有在对应日间和夜间 SUHI 均通过质量控制时才有效。

## 2. 端到端流程图

```text
城市边界 + MODIS + GAIA/GLO30
                  │
                  ├─ 城市区域 LST 月统计 ───────────────┐
                  │                                      │
                  └─ rural halo → fraction → rural LST ─┤
                                                         ↓
                  Google Drive annual CSV → data/raw（原始文件保留）
                                                         ↓
                  01_convert_exports.py → 年度 clean Parquet
                                                         ↓
                  02_build_monthly_suhi.py → 月度 master panel
                                           ├→ 季节面板
                                           └→ 年度面板
                                                         ↓
                  EDA、趋势、异质性与驱动因素分析（待开展）
```

## 3. Earth Engine 数据生成

GEE 输入、资产、研究年份、产品波段、筛选阈值及导出目标统一配置在
[`scripts/gee_modul/config.json`](./scripts/gee_modul/config.json)。

### 3.1 城市温度

城市边界资产为 `GHSL_urban_bounds_2024`，城市键为 `ID_UC_G0`。
对每个城市和月份，在 MODIS 参考网格上统计四个温度槽的空间
`mean`、`median`、`stdDev`、`count`，并记录晴空观测数和有效 8 日合成数的
空间均值。城市导出入口是
[`export_city_monthly_lst.py`](./scripts/gee_modul/export_city_monthly_lst.py)；
它通过共享导出工具按年提交 CSV。

### 3.2 乡村参考区与 fraction

乡村参考区先由
[`build_rural_halos_asset.py`](./scripts/gee_modul/build_rural_halos_asset.py)
构建：

1. 对城市边界构建 2–10 km 环带。
2. 从环带中排除附近其他城市的 2 km 内缓冲区。
3. 去除空几何；最终无有效 rural halo 的城市不能贡献乡村温度。

接着，
[`build_rural_city_fraction_assets.py`](./scripts/gee_modul/build_rural_city_fraction_assets.py)
为每个城市计算 MODIS 像元内适宜乡村地表的比例。当前适宜性掩膜由以下条件
共同决定：

- GAIA `change_year_index == 0`；
- GLO30 地形坡度低于 5°；
- GLO30 水体掩膜为非水体。

至少 80% 为适宜乡村地表的 MODIS 像元才进入 rural LST 汇总。少数大城市
（当前配置为 5929、6090）的 fraction 资产可按 MODIS 对齐 subtile 分块，
这是 fraction 计算的内部分块；温度导出仍是按年度汇总，不是 tile 批次温度
导出。fraction builder 会跳过已经存在的资产，可在任务分批完成后重跑。

### 3.3 月合成、QC 字段和年度 CSV

MODIS 8 日合成按合成时间中点分配到日历月份；每个像元的月 LST 以该合成的
晴空观测日数加权。LST 在 GEE 中由比例因子转换为摄氏度。当前
`modis.qa.apply_qa_mask` 为 `false`：不额外按 MODIS QC 位筛选，但仍要求
LST 可用且至少有晴空观测；后续分析再用导出的 QC 字段筛除低质量城乡月份。

每个城市—年—月记录包含四个槽的 LST 空间统计及 QC 字段，包括：

- `*_count`：有效温度像元数；
- `*_clear_days_mean`：像元晴空日数的空间均值；
- `*_n8day_mean`：有效 8 日合成数的空间均值。

乡村全量年度导出入口为
[`export_rural_all_cities_annual.py`](./scripts/gee_modul/export_rural_all_cities_annual.py)。
它只处理有非空 halo 且 fraction 资产齐全的城市，每年一个 CSV。当前 GEE 配置
将 `rural_city_ids` 设为 `null`（所有符合条件的城市），输出数据集名为
`rural_lst_all_cities`，Drive 文件夹为 `GlobalUrbanThermalTrajectories`。
运行前应检查 fraction 资产完整性，并确认导出范围和年份。

优先城市点的要素数不等于最终城市数：优先队列按与点相交的城市 polygon
`ID_UC_G0` 去重；之后还会排除没有非空 rural halo 的城市。新全量导出同样
以有非空 halo 的城市为准，而不是以优先点数量或旧的 1,700 城结果为准。

fraction builder 会跳过已经存在的城市资产。若 GAIA/DEM、坡度、缓冲区或
fraction 阈值等定义变化，应使用新的目标资产版本/集合，避免旧资产被静默复用；
资产版本切换和旧资产清理应单独记录并谨慎执行。

常用入口：

```bash
# 已有 halo 资产时跳过此步；需要重建 halo 时运行
python scripts/gee_modul/build_rural_halos_asset.py

# 提交尚缺的逐城市 fraction 资产任务；等待任务完成后按需重跑
python scripts/gee_modul/build_rural_city_fraction_assets.py

# 按当前 config.json 的 export.targets 提交导出任务
python scripts/gee_modul/run_export.py
```

`run_export.py` 的目标由 `export.targets` 控制。当前配置是
`["samplepoint_attributes"]`，会提交样点属性批次；若要导出全量乡村温度，
将目标改为 `["rural"]`；
需要重新导出城市温度时将目标改为 `["city"]`。全量乡村年度导出也可直接运行：

```bash
python scripts/gee_modul/export_rural_all_cities_annual.py
```

导出任务提交成功不等于任务完成。等待 Earth Engine 任务成功后，再从 Drive
下载年度 CSV。全量乡村文件放入 `data/raw/rural_all`；不要与旧优先
城市文件 `data/raw/rural` 混放，否则会造成同一年有多个输入文件。

### 3.4 样点 LST 空间分区试验

全球样点按年或按月整体导出均已失败；现提供按 MODIS 网格坐标分区的
试验入口
[`export_samplepoint_lst_spatial.py`](./scripts/gee_modul/export_samplepoint_lst_spatial.py)。
640 和 1280 像元网格各有一个 2020 年任务已完成；当前测试网格边长
为 640 像元，仍仅提交一个包含测试城市某个样点的网格；
当前城市范围是历史乡村结果中实际导出的 1,700 个城市，先按城市 ID 过滤，
再划分网格；该网格中属于其他优先城市的样点也会被导出。一个任务覆盖配置的
整个研究期，当前测试配置为 2000–2024 年。月合成、质量字段和输出表头沿用
既有样点 LST 逻辑。网格大小只是试验起点，成功后仍需对样点密集区验证。

分区设置、全量提交方式及任务日志见
[`scripts/gee_modul/README.md`](./scripts/gee_modul/README.md)。

## 4. 本地数据层级

```text
data/
├── raw/                          # 下载的原始 CSV，保留不改写
│   ├── urban/
│   ├── rural/                    # 旧优先城市数据，非当前全量输入
│   ├── rural_all/                # 当前配置的乡村全量输入目录
│   ├── GHSL_bounds_shp/          # 城市边界 Shapefile（原始投影）
│   └── Beck_KG_V1_present_0p0083.tif  # Köppen-Geiger 现状气候分类
├── parquet/                      # 按年清洗后的 Parquet
│   ├── urban/
│   └── rural/
└── processed/
    ├── city_metadata.parquet     # city centroid + Köppen-Geiger lookup
    ├── suhi_monthly.parquet      # master analysis table
    ├── suhi_seasonal.parquet
    └── suhi_annual.parquet
```

目录位置、输出文件名、压缩方式和分析 QC 阈值由
[`scripts/analysis/config.json`](./scripts/analysis/config.json) 统一控制；
相对路径以该配置文件所在目录为基准。原始 CSV 是输入档案，不由处理脚本覆盖。
历史优先城市原始数据与当前全量乡村数据分开保存。

## 5. 分析预处理

在仓库根目录执行。处理说明和脚本配置见
[`scripts/analysis/README.md`](./scripts/analysis/README.md)。

### 5.1 年度 CSV → clean Parquet

```bash
python scripts/analysis/01_convert_exports.py
```

[`01_convert_exports.py`](./scripts/analysis/01_convert_exports.py) 从配置指定的
urban 与 rural raw 目录读取年度 CSV，并写出 `urban_YYYY.parquet` 和
`rural_YYYY.parquet`。转换时检查字段结构、文件名与年份一致、每个城市每年
恰有 12 个月、城市—年—月键唯一、年份连续，以及 urban/rural 年份一致。
数值列转换失败会报错；空测量保留为空值。输入文件应按
`..._YYYY.csv` 命名。

### 5.2 clean Parquet → SUHI 面板

```bash
python scripts/analysis/02_build_monthly_suhi.py
```

[`02_build_monthly_suhi.py`](./scripts/analysis/02_build_monthly_suhi.py) 按年读取
城乡 Parquet，以 `ID_UC_G0 + year + month` 关联，并验证城乡测量字段结构一致。
每个槽的 QC 分别对 urban 和 rural 两侧同时检查：

如需先独立生成城市元数据（不依赖 SUHI 面板），可运行：

```bash
python scripts/analysis/build_city_metadata.py
```

此步骤还从 `paths.city_bounds_shapefile` 指向的本地 GHSL 2024 Shapefile
生成城市元数据。当前边界源文件采用 World Mollweide（ESRI:54009）投影；
程序先在原始投影下计算 polygon centroid，再将 centroid 点转换到 EPSG:4326，
避免直接在经纬度下求多边形质心。无效几何使用 `make_valid` 修复，并报告数量。
之后以质心经纬度在 `paths.koppen_geiger_raster` 配置的 Beck Köppen-Geiger 栅格
采样气候区。该栅格为 EPSG:4326、约 0.0083°（约赤道处 1 km）单波段分类图；
其现状分类基于 1980–2016。城市级 lookup 写入
`data/processed/city_metadata.parquet`，包含 `ID_UC_G0`、`centroid_lat`、
`koppen_geiger_code`（1–30 原始类别编号）和 `koppen_geiger_class`（如 Af、Cfa、
ET），以及按配置中 `cz_macro_groups` 派生的
`koppen_geiger_macro_group` 大类。大类覆盖全部 1–30 原始类别且每个编号只属于
一个大类；栅格像元值 0 表示无分类/无数据，三个气候字段在该城市保留为空。
程序报告有气候分类的城市数。已观测到 11,422 个边界城市中有 66 个质心采样到
0 类。

这份 metadata 按 `ID_UC_G0` many-to-one 合并到月度、季节和年度 SUHI Parquet。
若分析记录中的城市 ID 无法匹配边界，流程会报错，而不是留下空质心。运行
`04_merge_city_metadata.py` 时会根据配置的边界和栅格重新生成 metadata lookup，
并刷新现有三个 SUHI Parquet 中对应的城市元数据，不需要重新构建年度源 Parquet。
原始 Shapefile、CSV 和 TIFF 均不修改。气候分类数据来源为 Beck et al. 2018，
*Present and future Köppen-Geiger climate classification maps at 1-km resolution*,
[Scientific Data 5, 180214](https://doi.org/10.1038/sdata.2018.214)。

`centroid_lat` 可用于后续纬度分组；本项目当前 EDA 将 `-10°` 至 `10°`
（含边界）设为赤道纬度带，具体阈值见 `eda.equatorial_latitude_band_degrees`。
质心纬度只是城市面状要素的代表位置，跨赤道城市后续可单独识别，不应将纬度
分组解释为城市范围完全位于该区域。

| 分析配置项 | 当前阈值 | 含义 |
|---|---:|---|
| `min_valid_pixels_per_side_slot` | 5 | 城乡两侧各至少 5 个有效像元 |
| `min_clear_days_mean_per_side_slot` | 4.0 | 城乡两侧晴空日数空间均值均至少为 4 |
| `min_n8day_mean_per_side_slot` | 1.0 | 城乡两侧有效 8 日合成数空间均值均至少为 1 |
| `min_valid_months_per_season` | 2 | 季节聚合每项指标至少 2 个合格月 |
| `min_valid_months_per_year` | 9 | 年度聚合每项指标至少 9 个合格月 |

以上阈值是当前经验起点，不是最终定论。月度 QC 未通过时，对应城乡 LST 与
SUHI 置为空值；原 QC 字段和每槽 `qc_pass_*` 标记保留。派生指标要求对应的
昼夜两个槽都通过 QC。

`suhi_monthly.parquet` 是 master analysis table，包含城乡 LST、四个槽的 SUHI、
`A_T`、`A_A`、`M_T`、`M_A`、原始 QC 字段及通过标记。季节和年度值对合格月
做等权平均，不按日数或有效像元数加权；每项指标单独计算合格月数与 QC 标记。

- 季节采用固定气象季节：DJF、MAM、JJA、SON。
- 12 月归入下一 `season_year` 的 DJF。
- `is_complete_season` / `is_complete_year` 表示日历月份是否完整；边界 DJF
  可存在但会标记为不完整。
- 聚合指标未达到最低合格月数时置为空值。
- LST 已是摄氏度，不在分析阶段再次换算。

### 5.3 第一项 EDA：昼夜 SUHI 特征

```bash
python scripts/analysis/03_eda.py
```

[`03_eda.py`](./scripts/analysis/03_eda.py) 从月度 master table 读取数据，按
传感器分别输出一张 3×4 月份面板图：横轴为日间 SUHI，纵轴为夜间 SUHI，
每个点代表一个城市在该日历月份跨 2000–2020 年的中位数。Terra 使用
`qc_pass_A_T`，Aqua 使用 `qc_pass_A_A`，确保每个点的昼夜值来自同一组通过
对应传感器昼夜 QC 的年份。两张图采用相同坐标范围，并绘制 `day = night`
参考线；横纵轴均以 0 为中心，坐标轴在原点相交，从而显示正负 SUHI 象限。
若某城市在该月份至少有一个通过 QC 的年份，即纳入该城市点；每点可用年份数
也保存在聚合结果中。

运行时会逐传感器、逐月打印合格城市—年—月份记录数、唯一合格城市数、
通过率和作图城市数。图片保存到
`results/figures/eda/terra_day_vs_night_suhi_median_2000-2020.png` 和
`results/figures/eda/aqua_day_vs_night_suhi_median_2000-2020.png`。年份、
跨年聚合统计量、输出目录和 DPI 由
[`scripts/analysis/config.json`](./scripts/analysis/config.json) 中的 `eda`
和 `paths.eda_figure_dir` 设置。

如果 SUHI parquet 已经生成但没有 `centroid_lat`，可单独运行
`python scripts/analysis/04_merge_city_metadata.py`：它会将城市 lookup 合并到
月度、季节和年度文件，不必重跑年度 CSV 清洗与 SUHI 构建。该操作可重复执行，
每次都以配置的 `city_metadata.parquet` 为权威来源替换原有质心列。

`03_eda.py` 的运行信息会同时输出到终端和仓库根目录的 `logs/`，每次运行生成
一个 `03_eda_YYYYMMDD_HHMMSS_microseconds.log` 文件，不覆盖旧日志。日志目录由
`paths.eda_log_dir` 配置并在运行时自动创建。`01_convert_exports.py` 和
`02_build_monthly_suhi.py` 当前仍只向终端输出日志。

### 5.4 第二项 EDA：按年份比较每月 D/N 关系

同样通过 `python scripts/analysis/03_eda.py` 生成。对每个传感器和日历月份，
单独绘制一张年度分面图：2000–2020 年（含首尾，共 21 年）每年一个 panel，
每个点是该年该月一个通过该传感器昼夜 QC 的城市 SUHI 观测，不做跨年均值或
中位数聚合。共输出 24 张图（12 个月 × Terra/Aqua），分别放在
`results/figures/eda/by_year/terra/` 和
`results/figures/eda/by_year/aqua/`。

所有年份、月份和传感器使用一致的对称坐标范围；坐标轴在 `(0, 0)` 相交，
并保留 `day = night` 参考线，方便比较各年点云分布及日夜关系变化。每个 panel
标注通过 QC 的城市数，运行日志逐年打印合格城市—月份数、总记录数、通过率
及合格城市数。年份范围沿用分析配置中的 `eda.start_year` 和 `eda.end_year`。

### 5.5 第三项 EDA：local-season 分纬度组对齐

该 EDA 既可由 `03_eda.py` 一并运行，也可独立运行：

```bash
python scripts/analysis/eda_local_season.py
```

纬度组依据城市 `centroid_lat` 划分：北半球 `> 10°`、南半球 `< -10°`、
赤道纬度带 `-10°` 至 `10°`（含边界）。带宽由
`scripts/analysis/config.json` 的 `eda.equatorial_latitude_band_degrees`
配置。北半球和赤道纬度带城市不平移；南半球月份向后平移 6 个月。公式为：

```text
month_index = month - 1 + (6 if centroid_lat < -equatorial_band else 0)
local_month = month_index mod 12 + 1
local_year  = year + floor(month_index / 12)
```

因此南半球 2000 年 7 月会对齐为 2001 年 1 月；跨过 12 月时年份同步推进。
月度汇总图分别输出 Terra/Aqua × 三个纬度组共 6 张图，按配置的源年份范围
进行城市内 mean/median 汇总。年度分面图输出 Terra/Aqua × 三个纬度组 × 12
个月共 72 张图；它只使用完整的对齐年度，源范围为 2000–2020 时即 2001–2020，
避免首尾年份只有半个季节周期。每个传感器仍分别使用 `qc_pass_A_T` 和
`qc_pass_A_A`。各组和传感器共享坐标范围，分组图不会在同一图中混合。月度图
位于 `results/figures/eda/local_season/<latitude_group>/`，年度分面位于
`results/figures/eda/local_season/by_year/<latitude_group>/`。

### 5.6 第四项 EDA：年度 D/N/A/M 中位数与 bootstrap CI

单独运行：

```bash
python scripts/analysis/eda_yearly_trends.py
```

也可通过 `03_eda.py` 与其他 EDA 一并运行。每张图以年份为横轴；一张图对应
一个传感器和一个纬度组，内部按 12 个日历月分面。每个年度/月/组内，先按对应
传感器的昼夜配对 QC 筛选城市，再跨城市计算中位数 D（day）、中位数 N（night）
、A 和 M。A、M 分别直接使用月度主表中已有的 `A_T` / `A_A`、
`M_T` / `M_A`，不会从年度中位数 D/N 再次推算。Terra 与 Aqua 分别使用
`qc_pass_A_T`、`qc_pass_A_A`。

每年分别在该年的城市样本内有放回抽样，计算 D、N、A、M 中位数的百分位
bootstrap 置信区间；每次抽样以城市为单位，同时抽取该城市 D/N/A/M，保持日夜、
差值和平均强度之间的配对。阴影表示配置的置信区间，图中还标记每个月面板的
年度城市样本量范围。该区间描述当年跨城市中位数的不确定性，不是拟合时间趋势
斜率的置信区间。默认 95% CI、
1,000 次重抽样和固定随机种子，均可在 `scripts/analysis/config.json` 的 `eda`
配置中调整。

输出共 6 张图（2 个传感器 × 3 个纬度组），每张图包含 12 个月面板及四条
D/N/A/M 中位数趋势线。所有传感器和分组共用对称纵轴范围，便于比较；结果保存在
`results/figures/eda/yearly_trends/<sensor>/<latitude_group>/`。

### 5.7 第五项 EDA：城市级 (βD, βN) 年趋势斜率

单独运行：

```bash
python scripts/analysis/eda_city_slopes.py
```

也可通过 `03_eda.py` 一并运行。对每个城市、传感器和日历月份，使用该传感器
昼夜配对 QC 通过的年度记录，分别对 D（日间 SUHI）和 N（夜间 SUHI）与年份做
普通最小二乘线性拟合，计算每十年变化量 `βD` 和 `βN`。一个城市—月份至少需要
多少个有效年份由 `eda.min_city_slope_years` 配置，当前为 15；不足的城市—月份
不计算斜率，并在日志中报告。

每张图以 `βD` 为横轴、`βN` 为纵轴；按日历月份分成 12 个子图。Terra/Aqua
分别作图，且北半球、南半球、赤道纬度带仍分别输出，避免将不同纬度组混在一起；
散点改按 `cz_macro_groups` 中的大气候区着色，未分类气候用灰色表示。共 6 张图，
所有图采用一致的对称坐标范围，并显示零值参考线和 `βD = βN` 参考线。该图需要
SUHI 表中已有 `koppen_geiger_macro_group` 列；如缺失，先运行
`04_merge_city_metadata.py` 刷新元数据。输出位于
`results/figures/eda/city_slope_scatter/<sensor>/<latitude_group>/`。

此外，每次运行也会输出气候区拆分版本：按传感器 × 大气候区 × 纬度组分别作图，
共最多 36 张；每张仍包含 12 个日历月子图，气候区采用一致的单色，并保留共同的
对称坐标范围。输出位于
`results/figures/eda/city_slope_scatter/by_climate/<sensor>/<climate_group>/<latitude_group>/`。

### 5.8 第六项 EDA：Climate × local-season Month 趋势统计热图

单独运行：

```bash
python scripts/analysis/eda_climate_month_trends.py
```

也可通过 `03_eda.py` 一并运行。先按城市、传感器和月份，基于各传感器昼夜配对
QC 通过的年度观测拟合城市级 OLS 斜率；D/N 使用对应昼间、夜间 SUHI，A 使用已
存储的 `A_T` / `A_A`。每个城市—月达到 `eda.min_city_slope_years`（当前 15 个）
有效年份后才纳入汇总。南半球月份平移 6 个月，跨 12 月时对齐年加 1；北半球和
赤道纬度带保持原月。之后按 `cz_macro_groups` 中的大气候区和对齐后的 local
month 汇总，所以北/南半球城市会按月对齐后进入同一 Climate × Month 矩阵。

每个大区 × 月单元格分别计算城市级 βD、βN、βA 的中位数，并以城市为单位做
百分位 bootstrap，得到跨城市中位数的配置置信区间（默认 95%）。该 CI 描述城市
斜率中位数的不确定性，不等于单个城市回归斜率的 CI。每个传感器分别输出 βD、
βN、βA 三张热图，总计 6 张；格内标注斜率中位数、城市数，贡献城市至少 2 个且
CI 不跨 0 时附星号。并输出包含中位数、CI 上下界、样本城市数、置信水平和
bootstrap 次数的长表 CSV。结果位于
`results/figures/eda/climate_month_trend_heatmaps/`。Bootstrap 次数、置信水平和
随机种子沿用 `eda.bootstrap_replicates`、`eda.bootstrap_confidence_level`、
`eda.bootstrap_random_seed` 配置。

### 5.9 暖季—冷季 SUHI 差异及城乡分解

单独运行：

```bash
python scripts/analysis/eda_seasonal_contrasts.py
```

也可通过 `03_eda.py` 运行。北半球暖季默认为 6–8 月、冷季为 12–2 月；
南半球反过来。12 月归入下一 season year（例如 2000 年 12 月与 2001 年
1–2 月组成 season year 2001）。每条城市月度记录内部派生整数标签
`is_summer` 和 `is_winter`：对应季节为 1，肩季月份为 0。标签另存为 Parquet，
不会改写月度 master 文件。赤道带按 `eda.equatorial_latitude_band_degrees`
配置（当前 ±10°）排除在暖冷季主分析之外，标签表中仍保留且两标志均为 0。

每个城市—季节年只有当该季节三个月均通过该传感器的昼夜配对 QC 时才计算季节
均值（可由 `seasonal_contrast.minimum_valid_months_per_season` 配置，默认 3）。
同一有效月份集同时用于 urban LST、rural LST 和 SUHI，以保持分解可比。输出
`City × Year × Season` 的 Terra/Aqua D、N、A，再逐城市逐年计算
`C = warm − cold`。对至少有 `eda.min_city_slope_years` 个完整冷暖对比年份的城市，
用 OLS 估计 γ（°C/decade）。由于城乡和 SUHI 使用同一配对月份和年份，城市级
分量满足 `γ_SUHI = γ_urban − γ_rural`；这表示数值分解，不代表因果归因。

Climate 大区汇总报告 γ 中位数及跨城市 bootstrap CI，以及 γ 正、负、零的城市
比例。至少两个城市且中位数 CI 不跨 0 时，标记中位趋势为上升或下降；正负比例
描述城市斜率方向，不表示单个城市趋势显著。`All cities` 汇总包含气候未分类但
符合年份和 QC 条件的非赤道城市。结果输出至
`results/figures/eda/warm_cold_seasonal_contrasts/`，包括月度季节标签 Parquet、
城市季节值、暖冷对比、城市 γ、气候区统计 CSV，以及中位趋势 CI 和正向城市比例
图。另输出 `urban_rural_seasonal_lst_trends_<years>.png`，按 Terra/Aqua 和 D/N/A
分面，分别展示 Urban/Rural 暖季与冷季的城市斜率中位数及 bootstrap CI。暖、冷季
趋势在每个城市内使用相同的完整配对年份，因此可判断 contrast 变化与暖季升温、
冷季降温或两者共同变化的关联。该分析对应“SUHI 季节性是否系统变化”及
“其变化如何与城乡季节循环相关”。

## 6. 当前实现与数据状态

- 旧的 1,700 个优先城市分析曾生成 510,000 行月度、171,700 行季节和
  42,500 行年度记录；这是历史运行记录，不代表新全量乡村数据的最终行数。
- 新全量 rural 导出使用独立目录和数据集名，最终城市数由非空 rural halo、
  fraction 资产及城乡可匹配记录共同决定；不要直接把优先点数量当成最终面板城市数。
- 已从本地 GHSL Shapefile 生成
  [`city_metadata.parquet`](./data/processed/city_metadata.parquet)：11,422 个唯一
  `ID_UC_G0`，`centroid_lat` 范围为 -54.809232° 至 69.660579°；22 个无效几何
  在质心计算前使用 `make_valid` 修复。气候大类由 `cz_macro_groups` 配置，
  运行 `02_build_monthly_suhi.py` 时会重新生成 lookup 并合并到月度、季节和年度
  SUHI 输出。当前 11,356 个有分类的城市已分入六个大类，66 个质心仍落在无分类
  像元。
- 当前环境尚未生成年度 clean Parquet 与 SUHI master/聚合表。将匹配年份的
  全量 rural 年度 CSV 放入配置目录后，运行本地预处理即可生成。
- [`scripts/analysis/03_eda.py`](./scripts/analysis/03_eda.py) 已实现 Terra/Aqua
  昼夜 SUHI 日历月汇总、逐年分面、local-season 分纬度组图和年度 D/N/A/M 中位数
  及 bootstrap CI、城市级 `(βD, βN)` 年趋势散点，以及 Climate × local-season
  Month 的 βD/βN/βA 趋势统计热图，以及暖季—冷季 SUHI 差异和城乡分解；
  EDA 模块也可
  分别由 `eda_calendar_months.py`、`eda_yearly_panels.py` 和
  `eda_local_season.py`、`eda_yearly_trends.py`、`eda_city_slopes.py`、
  `eda_climate_month_trends.py` 和 `eda_seasonal_contrasts.py` 独立运行。
  其他系统化 EDA 和统计建模尚未实现。

## 7. 后续分析路线（待开展，不属于当前已完成处理）

建议先完成数据覆盖与质量审计，再开展解释性分析。下面是候选路线，需结合研究
问题和可用协变量进一步确定。

1. **覆盖率与质量审计**
   - 按年份、季节、传感器、昼夜统计 QC 通过率、缺失率和有效城市数。
   - 检查 Aqua 与 Terra 的可用时期差异；把数据缺测和热岛值为零区分开。
   - 检查 SUHI 分布、极端值、城乡匹配率和空间覆盖。
2. **描述性时空变化**
   - 展示月、季节、年度 SUHI 分布与城市轨迹。
   - 比较 Terra/Aqua、白天/夜间，并分别描述 `A_*` 日夜差异和 `M_*` 平均强度。
   - 输出全球地图、分区汇总和不确定性/样本量信息。
3. **长期趋势**
   - 对城市年度或季节序列估计趋势，并报告样本覆盖、置信区间和显著性。
   - 考虑城市与年份效应、序列相关及城市内相关；不要仅凭相关性作因果解释。
4. **异质性与驱动因素**
   - 评估气候带、纬度/半球、城市规模或发展阶段的差异。
   - 将人口、植被、城市扩张、地形及气候等协变量按城市或网格、年份匹配后，
     再建立分层或面板模型；这些协变量目前不在 SUHI 主表中。
5. **稳健性分析**
   - 比较 QC 阈值、fraction 阈值及 rural buffer 设置的敏感性。
   - 检查季节定义、城乡配对规则与传感器重叠时段对结论的影响。
   - 评估异常值处理、城市样本边界和完整观测期筛选方案。

目前已确定先使用固定气象季节；南北半球分别定义冬夏季节可作为后续分析的
扩展，不应与当前 `suhi_seasonal.parquet` 的定义混淆。开展正式模型前应先完成
EDA、明确主要假设、协变量来源、缺失处理方案和统计模型，并将新增步骤实现为
独立脚本或 notebook，避免直接改写原始数据或 master analysis table。
