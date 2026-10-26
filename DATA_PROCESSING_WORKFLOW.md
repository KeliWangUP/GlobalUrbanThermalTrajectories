# DNA-SUHI：Day–Night Asymmetry of Surface Urban Heat Islands

**DNA-SUHI** 是本项目的研究名称：DNA 代表 **D**ay–**N**ight
**A**symmetry（日夜不对称），SUHI 则指地表城市热岛。名称强调项目同时研究
城市热岛的白天与夜间特征，以及两者之间的不对称差异；它也对应主分析表中的
`A_T`、`A_A` 日夜差异指标和 `M_T`、`M_A` 日夜平均指标。

本文记录 DNA-SUHI 从源数据、处理步骤到后续分析的主线。

> 本文是项目端到端数据流程的主记录。已经实现或运行过的步骤与建议中的后续分析分开标记；配置或处理逻辑变化时，应同步更新本文。  
> 最近核对：2026-10-26

## 1. 研究数据与核心对象

研究期由 GEE 配置控制，当前为 2000–2024 年。城市边界使用
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

`run_export.py` 的目标由 `export.targets` 控制。当前配置是 `["rural"]`；
需要重新导出城市温度时将目标改为 `["city"]`。全量乡村年度导出也可直接运行：

```bash
python scripts/gee_modul/export_rural_all_cities_annual.py
```

导出任务提交成功不等于任务完成。等待 Earth Engine 任务成功后，再从 Drive
下载年度 CSV。全量乡村文件放入 `data/raw/rural_all_cities`；不要与旧优先
城市文件 `data/raw/rural` 混放，否则会造成同一年有多个输入文件。

## 4. 本地数据层级

```text
data/
├── raw/                          # 下载的原始 CSV，保留不改写
│   ├── urban/
│   ├── rural/                    # 旧优先城市数据，非当前全量输入
│   └── rural_all_cities/         # 当前配置的乡村全量输入目录
├── parquet/                      # 按年清洗后的 Parquet
│   ├── urban/
│   └── rural/
└── processed/
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
参考线。若某城市在该月份至少有一个通过 QC 的年份，即纳入该城市点；
每点可用年份数也保存在聚合结果中。

运行时会逐传感器、逐月打印合格城市—年—月份记录数、唯一合格城市数、
通过率和作图城市数。图片保存到
`results/figures/eda/terra_day_vs_night_suhi_median_2000-2020.png` 和
`results/figures/eda/aqua_day_vs_night_suhi_median_2000-2020.png`。年份、
跨年聚合统计量、输出目录和 DPI 由
[`scripts/analysis/config.json`](./scripts/analysis/config.json) 中的 `eda`
和 `paths.eda_figure_dir` 设置。

## 6. 当前实现与数据状态

- 旧的 1,700 个优先城市分析曾生成 510,000 行月度、171,700 行季节和
  42,500 行年度记录；这是历史运行记录，不代表新全量乡村数据的最终行数。
- 新全量 rural 导出使用独立目录和数据集名，最终城市数由非空 rural halo、
  fraction 资产及城乡可匹配记录共同决定；不要直接把优先点数量当成最终面板城市数。
- 当前工作区可见 `data/raw/urban` 和旧 `data/raw/rural`；全量输入目录及
  `data/parquet`、`data/processed` 目前尚未创建。把新年度 CSV 下载到配置目录后，
  才能运行本地两步分析脚本生成这些输出。
- [`scripts/analysis/03_eda.py`](./scripts/analysis/03_eda.py) 已实现首个 Terra/Aqua
  昼夜 SUHI 月份面板图；其他系统化 EDA 和统计建模尚未实现。

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
