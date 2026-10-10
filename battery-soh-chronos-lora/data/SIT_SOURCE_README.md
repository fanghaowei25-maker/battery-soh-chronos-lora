# SIT Battery Degradation Dataset

**Version:** 1.0  
**Published:** Singapore Institute of Technology Institutional Research Repository (SIT IRR)  
**Dataset DOI:** https://doi.org/10.25447/sit.32101523  
**Landing page:** https://irr.singaporetech.edu.sg/articles/dataset/SIT_Battery_Data/32101523  
**Companion code (Python loader + notebooks):** https://github.com/karthicgrepo/sit-battery-dataset-code  
**Code DOI (all versions):** https://doi.org/10.5281/zenodo.19769514  
**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Overview

This dataset contains battery degradation data collected at the **Singapore Institute of Technology (SIT)** as part of an 18-month battery ageing and cycle-life research campaign. Twenty-two large-format prismatic LiFePO₄ cells (50 Ah, 3.2 V nominal, Narada FE50B) were cycled under four experimental conditions spanning two discharge rates (1C and 2C), two thermal environments (ambient laboratory and 40 °C), and both full depth-of-discharge and partial micro-cycling protocols.

The data captures voltage, current, capacity, energy and surface temperature at 1-second resolution across multiple testers, channel configurations and cycling protocols. All measurements are provided in simplified, analysis-ready formats alongside cycle summary tables and standalone continuous temperature files.

The companion Python loader library (`sit_utils`) and four Jupyter notebooks are available separately at the code repository listed above.

---

## Test Setup Summary

| Tester | Brand | Cells | Discharge Rate | Environment | Cycling Mode |
|--------|-------|:-----:|:--------------:|:-----------:|:------------:|
| **Repower #001** | Repower HRCDS-5V60A | 8 (001-1 to 001-8) | 1C (50 A) | Ambient (bag) | Deep cycling |
| **Repower #002** | Repower HRCDS-5V60A | 6 (002-1 to 002-7) | 1C (50 A) Ch1–4 / 2C (100 A) Ch5–8 | 40 °C (chamber) | Deep cycling |
| **Repower #003** | Repower HRCDS-5V60A | 4 (003-1, 003-3, 003-5, 003-7) | 2C (100 A) | 40 °C (chamber) | Deep cycling |
| **Chroma #101** | Chroma 17010 | 4 (101-1, 101-3, Ch5+6, Ch7+8) | 1C (50 A) deep / micro-cycle | Ambient (bag) | Deep + Micro cycling |

**Key:**
- **1C** = 50 A (full-capacity rate for 50 Ah cells)
- **2C** = 100 A (two Repower channels connected in parallel)
- **Deep cycling** = Full charge/discharge across the full voltage window (2.50 to 3.65 V)
- **Micro cycling** = Shallow partial charge/discharge cycles centred at 50% SoC (Chroma #101, Ch5+6 and Ch7+8 only)
- **Ambient** = Air-conditioned SIT laboratory (~25 ± 2 °C); cells stored in Li-ion safety bags
- **Chamber** = Programmable environmental chamber (Sanwood SM-408-CD-FB) at 40 °C setpoint

---

## Folder Structure

### What is on the IRR landing page

The dataset's IRR landing page (https://irr.singaporetech.edu.sg/articles/dataset/SIT_Battery_Data/32101523) hosts four files at the same level:

```
SIT IRR landing page
├── README.md                     ← This file (orientation, no download required)
├── README.txt                    ← Plain-text mirror of README.md
├── EXCEL_File_GUIDE.pdf          ← Description of the per-cycle xlsx file structure
└── Data.zip                      ← The dataset (~4 GB compressed, 10.1 GB uncompressed, ~14,800 files)
```

The three orientation files (`README.md`, `README.txt`, `EXCEL_File_GUIDE.pdf`) can be inspected without downloading the archive. The full data are inside `Data.zip`.

### What is inside Data.zip

Extracting `Data.zip` produces a single top-level folder named `Data/` that contains **six** sub-folders. There is no README inside the archive itself — the README on the landing page covers it.

```
Data/
│
├── Repower_001/                             ← Tester #001: 8 cells, 1C, ambient
│   ├── Cycle_Data/
│   │   ├── 001-1/                               xlsx files (one per charge/discharge cycle)
│   │   ├── 001-2/
│   │   ├── 001-3/
│   │   ├── 001-4/
│   │   ├── 001-5/
│   │   ├── 001-6/
│   │   ├── 001-7/
│   │   └── 001-8/
│   └── Temperature/
│       ├── 001-1_temperature.csv             Continuous surface temperature, cell 001-1
│       ├── 001-2_temperature.csv
│       ├── 001-3_temperature.csv
│       ├── 001-4_temperature.csv
│       ├── 001-5_temperature.csv
│       ├── 001-6_temperature.csv
│       ├── 001-7_temperature.csv
│       ├── 001-8_temperature.csv
│       └── README.md                         Column descriptions for temperature files
│
├── Repower_002/                             ← Tester #002: 6 cells, 1C/2C, 40 °C
│   ├── Cycle_Data/
│   │   ├── 002-1/                               1C cells (Ch1 to Ch4 single channel, 50 A)
│   │   ├── 002-2/
│   │   ├── 002-3/
│   │   ├── 002-4/
│   │   ├── 002-5/                               2C cells (Ch5+6 parallel, 100 A)
│   │   └── 002-7/                               2C cells (Ch7+8 parallel, 100 A)
│   └── Temperature/
│       ├── 002-1_temperature.csv
│       ├── 002-2_temperature.csv
│       ├── 002-3_temperature.csv
│       ├── 002-4_temperature.csv
│       ├── 002-5_temperature.csv
│       ├── 002-7_temperature.csv
│       └── README.md
│
├── Repower_003/                             ← Tester #003: 4 cells, 2C, 40 °C
│   ├── Cycle_Data/
│   │   ├── 003-1/                               All cells paired (Ch1+2, Ch3+4, Ch5+6, Ch7+8)
│   │   ├── 003-3/
│   │   ├── 003-5/
│   │   └── 003-7/
│   └── Temperature/
│       ├── 003-1_temperature.csv
│       ├── 003-3_temperature.csv
│       ├── 003-5_temperature.csv
│       ├── 003-7_temperature.csv
│       └── README.md
│
├── Chroma_101/                              ← Tester #101: 4 cells, 1C deep + micro, ambient
│   ├── Cycle_Data/
│   │   ├── 101-1/                               Deep cycling (Ch1+2 paired, 50 A)
│   │   └── 101-3/                               Deep cycling (Ch3+4 paired, 50 A)
│   ├── Temperature/
│   │   ├── 101-1_temperature.csv
│   │   ├── 101-3_temperature.csv
│   │   └── README.md
│   └── Micro_Cycling/
│       ├── Channel_5/                           Daily detail CSV files (Ch5+6 paired)
│       └── Channel_7/                           Daily detail CSV files (Ch7+8 paired)
│
├── Cycle_Summary/                           ← Per-cell capacity fade tables (20 CSV files)
│   ├── 001-1.csv  001-2.csv  001-3.csv  001-4.csv
│   ├── 001-5.csv  001-6.csv  001-7.csv  001-8.csv
│   ├── 002-1.csv  002-2.csv  002-3.csv  002-4.csv
│   ├── 002-5.csv  002-7.csv
│   ├── 003-1.csv  003-3.csv  003-5.csv  003-7.csv
│   └── 101-1.csv  101-3.csv
│
└── Documentation/                           ← Reference documents
    ├── Column_Descriptions.md                   Signal names and units for all file types
    ├── Thermocouple_Map.md                      Channel-to-cell thermocouple assignment
    ├── Chroma_Tester_17010-EN.pdf               Chroma 17010 tester datasheet
    ├── Sanwood-Catalog.pdf                      Sanwood environmental chamber catalogue
    └── 04 Battery Explosion Proof Chamber SM-408-CC-FB Catalogue.pdf
```

**Note on tester renaming:** In May 2023, the SIT laboratory physically relabelled the Repower units. Repower #001 became #005, #002 became #003, and #003 became #004. All data files and folder names in this dataset use the **original** numbering (001, 002, 003) throughout for consistency. Some filenames within `Repower_001/Cycle_Data/` may carry a `005_` prefix; these belong to tester #001.

---

## Cycling Groups

| Group | Tester | Cells | Discharge Rate | Environment | Protocol |
|-------|--------|-------|:--------------:|:-----------:|----------|
| G1 | Repower #001 + Chroma #101 (Ch1–4) | 001-1 to 001-8, 101-1, 101-3 | 1C (50 A) | Ambient | Deep cycling |
| G2 | Repower #002 (Ch1–4) | 002-1 to 002-4 | 1C (50 A) | 40 °C | Deep cycling |
| G3 | Repower #002 (Ch5–8) + Repower #003 | 002-5, 002-7, 003-1, 003-3, 003-5, 003-7 | 2C (100 A) | 40 °C | Deep cycling |
| G4 | Chroma #101 (Ch5–8) | Channel_5, Channel_7 | Micro-cycle | Ambient | Micro cycling |

---

## Cycling Protocols

### Deep Cycling (Groups G1, G2, G3)
Full charge/discharge cycles across the manufacturer-specified voltage window.

1. **CC charge:** 50 A constant current to 3.65 V
2. **CV hold:** 3.65 V constant voltage until taper current reaches 2.5 A
3. **Rest:** 15 min
4. **CC discharge:** 50 A (1C) or 100 A (2C) constant current to 2.50 V
5. **Rest:** 15 min

### Micro Cycling (Group G4)
Shallow partial charge/discharge cycles centred at 50% SoC with 15-minute rest periods between each step. Approximately 10 Ah charge throughput per micro-cycle. Periodic full charge/discharge reference performance tests (RPTs) were also conducted; see the associated paper for RPT capacity values.

---

## Data Formats and File Naming

| Folder | Format | Extension | Description |
|--------|--------|-----------|-------------|
| Cycle Data | Excel workbook | `.xlsx` | One file per charge/discharge cycle; two sheets: *Charge* and *Discharge* |
| `Temperature/` | CSV | `.csv` | Continuous surface temperature with absolute ISO 8601 timestamps |
| `Cycle_Summary/` | CSV | `.csv` | One file per cell; two rows per cycle (charge + discharge) |
| `Micro_Cycling/` | CSV | `.csv` | Daily detail files from the Chroma tester (22 columns) |

### Cycle Data Columns (xlsx files)

| Column | Repower name | Chroma name | Units |
|--------|-------------|-------------|-------|
| Time | `Relative Time(Sec)` | `Relative Time(Sec)` | s (resets per file) |
| Voltage | `Voltage(V)` | `Voltage(V)` | V |
| Current | `Current(A)` | `Current(A)` | A (positive = charge, negative = discharge) |
| Capacity | `Capacity(Ah)` | `Capacity(Ah)` | Ah (cumulative within half-cycle) |
| Energy | `Energy(Wh)` | `Wh` | Wh (cumulative within half-cycle) |
| Temperature | `MTV Celsius TN` | `MTV Celsius TN1`, `MTV Celsius TN2` | °C |

### Cycle Data File Naming

Files follow the pattern `YYYY_MM_DD_HH_MM_SS__TTT_C_N_S.xlsx`, where:
- `YYYY_MM_DD_HH_MM_SS` is the session start date and time
- `TTT` is the tester identifier (001, 002, 003, or 101)
- `C` is the channel number
- `N` is the step type index
- `S` is the segment number

Sorting filenames alphabetically gives chronological order.

### Cycle Summary Columns

| Column | Description |
|--------|-------------|
| `Cycle` | Cycle number (integer, starting from 1) |
| `Type` | `charge` or `discharge` |
| `Capacity_Ah` | Accumulated capacity for the half-cycle (Ah) |
| `Energy_Wh` | Accumulated energy for the half-cycle (Wh) |
| `Duration_s` | Half-cycle duration (s) |
| `Start_Timestamp` | Absolute start time (parsed from filename) |
| `Filename` | Source xlsx filename in `Cycle_Data/` |

---

## File Counts

| Folder | Cycle Data | Temperature | Other | Total |
|--------|:----------:|:-----------:|:-----:|------:|
| Repower_001 | 6,506 | 9 (8 CSV + README) | n/a | 6,515 |
| Repower_002 | 3,641 | 7 (6 CSV + README) | n/a | 3,648 |
| Repower_003 | 2,557 | 5 (4 CSV + README) | n/a | 2,562 |
| Chroma_101 | 1,561 | 3 (2 CSV + README) | 504 (micro) | 2,068 |
| Cycle_Summary | n/a | n/a | 20 CSV | 20 |
| Documentation | n/a | n/a | 5 files | 5 |
| **Total** | | | | **14,818** |

---

## Quick Start for Researchers

1. **Capacity fade analysis (fastest start)** → Load `Cycle_Summary/<cell_id>.csv`, filter `Type == discharge`, plot `Capacity_Ah` vs `Cycle`.
2. **Time series data** → Open `.xlsx` files inside `<tester>/Cycle_Data/<cell_id>/`. Each file = one cycle; two sheets (Charge, Discharge).
3. **Temperature analysis** → Use `<tester>/Temperature/<cell_id>_temperature.csv` for continuous records. See `Documentation/Thermocouple_Map.md` for channel-to-cell mapping.
4. **Micro-cycling data** → Use `Chroma_101/Micro_Cycling/Channel_5/` or `Channel_7/` (daily detail CSV files, 22 columns).
5. **Column descriptions** → See `Documentation/Column_Descriptions.md`.
6. **Python loader + notebooks** → Available separately at https://doi.org/10.5281/zenodo.19769514 (concept DOI; always resolves to the latest version; MIT Licence). The companion code provides a `SITDataset` loader class and four Jupyter notebooks covering quickstart, capacity fade analysis, feature extraction and RUL prediction.

### Recommended Preprocessing

- **Initial capacity baseline:** Use the maximum discharge capacity across the first five cycles as the initial capacity reference to account for partial first cycles.
- **Outlier cells:** Cells 001-3, 001-8 and 101-3 exhibit anomalously steep capacity fade far below their group peers. Flag and handle explicitly when training prognostic models.
- **Rolling smoothing:** A 5-cycle rolling median filter effectively removes tester-induced oscillations from cycle-by-cycle capacity traces.
- **Tester relabelling:** Late-campaign files in `Repower_001/` may carry a `005_` prefix; map these to tester #001.
- **Time gaps:** Overnight, weekend and holiday gaps are inherent to the weekday laboratory schedule and appear as date discontinuities in filenames and temperature timestamps.

---

## Key Research Comparisons Enabled

| Comparison | Groups / Folders | Research Question |
|------------|-----------------|-------------------|
| Temperature effect | G1 (ambient) vs G2 (40 °C), same 1C rate | Does elevated temperature accelerate capacity fade? |
| Combined C-rate + thermal stress | G2 (1C, 40 °C) vs G3 (2C, 40 °C) | Effect of higher discharge current at elevated temperature |
| Cell-to-cell variability | Within G1 (10 cells, identical conditions) | Manufacturing variability and its effect on lifetime prediction |
| Cycling pattern | G1 to G3 (deep) vs G4 (micro) | Effect of depth of discharge on ageing pathway |
| Cross-tester consistency | Repower #001 (G1) vs Chroma #101 (G1) | Measurement reproducibility across tester platforms |

---

## Citation

If you use this dataset, please cite both the dataset and the associated Data Descriptor paper:

**Dataset:**
> Ponnambalam, K. et al. SIT Battery Data (SIT Large-Format LiFePO₄ Battery Degradation Dataset, 50 Ah Prismatic Cells). SIT Institutional Research Repository (2026). https://doi.org/10.25447/sit.32101523

**Data Descriptor paper:**
> Ponnambalam, K., Balakrishnan, S., Soh, C. B., Sharma, A. & Lee, S. S. Multi-condition ageing dataset for large-format prismatic LiFePO₄ batteries with per-cell thermal monitoring. *Scientific Data* (2026). DOI available upon journal acceptance.

**Companion code:**
> Ponnambalam, K. et al. SIT Battery Degradation Dataset: Companion Code. Zenodo. https://doi.org/10.5281/zenodo.19769514 (2026).
> *(Concept DOI; always resolves to the latest version. To cite the specific snapshot v1.0.0, use https://doi.org/10.5281/zenodo.19769515.)*

---

## Contact

For questions about this dataset, please contact the authors as listed in the associated Data Descriptor paper.

Singapore Institute of Technology / Newcastle University in Singapore

---

*Last updated: 29 April 2026*
