English | [简体中文](README.zh-CN.md)

# Southwest China Vortex Activity: Interdecadal Changes and Warming Responses

Joint analyses of Southwest China vortex (SWCV) frequency, intensity, and tracks, focusing on diurnal changes in winter basin-local genesis. CMIP6/DAMIP experiments provide complementary analyses of regional background circulation responses to warming.

## Core scripts

| Script | Analysis | Main outputs |
| --- | --- | --- |
| `analyze_tracks.py` | Track features, K-means clustering, annual frequency, and temperature associations | Event tables, cluster summaries, and interdecadal plots |
| `analyze_joint_response.py` | Joint frequency–intensity–track statistics | Annual joint metrics and strong/eastward-moving event statistics |
| `analyze_path_intensity_fixed_windows.py` | Associations between early-life intensity and eastward reach | Standardized regression coefficients and confidence intervals |
| `analyze_winter_genesis_temporal_structure.py` | Day/night trends and temporal clustering of winter local genesis | Day/night trends, trend contrasts, and hourly counts |
| `analyze_winter_night_transition.py` | Nighttime genesis associations with precursor convergence and vorticity stretching changes | Poisson regression coefficients and adjusted year coefficients |
| `analyze_cmip6_multimember_responses.py` | Height responses under greenhouse-gas-only and historical all-forcing experiments | Multimodel comparisons of raw, zonal-mean, and eddy height |

`audit_panel_alignment.py` checks subplot rectangles and labels for the CMIP6 figure.

## Environment

Use Python 3.13. Run from the repository root:

```sh
python -m venv .venv
# Activate the virtual environment using your shell's activation command.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Data preparation

Tracks come from the CNN/ERA5 SWCV catalogue published by He & Cao: [Zenodo dataset](https://doi.org/10.5281/zenodo.15073815). Place annual `YYYY.txt` files in `data/raw/swcv_zenodo_15073815/`. Each row contains the track step, latitude, longitude, and UTC timestamp (`YYYYMMDDHH`); step 0 marks the start of a new track.

Global mean temperature comes from [NASA GISTEMP](https://data.giss.nasa.gov/gistemp/). Save the annual table as `data/raw/gistemp_v4_global.csv`, retaining its initial explanatory row and the `Year` and `J-D` columns.

Intensity, winter-environment, and CMIP6 analyses read preprocessed tables from `data/processed/`. Required filenames and fields are listed below. Results are written to `data/processed/`, and figures to `figures/`.

## Usage

### 1. Tracks and annual statistics

```sh
python scripts/analyze_tracks.py
```

Reads tracks and GISTEMP, computes genesis/lysis positions, duration, and net displacement, then standardizes track features for K-means clustering. Outputs include `events.csv`, `track_points.csv.gz`, `annual_metrics.csv`, `track_cluster_summary.csv`, and temperature-association tables.

Seasonal counts in the annual statistics are grouped by calendar year. The winter-specific analyses use cross-year winters, assigning December to the following season year. Eastward escape is defined by a maximum track longitude of at least 110°E; track clusters are based on multiple trajectory features.

### 2. Joint frequency–intensity–track analysis

Prepare `era5_event_intensity.csv`, with one row per event matched to the track table by `event_id`, including intensity fields such as `max_height_deficit_gpm` and `max_vorticity_s_1`. These metrics come from ERA5 fields collocated along the tracks. Use the current three-cluster classification: 0, southern-local; 1, basin-local; 2, long-lived eastward-moving.

```sh
python scripts/analyze_joint_response.py
```

Outputs `annual_joint_fip_metrics.csv`, `joint_warming_associations_hac.csv`, and `events_with_era5_intensity.csv.gz`. Separately standardized height deficit and vorticity form a joint intensity index; strong events use its upper-decile threshold.

### 3. Tracks and early-life intensity

Prepare three tables:

- `events_season_year_terrain_qc.csv.gz`: terrain-screened event records containing `event_id`, `genesis_time`, `lysis_time`, `season_year`, `season`, `track_cluster`, `genesis_lat`, `genesis_lon`, `lysis_lat`, `lysis_lon`, and `max_lon`.
- `terrain_qc_intensity_points.csv.gz`: track-point records with `event_id`, `time`, `height_deficit_gpm_700_c25`, and `max_vorticity_s_1_700_c25`.
- `jja_event_moisture_lag_event_level.csv.gz`: summer event records with `event_id`, `lag_hours`, and `area_convergence_anom_kg_m2_s`.

```sh
python scripts/analyze_path_intensity_fixed_windows.py
```

Compares intensity within 0–6, 0–12, and 0–24 hours after genesis, fits continuous eastward-reach associations and track-group contrasts, and includes covariates such as valid sampling counts. Outputs `path_intensity_fixed_window_models.csv` and a coefficient figure. Prepare the full set of fields used in the script's event filters and regression formulas.

### 4. Day/night changes in winter local genesis

Prepare `events_terrain_qc_intensity.csv.gz`, including `genesis_time`, `genesis_lon`, `track_cluster`, `observed_hours`, and `duration_hours`. The script selects local events in track cluster 1 and complete winters during 1980–2023. Nighttime is defined as 20:00–08:00 Beijing time.

```sh
python scripts/analyze_winter_genesis_temporal_structure.py
```

Outputs `winter_genesis_temporal_structure_annual.csv`, `winter_genesis_temporal_structure_statistics.csv`, `winter_genesis_beijing_hour_counts.csv`, and a summary JSON, estimating nighttime and daytime trends and their contrast.

### 5. Dynamical precursors of nighttime genesis

Prepare `winter_local_convergence_transition_precursors.csv.gz`, with one row per risk interval. Include `utc_hour`, `season_year`, `regime`, `month`, `decade`, `events`, and `convergence_700`, `convergence_850`, `stretching_700`, `stretching_850`. The dynamical fields represent changes from −12 to −6 hours relative to genesis; `decade` is `(season_year - 2001.5) / 10`.

```sh
python scripts/analyze_winter_night_transition.py
```

Fits Poisson models controlling for circulation regime, month, and time of day, with precursors standardized using the early-period standard deviation. Outputs `winter_night_transition_conditional_models.csv`, including log-rate coefficients per standard deviation, confidence intervals, and year coefficients adjusted for precursor conditions.

### 6. CMIP6 multimember height responses

Prepare `cmip6_multimember_z850_responses.csv`, containing `model`, `experiment`, `member`, `raw_z850_m_change`, `zonal_z850_m_change`, and `eddy_z850_m_change`: eight models, two experiments (`hist-GHG` and `historical`), and three members per model per experiment. The regional index is summer 850 hPa height over 15–25°N, 115–150°E; changes are the 1985–2014 mean minus the 1951–1980 mean, in metres.

Models: BCC-CSM2-MR, CNRM-CM6-1, CanESM5, GISS-E2-1-G, HadGEM3-GC31-LL, IPSL-CM6A-LR, MIROC6, and MRI-ESM2-0.

Also prepare `cmip6_eddy_z850_forced_responses.csv` for single-member comparisons, including `source`, `experiment`, `member`, and the three change fields above.

```sh
python scripts/analyze_cmip6_multimember_responses.py
```

First averages three members within each model, then summarizes responses across models. Outputs `cmip6_multimember_z850_model_summary.csv`, `cmip6_multimember_z850_multimodel_summary.csv`, experiment-contrast tables, and height-response figures. Eddy height is regional height minus the zonal-mean height over the same latitude band, describing regional relative-height changes.

## Methods and references

Bibliographic records are in `refs/`. ERA5 data are provided by the Copernicus Climate Data Store, and CMIP6/DAMIP data by ESGF; data use follows the respective providers' terms. Contact the repository maintainer regarding reuse permissions for project code.
