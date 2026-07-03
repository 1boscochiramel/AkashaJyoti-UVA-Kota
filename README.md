# Akāsha Jyoti — Balloon-Borne Global UV-A Attenuation over the Kota Region (5 October 2025)

[![DOI](https://zenodo.org/badge/1287986362.svg)](https://doi.org/10.5281/zenodo.21155989)

Paper, analysis code, and data for the high-altitude balloon (HAB) flight launched from
**Sultanpur, Kota district, Rajasthan, India** (25.298°N, 76.192°E) on **5 October 2025**,
reaching ≈17.6 km pressure altitude.

**Author:** Bosco Chiramel (Independent Researcher, ORCID
[0009-0001-8456-5302](https://orcid.org/0009-0001-8456-5302))

## Headline result

The effective vertical optical depth of downwelling **global (hemispherical) UV-A**
(Apogee SU-202, 305–390 nm) was

> τ_eff = 0.233 ± 0.023

versus a direct-beam extinction expectation of τ_ext = 1.74 ± 0.21 (Rayleigh at 360 nm +
MAIAC AOD Ångström-extrapolated to 360 nm), giving

> η = τ_eff / τ_ext = 0.134 ± 0.021

— a quantified measure of how strongly multiple scattering replenishes hemispheric UV-A
under a moderately turbid, non-absorbing atmosphere.

## Repository layout

```
paper/    LaTeX source, final SSRN PDF, and all figures (PDF + PNG)
code/     Analysis scripts (Python: pandas, numpy, matplotlib)
data/     Flight data and ancillary public-data extracts
data/raw/ Raw payload datalog as recorded in flight
```

## Data dictionary

| File | Contents |
|---|---|
| `data/datalog_cleaned_with_altitude.csv` | Cleaned flight record (~7 s cadence): GPS time string, lat/lon, SU-202 output (V), T (°C, payload-internal), RH (%), pressure (hPa), pressure altitude (m) |
| `data/raw/datalog.csv` | Raw logger output incl. Geiger CPM, O3 channel, thermal-array frames |
| `data/Satellite_data_extracted.csv` | Sentinel-5P TROPOMI columns (SO2, O3, NO2, CO, UV aerosol index) along the flight track (Google Earth Engine extraction) |
| `data/Satellite_data_extracted2.csv` | MAIAC MCD19A2 AOD at 470/550 nm (×1000) + UVAI along the track |
| `data/AOD_Stat.csv` | MAIAC AOD statistics in 10/25/50/100 km buffers around the launch site |
| `data/era5.xlsx` | ERA5 extraction (boundary-layer height, cloud base, t2m, 100-m winds, etc.) |
| `data/Nayapura.xlsx`, `data/Shrinath_Puram.xlsx` | CPCB CAAQMS 4-hourly surface air quality, Kota city stations, Oct–Nov 2025 |
| `data/results.json` | All headline numbers produced by `code/paper_analysis.py` |

**Not included (size):** onboard video (≈100 GB) and the 1-kHz camera IMU `.gcsv` logs
(≈540 MB). The IMU-derived tilt/rotation statistics used in the paper are reproduced by
`code/imu_swing.py` + `code/imu_ascent_stats.py`; contact the author for the raw files.

## Reproducing the paper

```bash
pip install pandas numpy matplotlib openpyxl
python code/paper_analysis.py   # regenerates all figures + results.json
pdflatex paper/AkashaJyoti_UVA_Kota.tex  # run twice for cross-references
```

Note: `paper_analysis.py` reads the data files from a flat directory; adjust the `BASE`
path at the top of the script to point at `data/`.

## Instrument notes (documented failures)

- Geiger–Müller channel valid only below ≈4.7 km (HV corona discharge at low pressure).
- O3 channel non-responsive throughout.
- GPS time strings froze at altitude; timeline reconstructed from the logger RTC anchored
  to the first valid GPS fix and corroborated by the camera-IMU event chain.
- Post-touchdown radiometer data excluded (payload no longer sky-facing).

## Public data credits

NASA (MAIAC MCD19A2; POWER), ESA/Copernicus (Sentinel-5P TROPOMI; ERA5 via CDS), and the
Central Pollution Control Board of India (CAAQMS).

## License

- Paper text, figures, and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Code (`code/`): MIT

## Citation

> Chiramel, B. (2026). *Balloon-Borne Profiling of Global UV-A Attenuation over the Kota
> Region, Rajasthan, India: Effective Optical Depth, Satellite Aerosol Context, and
> Instrument Performance from a Low-Cost High-Altitude Flight.* Preprint.
