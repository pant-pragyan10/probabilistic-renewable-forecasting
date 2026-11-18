# Probabilistic forecasting of solar and wind generation (GEFCom2014)

Instead of one number per hour, every model outputs 19 quantiles (5% ... 95%), i.e. a range with calibrated
probabilities. Three models are compared against a climatology reference:

| Model | File | Notes |
|---|---|---|
| Climatology | `probfc/models/climatology.py` | quantiles of past power by zone and hour; no weather |
| Quantile Regression | `probfc/models/linear_qr.py` | statsmodels, one linear model per zone and quantile |
| LightGBM | `probfc/models/lgbm_qr.py` | one boosted model per quantile, pooled over zones |
| Neural network | `probfc/models/nn_qr.py` | PyTorch MLP, pinball loss, 19 ordered outputs, 3-seed average |

## Quick start
```
pip install -r requirements.txt          # Mac: also `brew install libomp` for LightGBM
python scripts/run_all.py --tests        # everything, ~15 min on a laptop CPU
python scripts/run_all.py --final        # LAST: refit on train+val and score the test period once
```
Or step by step (each script needs the ones before it):

| Script | What it does |
|---|---|
| `00_get_data.py` | downloads the GEFCom2014 solar and wind CSVs into `data/raw` |
| `01_validate_data.py` | integrity checks, writes `results/data_report.md` and clean parquet files |
| `02_eda.py` | exploratory figures |
| `03_baselines.py` | climatology + Quantile Regression |
| `04_lightgbm.py` | LightGBM, tuned on validation |
| `05_nn.py` | neural network, tuned on validation (CPU by default; `--device auto` on Colab/Kaggle) |
| `06_evaluate.py` | results table, reliability diagrams, fan charts, slices by hour/month/zone |
| `07_shap.py` | TreeSHAP importance for the 10/50/90% LightGBM models |
| `08_usefulness.py` | compares the models' ranges with a fixed safety margin at equal coverage |
| `run_all.py` | runs the above in order |

Settings (paths, split dates, quantile levels, column names) are in `configs/default.yaml`.

## Method in brief
- **Task:** day-ahead hourly forecast. Zones: 3 (solar), 10 (wind).
- **Split:** chronological, no shuffling. Solar 15 / 6 / 6 months (train / validation / test), wind 12 / 6 / 6.
  Validation is for tuning. The test period is only used by `--final`, after refitting on train+validation.
- **No leakage:** measured power is used only if at least 24 h old; weather forecasts for the target hour and its
  neighbours are allowed (`tests/test_features.py` proves it by corrupting recent power).
- **Post-processing:** quantiles sorted (so they never cross), clipped to [0, 1], zero at night for solar.
- **Metrics:** pinball loss (main), approximate CRPS (2 x mean pinball over the grid), 90%/50% interval coverage,
  interval width, reliability diagrams, skill vs. climatology. Solar is reported on all hours and on daylight hours only;
  daylight is the honest figure because night is trivially zero.

## Outputs
`results/results_table_<val|test>.csv`, `results/figures/*.png`, `results/slice_*.csv`, `results/shap_importance_*.csv`,
saved predictions in `results/predictions/`, tuned settings in `results/lgbm_best_*.json` and `results/nn_best_*.json`.

## Notes and limitations
- Data are the GEFCom2014 files from the repository accompanying Dumas et al. (2022); the competition is described in
  Hong et al. (2016). The official release could not be reached from the build environment, so the copy used here is the
  one in that repository.
- Training periods are short (12-15 months), so seasonal effects are only partly covered.
- LightGBM's ranges are slightly too narrow (90% intervals cover about 85-86%); no recalibration step is implemented.
- Neural networks are not bit-for-bit reproducible across machines; expect differences in the third decimal.
- The CRPS here is an approximation from 19 quantiles, used to compare models, not an exact CRPS.

## Demo app and report
```
pip install streamlit
streamlit run app/streamlit_app.py          # explore forecasts by source, zone, model and date (reads saved results; no training)
python report/build_report.py               # rebuilds report/Milestone2_Results_and_Value_Report.docx from results/
```