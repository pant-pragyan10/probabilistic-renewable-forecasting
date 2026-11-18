# Data validation report

## solar — PASS

| check | result | detail |
|---|---|---|
| required columns present | pass | 15 columns |
| zone ids are 1..n_zones | pass | found [1, 2, 3] |
| rows per zone as expected | pass | expected 19680, found {1: 19680, 2: 19680, 3: 19680} |
| no missing values | pass | 0 NaN |
| all numeric values finite | pass |  |
| timestamps unique within each zone | pass |  |
| timestamps strictly hourly (no gaps) in each zone | pass |  |
| all zones cover the same time range | pass |  |
| date range as expected | pass | found 2012-04-02 01:00:00 -> 2014-07-01 00:00:00 |
| target within [0, 1.01] | pass | min=0.0000, max=1.00355 |

```
{
  "rows": 59040,
  "zones": 3,
  "period": [
    "2012-04-02 01:00:00",
    "2014-07-01 00:00:00"
  ],
  "target_mean": 0.1838,
  "share_target_exactly_zero": 0.4454,
  "split_rows": {
    "train": 32757,
    "val": 13248,
    "test": 13035
  },
  "share_sun_down": 0.4797
}
```

## wind — PASS

| check | result | detail |
|---|---|---|
| required columns present | pass | 7 columns |
| zone ids are 1..n_zones | pass | found [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] |
| rows per zone as expected | pass | expected 17520, found {1: 17520, 2: 17520, 3: 17520, 4: 17520, 5: 17520, 6: 17520, 7: 17520, 8: 17520, 9: 17520, 10: 17520} |
| no missing values | pass | 0 NaN |
| all numeric values finite | pass |  |
| timestamps unique within each zone | pass |  |
| timestamps strictly hourly (no gaps) in each zone | pass |  |
| all zones cover the same time range | pass |  |
| date range as expected | pass | found 2012-01-02 00:00:00 -> 2013-12-31 23:00:00 |
| target within [0, 1.01] | pass | min=0.0000, max=1.00000 |

```
{
  "rows": 175200,
  "zones": 10,
  "period": [
    "2012-01-02 00:00:00",
    "2013-12-31 23:00:00"
  ],
  "target_mean": 0.3586,
  "share_target_exactly_zero": 0.0805,
  "split_rows": {
    "train": 87600,
    "val": 43440,
    "test": 44160
  }
}
```
