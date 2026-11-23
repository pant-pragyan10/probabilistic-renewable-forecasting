import numpy as np
import pandas as pd

from probfc.live import parse_response, toa_radiation


def fake_json(n=26):
    t = pd.date_range("2026-11-20 23:00", periods=n, freq="h")
    return {"utc_offset_seconds": 19800, "hourly": {
        "time": [x.strftime("%Y-%m-%dT%H:%M") for x in t], "shortwave_radiation": [100.0] * n, "cloud_cover": [50] * n,
        "temperature_2m": [20.0] * n, "relative_humidity_2m": [60] * n, "wind_speed_100m": [8.0] * n, "wind_direction_100m": [270] * n}}


def test_parse_response_converts_units_and_direction():
    df = parse_response(fake_json(), 28.45, 77.58)
    assert df["cloud"].iloc[0] == 0.5 and df["dir_to"].iloc[0] == 90.0       # wind FROM west blows TOWARD east
    assert {"ssrd", "temp_c", "humidity", "speed100", "tsr"} <= set(df.columns) and len(df) == 26


def test_sun_geometry_day_and_night():
    t = pd.DatetimeIndex(["2026-06-21 06:30", "2026-06-21 18:30"])         # UTC; near noon and midnight at lon 77.5 (UTC+5.2)
    toa = toa_radiation(28.0, 77.5, t)
    assert toa[0] > 800 and toa[1] == 0
    assert toa_radiation(60.0, 0.0, pd.DatetimeIndex(["2026-12-21 12:00"]))[0] < 200    # winter noon at 60 N: sun only ~6 degrees up