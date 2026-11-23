"""Live weather forecast for the app's 'Fetch' button, from Open-Meteo (free, no account or API key needed).

It returns the hourly inputs our models use: sunlight at the ground, cloud cover, temperature, humidity and 100 m wind. Open-Meteo
forecasts are not the ECMWF archive the models were trained on, and the location is not one of the (anonymous) GEFCom sites,
so the output is a demonstration of the pipeline on live inputs, not a calibrated forecast for that place."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

URL = os.environ.get("PROBFC_WEATHER_URL", "https://api.open-meteo.com/v1/forecast")   # env var only used by the tests
HOURLY = "shortwave_radiation,cloud_cover,temperature_2m,relative_humidity_2m,wind_speed_100m,wind_direction_100m"


def toa_radiation(lat: float, lon: float, utc_times: pd.DatetimeIndex) -> np.ndarray:
    """Sunlight arriving at the top of the atmosphere (W/m2) from solar geometry; hour-mid-point approximation. Zero at night."""
    t = utc_times - pd.Timedelta(minutes=30)
    doy = t.dayofyear.to_numpy()
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    hang = np.deg2rad(15.0 * (hour + lon / 15.0 - 12.0))
    la = np.deg2rad(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(hang)
    return 1361.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)) * np.clip(cosz, 0.0, None)


def parse_response(js: dict, lat: float, lon: float) -> pd.DataFrame:
    """Open-Meteo JSON -> hourly frame indexed by LOCAL time, with model-ready columns."""
    h = js["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"]), "ssrd": h["shortwave_radiation"], "cloud": np.asarray(h["cloud_cover"], float) / 100.0,
                       "temp_c": h["temperature_2m"], "humidity": h["relative_humidity_2m"],
                       "speed100": h["wind_speed_100m"], "dir_from": h["wind_direction_100m"]})
    df = df.set_index("time")
    if df.isna().any().any():
        df = df.interpolate(limit_direction="both")
    off = pd.Timedelta(seconds=int(js.get("utc_offset_seconds", 0)))
    df["tsr"] = 0.7 * toa_radiation(lat, lon, df.index - off)     # net top-of-atmosphere radiation ~ 70% of incoming (rough)
    df["dir_to"] = (df["dir_from"] + 180.0) % 360.0                # model's u/v convention is the direction the wind blows TO
    return df


def fetch_forecast(lat: float, lon: float, day, timeout: float = 15.0) -> pd.DataFrame:
    """Hourly forecast for `day` (local date) plus the hour before and after it. Raises RuntimeError with a plain message on failure."""
    import requests

    day = pd.Timestamp(day).normalize()
    params = {"latitude": lat, "longitude": lon, "hourly": HOURLY, "wind_speed_unit": "ms", "timezone": "auto",
              "start_date": (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "end_date": (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")}
    try:
        r = requests.get(URL, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach the weather service (is the internet on?): {e}") from e
    if r.status_code != 200:
        raise RuntimeError(f"Weather service said: {r.text[:200]}")
    try:
        df = parse_response(r.json(), lat, lon)
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Unexpected reply from the weather service: {e}") from e
    need = pd.date_range(day - pd.Timedelta(hours=1), day + pd.Timedelta(hours=24), freq="h")
    if not all(t in df.index for t in need):
        raise RuntimeError("The weather service has no forecast for that date (it covers roughly the past 3 months and the next 16 days).")
    return df.loc[need]