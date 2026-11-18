"""Interactive demo: explore the saved forecasts. No training happens here; everything is read from results/.

Run from the project folder:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from probfc import metrics as M
from probfc import plots, whatif
from probfc.config import load_config, resolve_path
from probfc.evaluation import LABELS, MODELS, load_all

st.set_page_config(page_title="Probabilistic renewable forecasts", layout="wide")
plots.set_style()
CFG = load_config()
TAUS = np.asarray(CFG["quantiles"])
RESULTS = resolve_path(CFG, "results")
COLORS = {"quantile_regression": plots.BLUE, "lightgbm": plots.ORANGE, "neural_net": plots.AQUA, "climatology": plots.INK_2}


def available_tags():
    tags = [t for t, lab in (("val", "Validation period (used for tuning)"), ("test", "Test period (final, untouched until the end)"))
            if all((RESULTS / "predictions" / f"{tr}_{m}_{t}.parquet").exists() for tr in ("solar", "wind") for m in MODELS)]
    return tags


@st.cache_data(show_spinner=False)
def load(track, tag):
    base, Q = load_all(CFG, track, tag)
    return base, Q


def col(t):
    return int(np.argmin(np.abs(TAUS - t)))


tags = available_tags()
st.title("Probabilistic forecasting of solar and wind power")
st.caption("Instead of one number per hour, each model gives a range. Data: GEFCom2014 (3 solar zones, 10 wind zones). "
           "Power is shown as a fraction of installed capacity.")
if not tags:
    st.error("No saved predictions found. Run `python scripts/run_all.py` first (see README).")
    st.stop()

with st.sidebar:
    st.header("Choose what to look at")
    tag = st.radio("Period", tags, format_func=lambda t: {"val": "Validation", "test": "Test (final)"}[t])
    track = st.radio("Energy source", ["solar", "wind"], format_func=str.capitalize)
    try:
        base, Q = load(track, tag)
    except Exception as e:  # missing raw data for the solar sun-up mask, etc.
        st.error(f"Could not load predictions: {e}\n\nRun `python scripts/00_get_data.py` and the training scripts first.")
        st.stop()
    zone = st.selectbox("Zone", sorted(base["zone"].unique()))
    model = st.selectbox("Model", MODELS[::-1], format_func=LABELS.get)
    zb = base[base["zone"] == zone]
    lo_d, hi_d = zb["timestamp"].min().date(), zb["timestamp"].max().date()
    start = st.date_input("Start date", value=lo_d + (hi_d - lo_d) // 2, min_value=lo_d, max_value=hi_d)
    days = st.slider("Days to show", 1, 14, 5)
    show90 = st.checkbox("Show 90% range", True)
    show50 = st.checkbox("Show 50% range", True)

mask_all = base["sun_up"].to_numpy() if track == "solar" else np.ones(len(base), bool)
y_all = base["y"].to_numpy()

t_fore, t_try, t_compare, t_rel, t_why, t_value = st.tabs(
    ["Forecast", "Try your own inputs", "Compare models", "Is the range honest?", "What drives the forecast", "Why it matters"])

# ---- 1. forecast ---------------------------------------------------------------------------
with t_fore:
    t0 = pd.Timestamp(start)
    win = zb[(zb["timestamp"] >= t0) & (zb["timestamp"] < t0 + pd.Timedelta(days=days))]
    if win.empty:
        st.warning("No data in this window; pick another date.")
    else:
        q = Q[model][win.index.to_numpy()]
        fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
        x = win["timestamp"]
        if show90:
            ax.fill_between(x, q[:, col(0.05)], q[:, col(0.95)], color=plots.SEQ_BLUE[0], linewidth=0, label="90% range")
        if show50:
            ax.fill_between(x, q[:, col(0.25)], q[:, col(0.75)], color=plots.SEQ_BLUE[2], linewidth=0, label="50% range")
        ax.plot(x, q[:, col(0.5)], color=plots.SEQ_BLUE[5], lw=1.6, label="Median forecast")
        ax.plot(x, win["y"], color=plots.INK, lw=1.4, label="Actual")
        ax.set_ylabel("Power (fraction of capacity)")
        ax.set_ylim(-0.02, 1.15)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1 if days <= 7 else 2))   # one label per day
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.legend(loc="upper right", ncol=4)
        ax.set_title(f"{track.capitalize()}, zone {zone}: {LABELS[model]}")
        st.pyplot(fig)
        plt.close(fig)

        inside = np.mean((win["y"].to_numpy() >= q[:, col(0.05)]) & (win["y"].to_numpy() <= q[:, col(0.95)]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual power inside the 90% range (this window)", f"{inside:.0%}",
                  help="Over a long period this should be close to 90% for an honest model; short windows vary a lot.")
        yy = y_all[mask_all]
        qq = Q[model][mask_all]
        c2.metric("Whole period: pinball loss" + (" (daylight)" if track == "solar" else ""), f"{M.pinball(yy, qq, TAUS):.4f}",
                  help="Lower is better. Compare with the climatology reference in the next tab.")
        c3.metric("Whole period: 90% range covers", f"{M.coverage(yy, qq, TAUS, 0.05, 0.95):.0%}")
        st.caption("Reading the chart: the darker band should contain the actual line about half of the time, the lighter band about 90%. "
                   "A wide band means the model is unsure; a narrow band means it is confident.")


# ---- 1b. try your own inputs ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the training data...")
def whatif_resources(track_):
    from probfc.data import load_track
    from probfc.features import build_features, usable_rows
    from probfc.split import split_masks

    t = build_features(load_track(track_, CFG), track_, CFG)
    t = t[usable_rows(t)].reset_index(drop=True)
    train = t[split_masks(t["timestamp"], track_, CFG)["train"]]
    cols = [c for c in t.columns if c not in ("timestamp", "zone", "y", "sun_up")]
    return train, cols, whatif.load_boosters(CFG, track_), sorted(t["zone"].unique())


with t_try:
    st.subheader("Type in the weather, get a forecast range")
    st.write("This uses the trained LightGBM model. You set the main inputs; everything else is filled with the typical value for that "
             f"{track} zone, hour and time of year. The sidebar's energy source and zone apply here too.")
    try:
        train_w, cols_w, boosters_w, zones_w = whatif_resources(track)
    except Exception as e:
        boosters_w = None
        st.error(f"Could not prepare the model inputs: {e}")
    if boosters_w is None:
        st.info("Trained LightGBM models not found. Run `python scripts/04_lightgbm.py` first.")
    else:
        left, right = st.columns([1, 1.25])
        with left:
            day = st.date_input("Date (sets the time of year)", value=pd.Timestamp("2013-10-15").date(), key="wi_date")
            hour = st.slider("Hour of day", 0, 23, 12, key="wi_hour")
            doy = pd.Timestamp(day).dayofyear
            typ = whatif.typical_rows(train_w, zone, hour, doy)
            lag_default = float(np.clip(typ["y_lag24"].median(), 0, 1))
            if track == "solar":
                ssrd = st.slider("Forecast sunlight reaching the ground (W/m²)", 0, 1100, 600, key="wi_ssrd")
                cloud = st.slider("Forecast cloud cover (%)", 0, 100, 30, key="wi_cloud") / 100
                temp = st.slider("Forecast temperature (°C)", -5, 45, 22, key="wi_temp")
                hum = st.slider("Forecast humidity (%)", 0, 100, 50, key="wi_hum")
                lag = st.slider("Power at this hour yesterday (fraction of capacity)", 0.0, 1.0, round(lag_default, 2), 0.01, key=f"wi_lag_{track}_{zone}_{hour}")
                row = whatif.solar_row(typ, cols_w, CFG, hour, doy, ssrd, cloud, temp, hum, lag)
                sun_up = bool(row["tsr"] > CFG["sun_up_threshold"])
            else:
                speed = st.slider("Forecast wind speed at 100 m (m/s)", 0.0, 25.0, 7.0, 0.5, key="wi_speed")
                direction = st.slider("Wind direction (degrees)", 0, 359, 180, key="wi_dir")
                lag = st.slider("Power at this hour yesterday (fraction of capacity)", 0.0, 1.0, round(lag_default, 2), 0.01, key=f"wi_lag_{track}_{zone}_{hour}")
                row = whatif.wind_row(typ, cols_w, CFG, hour, doy, speed, direction, lag)
                sun_up = None
        q_model = whatif.predict(boosters_w, row, zone, zones_w, sun_up=sun_up)
        ref_y = typ["y"].to_numpy()
        q_clim = np.quantile(ref_y, TAUS) if (sun_up is not False and len(ref_y)) else np.zeros(len(TAUS))
        with right:
            if sun_up is False:
                st.warning("At this hour and time of year the sun is down at this site, so the forecast is zero.")
            fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
            for y_pos, (label, q) in enumerate((("With your inputs\n(LightGBM)", q_model), ("No weather info\n(climatology)", q_clim))):
                ax.barh(y_pos, q[col(0.95)] - q[col(0.05)], left=q[col(0.05)], height=0.5, color=plots.SEQ_BLUE[0])
                ax.barh(y_pos, q[col(0.75)] - q[col(0.25)], left=q[col(0.25)], height=0.5, color=plots.SEQ_BLUE[2])
                ax.plot([q[col(0.5)]] * 2, [y_pos - 0.25, y_pos + 0.25], color=plots.SEQ_BLUE[6], lw=2.5)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["With your inputs\n(LightGBM)", "No weather info\n(climatology)"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Power (fraction of installed capacity)")
            ax.grid(axis="y", visible=False)
            ax.invert_yaxis()
            ax.set_title("Forecast range: light = 90% likely, dark = 50% likely, line = median", fontsize=9.5)
            st.pyplot(fig)
            plt.close(fig)
            lo, med, hi = q_model[col(0.05)], q_model[col(0.5)], q_model[col(0.95)]
            st.markdown(f"**Model says:** most likely about **{med:.0%}** of capacity, and a 90% chance it ends up between **{lo:.0%}** and **{hi:.0%}**.")
            st.caption("The lower bar (climatology) shows the spread you would have without any weather forecast, using the same zone, hour and season. "
                       "Compare the two bars to see how much the weather inputs narrow the range.")
        with st.expander("All 19 quantiles"):
            st.dataframe(pd.DataFrame({"Quantile": [f"{t:.0%}" for t in TAUS], "Forecast (fraction of capacity)": q_model.round(3),
                                       "Climatology": q_clim.round(3)}), hide_index=True)
        st.caption("Assumptions: the weather in the hours before and after is taken to be the same as the chosen hour. The model was trained on 2012-2014 "
                   "data from these anonymous sites, so unusual combinations of inputs are outside what it has seen, and this is a demonstration of "
                   "how the model responds, not a forecast for a real date.")

# ---- 2. compare models -----------------------------------------------------------------------
with t_compare:
    rows = []
    ref = M.pinball(y_all[mask_all], Q["climatology"][mask_all], TAUS)
    for m in MODELS:
        s = M.summarize(y_all[mask_all], Q[m][mask_all], TAUS)
        rows.append({"Model": LABELS[m], "Pinball loss": s["pinball"], "Approx. CRPS": s["crps_approx"],
                     "90% range covers": s["coverage_90"], "50% range covers": s["coverage_50"],
                     "Avg 90% width": s["width_90"], "Improvement vs climatology": 1 - s["pinball"] / ref})
    tbl = pd.DataFrame(rows).set_index("Model")
    st.subheader(f"{track.capitalize()}, {'daylight hours' if track == 'solar' else 'all hours'} ({'validation' if tag == 'val' else 'test'} period)")
    st.dataframe(tbl.style.format({"Pinball loss": "{:.4f}", "Approx. CRPS": "{:.4f}", "90% range covers": "{:.1%}",
                                   "50% range covers": "{:.1%}", "Avg 90% width": "{:.3f}", "Improvement vs climatology": "{:.1%}"})
                 .highlight_min(subset=["Pinball loss"], color="#cde2fb"), use_container_width=True)
    st.caption("Pinball loss and CRPS: lower is better. Coverage: should be close to 90% and 50%. Width: narrower is better *only if* coverage is honest. "
               "Climatology = past power at that hour, no weather information (the reference to beat).")
    fig, ax = plt.subplots(figsize=(7, 3.2), layout="constrained")
    ax.barh([LABELS[m] for m in MODELS[::-1]], [tbl.loc[LABELS[m], "Pinball loss"] for m in MODELS[::-1]],
            color=[COLORS[m] for m in MODELS[::-1]], height=0.6)
    ax.set_xlabel("Pinball loss (lower is better)")
    ax.grid(axis="y", visible=False)
    st.pyplot(fig)
    plt.close(fig)

# ---- 3. reliability ----------------------------------------------------------------------------
with t_rel:
    st.write("If a model says “90% of the time the power will be below this line”, it should be right about 90% of the time. "
             "Points on the diagonal mean the ranges can be trusted; below it means overconfident on the upper side.")
    fig, ax = plt.subplots(figsize=(5.2, 5), layout="constrained")
    ax.plot([0, 1], [0, 1], color=plots.GRID, lw=2.4, zorder=1)
    for m in MODELS:
        ax.plot(TAUS, M.calibration(y_all[mask_all], Q[m][mask_all], TAUS), color=COLORS[m], marker="o", ms=3.5, label=LABELS[m],
                **({"ls": "--"} if m == "climatology" else {}))
    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Share of outcomes below the forecast quantile")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

# ---- 4. what drives it ----------------------------------------------------------------------------
with t_why:
    shap_csv = RESULTS / f"shap_importance_{tag}.csv"
    img = RESULTS / "figures" / f"shap_importance_{tag}.png"
    st.write("SHAP shows how much each input moves the LightGBM forecast on average. "
             "Solar is driven by forecast sunlight and time of day; wind by forecast wind speed.")
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.info("Run `python scripts/07_shap.py` to create this chart.")
    if shap_csv.exists():
        sh = pd.read_csv(shap_csv)
        sub = sh[(sh["track"] == track) & (np.isclose(sh["quantile"], 0.5))].sort_values("mean_abs_shap", ascending=False).head(8)
        st.dataframe(sub[["feature", "mean_abs_shap"]].rename(columns={"feature": "Input", "mean_abs_shap": "Average impact (median model)"}),
                     hide_index=True)

# ---- 5. why it matters -------------------------------------------------------------------------------
with t_value:
    st.subheader("Why a range is more useful than one number")
    st.write("A grid operator has to keep spare capacity ready in case wind or solar output comes in lower than forecast. "
             "A single-number forecast gives no hint how much to hold, so operators use a fixed safety margin. "
             "A range tells them how much margin is needed *this hour*: less on predictable hours, more on uncertain ones.")
    f = RESULTS / f"usefulness_{tag}.csv"
    if f.exists():
        u = pd.read_csv(f)
        u = u[u["track"] == track]
        show = u[["model", "own_range_coverage", "own_range_width", "fixed_margin_width_equal_coverage", "width_change_at_equal_coverage_pct"]].copy()
        show.columns = ["Model", "Model's own range covers", "Model's range width", "Fixed margin needed for the same coverage",
                        "Width change (negative = narrower)"]
        st.dataframe(show.style.format({"Model's own range covers": "{:.1%}", "Model's range width": "{:.3f}",
                                        "Fixed margin needed for the same coverage": "{:.3f}", "Width change (negative = narrower)": "{:+.0f}%"}),
                     hide_index=True, use_container_width=True)
        st.caption("Second half of the period. The fixed margin is median ± a constant, set *after the fact* so it covers as often as the model's range "
                   "(this favours the baseline). Widths are fractions of installed capacity. Climatology row is a sanity check, not a recommendation.")
    else:
        st.info("Run `python scripts/08_usefulness.py` to create this table.")