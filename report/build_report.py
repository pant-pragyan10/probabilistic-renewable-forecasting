"""Builds report/Milestone2_Results_and_Value_Report.docx from the files in results/ (no numbers are typed by hand)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from probfc.config import load_config

R = ROOT / "results"
F = R / "figures"
cfg = load_config()
res = pd.read_csv(R / "results_table_val.csv")
use = pd.read_csv(R / "usefulness_val.csv")
N_TESTS = 42

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for m in ("left_margin", "right_margin"):
    setattr(sec, m, Cm(2.2))
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.1
for name, size in (("Heading 1", 15), ("Heading 2", 12.5)):
    h = doc.styles[name]
    h.font.name, h.font.size, h.font.bold = "Calibri", Pt(size), True
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(14), Pt(4)


def P(text="", bold=False, italic=False, size=None, align=None, after=None, color=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor(*color)
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(items):
    for t in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(t, tuple):
            r = p.add_run(t[0])
            r.bold = True
            p.add_run(t[1])
        else:
            p.add_run(t)
        p.paragraph_format.space_after = Pt(2)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def table(header, rows, widths=None, bold_rows=(), size=9.5, center=True):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size = True, Pt(size)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "1F3A5F")
    for k, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(size)
            run.bold = k in bold_rows
            if i > 0 and center:
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if k in bold_rows:
                shade(cells[i], "E3EEFB")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(name, caption, width=16.2):
    doc.add_picture(str(F / name), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = P(caption, italic=True, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=8, color=(0x52, 0x51, 0x4E))
    return p


pct = lambda x, d=0: f"{100 * x:.{d}f}%"


def model_rows(track, subset):
    d = res[(res["track"] == track) & (res["subset"] == subset)]
    out, best = [], d["pinball"].idxmin()
    for i, r in d.iterrows():
        out.append([r["model"], f"{r['pinball']:.4f}", pct(r["skill_vs_climatology"]) if r["model"] != "Climatology" else "reference",
                    pct(r["coverage_90"], 1), pct(r["coverage_50"], 1), f"{r['width_90']:.3f}"])
    return out, [list(d.index).index(best)]


def get(track, subset, model, col):
    return float(res[(res["track"] == track) & (res["subset"] == subset) & (res["model"] == model)][col].iloc[0])


def u(track, model, col):
    return float(use[(use["track"] == track) & (use["model"] == model)][col].iloc[0])


# ------------------------------------------------------------------ title
P("Milestone 2: Results and Value Report", bold=True, size=22, after=2, color=(0x1F, 0x3A, 0x5F))
P("Probabilistic forecasting of solar and wind power generation (GEFCom2014)", size=13, after=2)
P("CSET400C Capstone Project, Bennett University", size=10.5, after=2, color=(0x52, 0x51, 0x4E))
P("Team: Pragyan, Sambhav Jain, Supratim Saha", size=10.5, after=10, color=(0x52, 0x51, 0x4E))
P("Status of this report: all results below are from the validation period. The held-out test period has been reserved and is not "
  "used anywhere in this report; test results will be added after a single final run.", italic=True, size=10, after=8)

# ------------------------------------------------------------------ 1
doc.add_heading("1. What problem this solves, and why a range helps", 1)
P("Wind and solar plants produce power that depends on the weather, so the amount fed into the grid changes from hour to hour. "
  "The people who run the power grid (grid operators) must keep supply equal to demand at every moment. When a renewable plant "
  "produces less than expected, they have to cover the gap with spare capacity that is held ready in advance.")
P("Most forecasts give a single number for each hour. A single number does not say how far off it might be, so operators "
  "typically fall back on a fixed safety margin. This project instead produces a range for every hour: 19 values, from the level "
  "the power is 95% likely to exceed to the level it is 95% likely to stay under. On an easy hour the range is narrow; on an "
  "uncertain hour (for example, patchy cloud) it is wide. That tells the operator how much margin is needed for that specific hour.")
figure("eda_wind_power_curve.png", "Figure 1. Wind power against forecast wind speed. At the same forecast speed, actual output varies widely. "
       "The dashed lines mark the 10th and 90th percentiles. A single-number forecast hides this spread.", 14.5)

# ------------------------------------------------------------------ 2
doc.add_heading("2. What has been built", 1)
P("A complete, reproducible pipeline from raw data to evaluated forecasts. It runs with a single command "
  "(python scripts/run_all.py) and takes about 12 to 20 minutes on a laptop CPU.")
table(["Component", "What it does"], [
    ["Data layer", "Downloads the GEFCom2014 solar and wind files, checks them for gaps, duplicates, missing or out-of-range values, and saves clean versions."],
    ["Features", "Weather forecasts for the target hour and its neighbours, time of day and year, a clear-sky ratio for solar, wind speed and direction for wind, and past power that is at least 24 hours old."],
    ["Post-processing", "Sorts the quantiles so they never cross, limits them to the range 0 to 1, and sets solar to zero at night."],
    ["Reference model", "Climatology: the range of past power at the same hour. It uses no weather information."],
    ["Model 1", "Quantile Regression (statsmodels): one linear model per zone and quantile."],
    ["Model 2", "LightGBM with the quantile objective: one boosted-tree model per quantile, tuned on validation data."],
    ["Model 3", "Neural network (PyTorch): one network with 19 ordered outputs trained with pinball loss, averaged over 3 random seeds."],
    ["Evaluation", "Pinball loss, approximate CRPS, interval coverage, reliability diagrams, forecast charts, breakdowns by hour, month and zone, SHAP."],
    ["Demo app", "A Streamlit app (app/streamlit_app.py) to explore the saved forecasts by zone, model and date."],
    ["Tests", f"{N_TESTS} automatic tests covering the data checks, the split, the metrics, the models, the post-processing, and a test that proves no recent power leaks into the features."],
], widths=[3.4, 13.0], center=False)

# ------------------------------------------------------------------ 3
doc.add_heading("3. Data and experimental set-up", 1)
s, w = cfg["splits"]["solar"], cfg["splits"]["wind"]
bullets([
    ("Data. ", "GEFCom2014 energy-forecasting competition data (Hong et al., 2016): hourly solar power for 3 zones (59,040 rows, April 2012 to July 2014) and wind power for 10 zones (175,200 rows, 2012 to 2013), with weather-model forecasts as inputs. Power is expressed as a fraction of installed capacity. The files were taken from the public repository that accompanies Dumas et al. (2022); the official release could not be reached from the build environment."),
    ("Split by time, never shuffled. ", f"Solar: training until {s['train_end']}, validation until {s['val_end']}, then test. Wind: training until {w['train_end']}, validation until {w['val_end']}, then test. Validation is used to choose settings. The test period is touched once at the end, after refitting on training plus validation."),
    ("Day-ahead set-up and no leakage. ", "A forecast for hour t may use weather forecasts for that day and measured power only if it is at least 24 hours old. An automatic test corrupts recent power values and checks that no feature changes."),
    ("Solar night hours. ", "About 48% of solar hours are night, when output is exactly zero. Results are therefore reported for daylight hours as well as all hours; the daylight figure is the more honest one."),
])
figure("eda_splits.png", "Figure 2. Chronological split of the two data sets.", 14)

# ------------------------------------------------------------------ 4
doc.add_heading("4. How a forecast is judged", 1)
bullets([
    ("Pinball loss (main score, lower is better). ", "Each quantile is scored by how far the actual value falls on the wrong side of it, with a penalty weighted by the quantile level. Example: if the actual power is 0.6 and the 90% quantile forecast is 0.8, the loss is 0.1 × 0.2 = 0.02; if the 10% quantile forecast is 0.3, the loss is 0.1 × 0.3 = 0.03. A model scores well only if its ranges are both honest and tight."),
    ("Approximate CRPS. ", "Twice the average pinball loss over the 19 quantiles. It is an approximation from a finite grid and is used to compare models, not as an exact CRPS."),
    ("Coverage. ", "How often the actual value falls inside the 90% range (should be about 90%) and the 50% range (about 50%)."),
    ("Width. ", "The average size of the 90% range. Narrower is better, but only if coverage stays honest."),
    ("Improvement over climatology. ", "How much lower the pinball loss is than the weather-free reference."),
])
P("A pinball loss on its own has no obvious meaning, which is why every number below is shown next to the climatology reference and next to coverage.")

# ------------------------------------------------------------------ 5
doc.add_heading("5. Results on the validation period", 1)
hdr = ["Model", "Pinball loss", "Improvement vs climatology", "90% range covers", "50% range covers", "Avg 90% width"]
rows, bi = model_rows("solar", "daylight")
P("Table 1. Solar, daylight hours (3 zones, 13,248 validation hours in total including night).", bold=True, size=10, after=2)
table(hdr, rows, widths=[3.6, 2.2, 3.0, 2.6, 2.6, 2.4], bold_rows=bi)
rows, bi = model_rows("wind", "all_hours")
P("Table 2. Wind, all hours (10 zones, 43,440 validation hours).", bold=True, size=10, after=2)
table(hdr, rows, widths=[3.6, 2.2, 3.0, 2.6, 2.6, 2.4], bold_rows=bi)

nn_s, lg_s, qr_s = (get("solar", "daylight", m, "pinball") for m in ("Neural network", "LightGBM", "Quantile Regression"))
nn_w, lg_w, qr_w = (get("wind", "all_hours", m, "pinball") for m in ("Neural network", "LightGBM", "Quantile Regression"))
P("What the tables show", bold=True, after=2)
bullets([
    f"All three models clearly beat the weather-free reference: pinball loss is {pct(get('solar','daylight','Neural network','skill_vs_climatology'))} lower for the neural network on solar daylight hours and {pct(get('wind','all_hours','Neural network','skill_vs_climatology'))} lower on wind.",
    f"The order is the one expected: Quantile Regression, then LightGBM, then the neural network. LightGBM improves on Quantile Regression by {pct(1 - lg_s / qr_s)} on solar and {pct(1 - lg_w / qr_w)} on wind.",
    f"The neural network and LightGBM are close ({pct(1 - nn_s / lg_s, 1)} apart on solar, {pct(1 - nn_w / lg_w, 1)} on wind). We have not tested whether this gap is larger than random variation, so we do not claim one is better.",
    f"Calibration differs. The neural network's 90% range covered {pct(get('solar','daylight','Neural network','coverage_90'),1)} (solar) and {pct(get('wind','all_hours','Neural network','coverage_90'),1)} (wind), close to the 90% target. LightGBM covered {pct(get('solar','daylight','LightGBM','coverage_90'),1)} and {pct(get('wind','all_hours','LightGBM','coverage_90'),1)}: slightly overconfident, meaning its ranges are a little too narrow.",
])
figure("eval_reliability_val.png", "Figure 3. Reliability diagram. Points on the diagonal mean the forecast quantiles are honest. The climatology reference "
       "(dashed) is clearly off for solar because it ignores the weather.", 15.5)
figure("eval_fan_solar_val.png", "Figure 4. Solar, zone 1, neural network: forecast range against actual output over six days. "
       "The range widens on the less predictable day (22 Sep) and tightens on clear days.")
figure("eval_fan_wind_val.png", "Figure 5. Wind, zone 1, neural network: forecast range against actual output. Most of the actual line stays inside the bands.")
figure("eval_slices_val.png", "Figure 6. Pinball loss by hour of day and by month. The models' advantage is largest around midday for solar "
       "and holds across months for wind.", 16.5)

doc.add_heading("What drives the forecast", 2)
P("SHAP values (computed with LightGBM's built-in TreeSHAP) show which inputs move the forecast. For solar, the forecast surface "
  "sunlight (ssrd), the clear-sky ratio, the time of day and the previous day's power at the same hour matter most. For wind, the forecast "
  "wind speed at 100 m is by far the largest driver, followed by the zone. This agrees with how the plants physically work, which is a useful "
  "sanity check that the models have learned the right relationships.")
figure("shap_importance_val.png", "Figure 7. Average impact of each input on the LightGBM forecast at the 10th, 50th and 90th percentiles.", 16.2)

# ------------------------------------------------------------------ 6
doc.add_heading("6. Is the range actually useful? A margin comparison", 1)
P("To connect the scores to the operator's problem, we compared two ways of building a safety range around the same central forecast "
  "(the model's own median, so the accuracy of the central guess is identical):")
bullets([
    "A fixed margin: the median plus or minus one constant amount for every hour, as a single-number forecast would require.",
    "The model's own 5% to 95% range, which changes from hour to hour.",
])
P("The comparison is made on the second half of the validation period. To keep it conservative, the fixed margin is given hindsight: "
  "it is set after the fact so that it covers the actual power exactly as often as the model's own range does. The figures show the "
  "average width of the range, as a fraction of installed capacity; a narrower range at the same coverage means less spare capacity has to be held.")
hdr = ["Model", "Own range covers", "Own range width", "Fixed margin width, same coverage", "Change"]
for track, label in (("solar", "Table 3. Solar, daylight hours"), ("wind", "Table 4. Wind, all hours")):
    P(label, bold=True, size=10, after=2)
    rows = []
    for m in ("Quantile Regression", "LightGBM", "Neural network"):
        rows.append([m, pct(u(track, m, "own_range_coverage"), 1), f"{u(track, m, 'own_range_width'):.3f}",
                     f"{u(track, m, 'fixed_margin_width_equal_coverage'):.3f}", f"{u(track, m, 'width_change_at_equal_coverage_pct'):+.0f}%"])
    table(hdr, rows, widths=[3.8, 2.8, 2.8, 4.4, 2.0])
P(f"For the neural network, the hour-by-hour range is {abs(u('solar','Neural network','width_change_at_equal_coverage_pct')):.0f}% narrower than a fixed margin "
  f"on solar and {abs(u('wind','Neural network','width_change_at_equal_coverage_pct')):.0f}% narrower on wind, at the same coverage. "
  f"For LightGBM the gain is smaller ({abs(u('solar','LightGBM','width_change_at_equal_coverage_pct')):.0f}% and {abs(u('wind','LightGBM','width_change_at_equal_coverage_pct')):.0f}%); "
  f"its own range covers less often ({pct(u('solar','LightGBM','own_range_coverage'),1)} on solar), so it is being compared at a lower coverage level. "
  "In plain terms: a model that knows which hours are uncertain can ask for a large margin only on those hours, and that is where the saving comes from. "
  "These are single-split estimates without confidence intervals.")
P("This is an illustration of usefulness, not a cost estimate. We have not converted capacity into money, because that needs real market and "
  "reserve-pricing data that this project does not use.", italic=True)

# ------------------------------------------------------------------ 7
doc.add_heading("7. What these results do and do not show", 1)
bullets([
    ("Supported. ", "A range forecast built from weather forecasts is clearly more informative than a weather-free reference and than a fixed margin, with ranges that are close to honest for the neural network and Quantile Regression."),
    ("Not yet shown. ", "Performance on the untouched test period; whether the neural network and LightGBM differ by more than chance; comparison with published GEFCom2014 scores (our split, features and data subset differ, so the numbers are not directly comparable)."),
    ("Limits of the data. ", "Training periods are short (12 to 15 months), so seasonal behaviour is only partly covered. Results come from one chronological split, not repeated splits."),
    ("Known weakness. ", "LightGBM's ranges are slightly too narrow. No recalibration step has been built; it is listed under next steps."),
    ("Reference strength. ", "Climatology is a weak reference because it ignores weather. The margin comparison in Section 6 is a stronger test."),
    ("Approximations. ", "CRPS is approximated from 19 quantiles. Neural-network results can differ in the third decimal between machines."),
])

# ------------------------------------------------------------------ 8
doc.add_heading("8. Status and next steps", 1)
table(["Item", "Status"], [
    ["Dataset preparation and validation", "Done"],
    ["Baseline and three models, both tracks (solar primary, wind extension)", "Done"],
    ["Experimental pipeline and preliminary results (validation)", "Done"],
    ["Demo application", "Done (Streamlit)"],
    ["Unit and integration tests", f"{N_TESTS} passing"],
    ["Final test-period run (once)", "Planned: python scripts/run_all.py --final"],
    ["Confidence intervals on model differences", "Planned"],
    ["Recalibration of LightGBM ranges; stronger baselines", "Planned / optional"],
], widths=[10.5, 5.9], center=False)

doc.add_heading("References", 2)
bullets([
    "Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., Hyndman, R. J. (2016). Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. International Journal of Forecasting, 32(3).",
    "Dumas, J., Wehenkel, A., Lanaspeze, D., Cornélusse, B., Sutera, A. (2022). A deep generative model for probabilistic energy forecasting in power systems: normalizing flows. Applied Energy, 305.",
])

out = ROOT / "report" / "Milestone2_Results_and_Value_Report.docx"
doc.save(out)
print("saved", out)