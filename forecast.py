"""
forecast.py
===========
Predictive layer — forecasts the elderly-population share of every
neighbourhood 5 and 10 years ahead.

The age CSV gives the elderly (>=65) percentage per neighbourhood for
every year 1997-2025.  With ~28 yearly points per neighbourhood a simple
linear-trend model is the honest choice — deep learning would over-fit.

We fit an ordinary-least-squares line to the last 15 years of each
neighbourhood's elderly-% series and project it to 2030 and 2035, with a
plausible range from the regression residuals.

Output (cached): processed/forecast.parquet
    barri_name, elderly_2025, elderly_2030, elderly_2035,
    trend_per_year, series_json
"""

import os, json
import numpy as np
import pandas as pd

BASE     = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE, "processed")


def fit_forecast(verbose=True):
    demo = pd.read_parquet(os.path.join(PROC_DIR, "demographics.parquet"))

    recs = []
    for _, row in demo.iterrows():
        series = json.loads(row["pct_series"])
        years  = np.array(sorted(int(y) for y in series))
        vals   = np.array([series[str(y)] for y in years], dtype=float)

        ok = ~np.isnan(vals)
        years, vals = years[ok], vals[ok]
        if len(years) < 5:
            continue

        # use the last 15 observations for a locally-relevant trend
        if len(years) > 15:
            years, vals = years[-15:], vals[-15:]

        # OLS linear fit
        A = np.vstack([years, np.ones_like(years)]).T
        slope, intercept = np.linalg.lstsq(A, vals, rcond=None)[0]
        resid = vals - (slope * years + intercept)
        sigma = resid.std()

        def proj(y):
            return slope * y + intercept

        e2025 = float(series.get("2025", proj(2025)))
        e2030 = float(proj(2030))
        e2035 = float(proj(2035))

        recs.append(dict(
            barri_name     = row["barri_name"],
            elderly_2025   = round(e2025, 2),
            elderly_2030   = round(e2030, 2),
            elderly_2035   = round(e2035, 2),
            trend_per_year = round(float(slope), 4),
            band           = round(float(sigma), 2),
            series_json    = row["pct_series"],
        ))

    fc = pd.DataFrame(recs)
    fc["delta_10yr"] = (fc["elderly_2035"] - fc["elderly_2025"]).round(2)
    fc = fc.sort_values("delta_10yr", ascending=False).reset_index(drop=True)
    fc.to_parquet(os.path.join(PROC_DIR, "forecast.parquet"))

    if verbose:
        print(f"✓ forecast for {len(fc)} neighbourhoods")
        print("\nFastest-ageing neighbourhoods (10-yr elderly-% rise):")
        print(fc[["barri_name", "elderly_2025", "elderly_2035",
                  "delta_10yr"]].head(8).to_string(index=False))
    return fc


def load_forecast():
    p = os.path.join(PROC_DIR, "forecast.parquet")
    if not os.path.exists(p):
        return fit_forecast(verbose=False)
    return pd.read_parquet(p)


if __name__ == "__main__":
    fit_forecast()
