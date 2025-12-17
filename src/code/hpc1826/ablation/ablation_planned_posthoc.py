# ablation_planned_posthoc.py
# Run in the folder containing: significance_input_rmse.csv
#   python ablation_planned_posthoc.py

import math
import numpy as np
import pandas as pd
from scipy import stats

FULL_NAME = "【full】"  # keep consistent with your column name

def holm(pvals):
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        factor = m - i
        p_adj = min(1.0, p * factor)
        running = max(running, p_adj)
        adj[k] = running
    return adj

def cohens_dz(x, y):
    d = x - y
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    return d.mean() / sd

# load blocks x variants
mat = pd.read_csv("../../../data/results/ablation/hpc1826/significance_input_rmse.csv")
# first column might be 'block'
if mat.columns[0].lower() in ["block", "seed", "run_id"]:
    mat = mat.set_index(mat.columns[0])

cols = list(mat.columns)
assert FULL_NAME in cols, f"Cannot find FULL_NAME column: {FULL_NAME}"

# summary
summary = pd.DataFrame({"mean": mat.mean(), "std": mat.std(ddof=1)})
summary["mean±std"] = summary["mean"].map(lambda v: f"{v:.6g}") + " ± " + summary["std"].map(lambda v: f"{v:.6g}")

full_mean = summary.loc[FULL_NAME, "mean"]
summary["delta_vs_full"] = summary["mean"] - full_mean
summary["pct_vs_full(%)"] = summary["delta_vs_full"] / full_mean * 100.0

print("\n================ Mean ± STD (RMSE) ================")
print(summary.sort_values("mean")[["mean±std", "delta_vs_full", "pct_vs_full(%)"]])

# Friedman across all variants
arrays = [mat[c].values for c in cols]
chi2, p = stats.friedmanchisquare(*arrays)
print("\n==================== Friedman =====================")
print(f"chi2 = {chi2:.6g}, p = {p:.6g}")

# Planned post-hoc: full vs each ablation
raw_p = {}
eff = []
for c in cols:
    if c == FULL_NAME:
        continue
    w_stat, p_raw = stats.wilcoxon(mat[FULL_NAME].values, mat[c].values, zero_method="wilcox")
    raw_p[c] = float(p_raw)

    dz = cohens_dz(mat[c].values, mat[FULL_NAME].values)  # >0 means worse than full (higher RMSE)
    eff.append((c, p_raw, dz))

p_holm = holm(raw_p)

out = []
for c, p_raw, dz in eff:
    out.append({
        "variant": c,
        "p_raw(full vs variant)": p_raw,
        "p_holm(m=5)": p_holm[c],
        "cohens_dz(worse>0)": dz
    })

out_df = pd.DataFrame(out).sort_values("p_holm(m=5)")
out_df.to_csv("ablation_posthoc_full_vs_each_holm.csv", index=False)

print("\n====== Planned post-hoc (full vs each, Holm m=5) ======")
print(out_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print("\nSaved: ablation_posthoc_full_vs_each_holm.csv")
