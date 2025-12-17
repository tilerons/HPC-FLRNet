# significance_test_single_dataset.py
#   python significance_test_single_dataset.py --metric rmse
#   python significance_test_single_dataset.py --metric r2 --higher_is_better
#

import argparse
import glob
import json
import math
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def parse_model_name(filename: str) -> str:
    """
    Extract model name from pattern: *_<MODEL>_all_metrics.json
    Example: 2025_12_09_11_58_HPCNet_all_metrics.json -> HPCNet
    """
    base = os.path.basename(filename)
    m = re.search(r"_([^_]+)_all_metrics\.json$", base)
    if not m:
        raise ValueError(f"Cannot parse model name from filename: {base}")
    return m.group(1)


def load_metrics_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Invalid JSON format (expect non-empty list): {path}")

    df = pd.DataFrame(data)
    # Create run index 0..N-1 (paired blocks for Friedman/Wilcoxon)
    df["run_id"] = np.arange(len(df), dtype=int)

    # seed is optional now (kept for traceability)
    if "seed" not in df.columns:
        df["seed"] = np.nan
    return df


def holm_correction(pvals: dict) -> dict:
    """
    Holm-Bonferroni correction for multiple comparisons.
    pvals: {(a,b): p}
    returns: {(a,b): p_adj}
    """
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    adj = {}
    running_max = 0.0
    for i, ((a, b), p) in enumerate(items):
        factor = m - i
        p_adj = min(1.0, p * factor)
        running_max = max(running_max, p_adj)
        adj[(a, b)] = running_max
    return adj


def nemenyi_posthoc(avg_ranks: pd.Series, n_blocks: int, alpha: float = 0.05):
    """
    Nemenyi post-hoc test based on average ranks.
    Returns:
      - p-value matrix (DataFrame) if SciPy studentized_range is available
      - CD (critical difference) for alpha
    """
    k = len(avg_ranks)
    se = math.sqrt(k * (k + 1) / (6.0 * n_blocks))

    studentized_range = getattr(stats, "studentized_range", None)
    if studentized_range is None:
        return None, None

    q_alpha = studentized_range.isf(alpha, k, np.inf)
    cd = q_alpha * se

    models = list(avg_ranks.index)
    pmat = pd.DataFrame(np.ones((k, k)), index=models, columns=models, dtype=float)
    for a, b in combinations(models, 2):
        q = abs(avg_ranks[a] - avg_ranks[b]) / se
        p = float(studentized_range.sf(q, k, np.inf))
        pmat.loc[a, b] = p
        pmat.loc[b, a] = p

    return pmat, cd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", type=str, default="rmse",
                    help="Metric field in JSON (e.g., rmse, mae, mape, r2, r). Default: rmse")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level. Default: 0.05")
    ap.add_argument("--higher_is_better", action="store_true",
                    help="If set, larger metric is better (e.g., r2, r). Otherwise smaller is better (e.g., rmse).")
    ap.add_argument("--pattern", type=str, default="*_all_metrics.json",
                    help="Glob pattern for metric json files. Default: *_all_metrics.json")
    ap.add_argument("--align_by", type=str, default="index", choices=["index", "seed"],
                    help="How to align repeated runs across models. Default: index (0..9).")
    args = ap.parse_args()

    metric = args.metric
    alpha = args.alpha
    higher_is_better = args.higher_is_better
    ascending = not higher_is_better

    paths = sorted(glob.glob(args.pattern))
    if len(paths) < 2:
        raise SystemExit(f"Need at least 2 JSON files matching pattern '{args.pattern}' in current folder.")

    # Load each model
    model_to_df = {}
    for p in paths:
        model = parse_model_name(p)
        df = load_metrics_json(p)
        if metric not in df.columns:
            raise SystemExit(f"Metric '{metric}' not found in {p}. Available columns: {list(df.columns)}")
        model_to_df[model] = df

    models = sorted(model_to_df.keys())

    # Align blocks
    if args.align_by == "index":
        # Use common run_id = 0..min_len-1
        lengths = {m: len(model_to_df[m]) for m in models}
        min_len = min(lengths.values())
        if min_len < 5:
            raise SystemExit(f"Too few runs per model (min length={min_len}). Need at least 5; you expect 10.")
        run_ids = list(range(min_len))
        print("Aligning by run index (run_id). Lengths:", lengths, f"-> using 0..{min_len-1}")
        index_col = "run_id"
    else:
        # Align by common seeds (only if seeds are meaningful and consistent)
        seed_sets = [set(model_to_df[m]["seed"].tolist()) for m in models]
        common = set.intersection(*seed_sets)
        common = sorted(list(common))
        if len(common) < 5:
            raise SystemExit(f"Too few common seeds across models: {len(common)}. "
                             f"Try --align_by index instead.")
        run_ids = common
        print(f"Aligning by seed. Common seeds N={len(run_ids)}")
        index_col = "seed"

    # Build blocks x models matrix
    mat = pd.DataFrame(index=run_ids, columns=models, dtype=float)
    for m in models:
        df = model_to_df[m].copy()
        s = df.set_index(index_col)[metric].astype(float)
        mat[m] = s.reindex(run_ids).values

    if mat.isna().any().any():
        raise SystemExit("NaN found after alignment. Check if some models are missing runs or metric values.")

    # Summary
    summary = pd.DataFrame({
        "mean": mat.mean(axis=0),
        "std": mat.std(axis=0, ddof=1)
    })
    summary["mean±std"] = summary["mean"].map(lambda x: f"{x:.6g}") + " ± " + summary["std"].map(lambda x: f"{x:.6g}")

    # Rank per block
    ranks = mat.rank(axis=1, method="average", ascending=ascending)
    avg_ranks = ranks.mean(axis=0).sort_values()

    print("\n====================== Models ======================")
    print(models)
    print(f"Blocks used (N={len(run_ids)}): {run_ids}")

    print("\n====================== Mean ± STD ===================")
    print(summary.sort_values("mean", ascending=ascending)[["mean±std"]])

    print("\n====================== Friedman =====================")
    arrays = [mat[m].values for m in models]
    fried_stat, fried_p = stats.friedmanchisquare(*arrays)
    print(f"metric = {metric}, higher_is_better = {higher_is_better}, align_by = {args.align_by}")
    print(f"Friedman chi2 = {fried_stat:.6g}, p = {fried_p:.6g}")

    print("\n====================== Avg Ranks ====================")
    print(avg_ranks.to_string())

    # Save
    out_input = f"significance_input_{metric}.csv"
    out_ranks = f"significance_ranks_{metric}.csv"
    out_summary = f"significance_summary_{metric}.csv"
    mat.to_csv(out_input, index_label="block")
    ranks.to_csv(out_ranks, index_label="block")
    summary.to_csv(out_summary, index_label="model")

    if fried_p < alpha:
        print("\nFriedman significant -> post-hoc...\n")

        # Nemenyi (if available)
        pmat, cd = nemenyi_posthoc(avg_ranks, n_blocks=len(run_ids), alpha=alpha)
        if pmat is not None:
            out_nem = f"posthoc_nemenyi_{metric}.csv"
            pmat.to_csv(out_nem, index_label="model")
            print("---- Nemenyi p-matrix ----")
            print(pmat.round(6).to_string())
            print(f"\nCD(alpha={alpha}) = {cd:.6g}")
            print(f"Saved: {out_nem}")
        else:
            print("---- Nemenyi not available (SciPy studentized_range missing). "
                  "You can rely on Wilcoxon+Holm below. ----\n")

        # Wilcoxon + Holm
        raw_p = {}
        for a, b in combinations(models, 2):
            try:
                _, p = stats.wilcoxon(mat[a].values, mat[b].values, zero_method="wilcox")
            except ValueError:
                p = 1.0
            raw_p[(a, b)] = float(p)

        holm_p = holm_correction(raw_p)
        df_w = pd.DataFrame([
            {"model_a": a, "model_b": b, "p_raw": raw_p[(a, b)], "p_holm": holm_p[(a, b)]}
            for (a, b) in raw_p.keys()
        ]).sort_values("p_holm", ascending=True)

        out_wil = f"posthoc_wilcoxon_holm_{metric}.csv"
        df_w.to_csv(out_wil, index=False)

        print("---- Wilcoxon + Holm ----")
        print(df_w.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        print(f"Saved: {out_wil}")
    else:
        print(f"\nFriedman not significant at alpha={alpha}. "
              f"You can still report mean±std and state no statistically significant differences.")

    print("\n====================== Files ========================")
    print(f"- {out_input}")
    print(f"- {out_ranks}")
    print(f"- {out_summary}")
    print("Done.")


if __name__ == "__main__":
    main()
