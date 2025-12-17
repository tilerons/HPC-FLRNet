import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


LONG_TO_SHORT = {
    "Cement": "Cem",
    "Slag": "Slg",
    "FlyAsh": "FA",
    "Water": "Wat",
    "W_B": "W/B",
    "W_C": "W/C",
    "SP_C": "SP/C",
    "CoarseAgg": "CA",
    "FineAgg": "FAgg",
    "Strength": "Strength",
    "Age": "Age",
}

SHORT_NUMERIC_COLUMNS = [
    "Cem",
    "Slg",
    "FA",
    "Wat",
    "W/B",
    "W/C",
    "SP/C",
    "CA",
    "FAgg",
    "Age",
    "Strength",
]


def _ensure_short_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {k: v for k, v in LONG_TO_SHORT.items() if k in df.columns and v not in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns=None,
    k: float = 1.5,
    how: str = "any",
):
    cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]

    outlier_flags = {}
    bounds = {}
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or not np.isfinite(iqr):
            outlier_flags[c] = pd.Series(False, index=df.index)
            bounds[c] = (None, None)
            continue
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        flag = (df[c] < lower) | (df[c] > upper)
        outlier_flags[c] = flag
        bounds[c] = (lower, upper)

    flags = pd.DataFrame(outlier_flags)
    outlier_any = flags.all(axis=1)

    inlier_mask = ~outlier_any
    df_clean = df.loc[inlier_mask].copy()
    outlier_counts = {col: outlier_flags[col].sum() for col in cols}
    return df_clean, inlier_mask, outlier_counts, bounds


def plot_figXa_curation_square(
    counts=(1950, 1925, 1826),
    labels=("Merged (aligned)", "After dedup", "HPC1826"),
    out_png="FigX_a_curation.png",
    out_svg="FigX_a_curation.svg",
    dpi=600,
):
    counts = np.array(counts, dtype=float)
    assert len(counts) == 3 and len(labels) == 3

    plt.rcParams.update(
        {
            "figure.figsize": (4.0, 4.0),
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots()
    x = np.arange(3)
    bar_width = 0.4
    colors = ["#B5B5B5", "#7D7D7D", "#2A6F97"]
    bars = ax.bar(x, counts, width=bar_width, color=colors, edgecolor="#222222", linewidth=0.6)

    ymax = counts.max()
    ax.set_ylim(0, ymax * 1.15)
    ax.set_ylabel("Number of samples")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.65, 2.65)

    for rect, val in zip(bars, counts):
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h - ymax * 0.06,
            f"{int(val)}",
            ha="center",
            va="top",
            fontsize=9,
            color="#111111",
            fontweight="normal",
        )

    d1 = counts[1] - counts[0]
    d2 = counts[2] - counts[1]
    outlier_pct = (-d2) / counts[1] * 100 if counts[1] > 0 else np.nan

    def arrow_with_text(x0, x1, y, text):
        arr = FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=0.8,
            color="#333333",
        )
        ax.add_patch(arr)
        ax.text((x0 + x1) / 2, y + ymax * 0.03, text, ha="center", va="bottom", fontsize=8.5, color="#333333")

    y_arrow_1 = ymax * 1.02
    y_arrow_2 = ymax * 1.04
    arrow_with_text(x[0], x[1], y_arrow_1, f"{int(d1)}  (dedup)")
    arrow_with_text(x[1], x[2], y_arrow_2, f"{int(d2)}  (IQR, {outlier_pct:.2f}%)")

    ax.set_title("(a) Curation pipeline")
    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved: {out_png}, {out_svg}")


def plot_pca_by_source(
    df: pd.DataFrame,
    x_cols,
    source_col="Source",
    save_path="HPC1826_with_source_pca_by_source.svg",
    random_state=2025,
):
    rawX = df[x_cols].values
    source_labels = df[source_col].values if source_col in df.columns else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rawX)

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(6, 5))
    if source_labels is None:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=18, alpha=0.7, edgecolors="none")
    else:
        source_labels = np.array(source_labels)
        unique_sources = np.unique(source_labels)
        color_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        for i, src in enumerate(unique_sources):
            mask = source_labels == src
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                s=18,
                alpha=0.7,
                label=f"{src}",
                edgecolors="none",
                c=color_palette[i % len(color_palette)],
            )
        plt.legend(fontsize=9, frameon=True, framealpha=0.8, loc="best")

    plt.xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)", fontsize=11)
    plt.ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)", fontsize=11)
    plt.title("(b) PCA of HPC1826 feature space by data source", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main() -> None:
    in_path = "./data/tmp_data1925_dedup.xlsx"
    df = pd.read_excel(in_path, sheet_name="Sheet1", engine="openpyxl")
    df = _ensure_short_columns(df)

    print("Missing values (numeric columns):")
    if all(c in df.columns for c in SHORT_NUMERIC_COLUMNS):
        print(df[SHORT_NUMERIC_COLUMNS].isnull().sum())
    else:
        missing = [c for c in SHORT_NUMERIC_COLUMNS if c not in df.columns]
        raise KeyError(f"Missing required columns: {missing}")

    df_numeric = df[SHORT_NUMERIC_COLUMNS].dropna()

    df_clean2, inlier_mask, outlier_counts, bounds = remove_outliers_iqr(
        df_numeric, SHORT_NUMERIC_COLUMNS, k=1.5, how="all"
    )
    df_clean2 = df_clean2.drop_duplicates()

    print(f"Before IQR: {len(df_numeric)}")
    print(f"After  IQR: {len(df_clean2)}")

    print("Outlier counts by column:")
    for col, count in outlier_counts.items():
        print(f"{col}: {count}")

    df_clean_full = df.loc[df_clean2.index].copy()
    df_clean_full = df_clean_full.drop_duplicates(subset=SHORT_NUMERIC_COLUMNS)

    out_xlsx = "./data/HPC1826_with_source.xlsx"
    df_clean_full.to_excel(out_xlsx, index=False)
    print(f"Final cleaned shape: {df_clean_full.shape}")
    print(f"Saved: {out_xlsx}")

    plot_figXa_curation_square(
        counts=(1950, 1925, 1826),
        labels=("Merged (aligned)", "After dedup", "HPC1826"),
        out_png="FigX_a_curation.png",
        out_svg="FigX_a_curation.svg",
        dpi=600,
    )

    x_cols = ["Cem", "Slg", "FA", "Wat", "W/B", "W/C", "SP/C", "CA", "FAgg", "Age"]
    pca_path = os.path.splitext(os.path.basename(out_xlsx))[0] + "_pca_by_source.svg"
    plot_pca_by_source(df_clean_full, x_cols=x_cols, save_path=pca_path, random_state=2025)


if __name__ == "__main__":
    main()


