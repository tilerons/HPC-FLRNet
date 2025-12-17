import pandas as pd


def main() -> None:
    in_path = "./data/tmp_data1950_merged.xlsx"
    df = pd.read_excel(in_path, sheet_name="Sheet1", engine="openpyxl")

    target_columns = [
        "Cement",
        "Slag",
        "FlyAsh",
        "Water",
        "W_B",
        "W_C",
        "SP_C",
        "CoarseAgg",
        "FineAgg",
        "Age",
        "Strength",
    ]

    # Deduplicate by mix design + strength, keep the first occurrence (and its Source).
    df_dedup = df.drop_duplicates(subset=target_columns)

    out_path = "./data/tmp_data1925_dedup.xlsx"
    df_dedup.to_excel(out_path, index=False)
    print(f"Input shape:  {df.shape}")
    print(f"Dedup shape:  {df_dedup.shape}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


