import pandas as pd


def main() -> None:
    # Input files
    df1 = pd.read_excel("../data103/data/data103.xlsx", sheet_name="Sheet1", engine="openpyxl")
    df2 = pd.read_excel("../data714/data/data714.xlsx", sheet_name="Sheet1", engine="openpyxl")
    df3 = pd.read_excel("../data1133/data/data1133.xlsx", sheet_name="Sheet1", engine="openpyxl")

    # Unified feature set for downstream processing
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

    # Dataset 1 (data103)
    df1_rename = {
        "Cement": "Cement",
        "Slag": "Slag",
        "Fly ash": "FlyAsh",
        "Water": "Water",
        "W/C": "W_C",
        "W/B": "W_B",
        "SP": "SP",
        "SP/C": "SP_C",
        "Coarse Aggr.": "CoarseAgg",
        "Fine Aggr.": "FineAgg",
        "SLUMP(cm)": "Slump",
        "FLOW(cm)": "Flow",
        "Compressive Strength (28-day)(Mpa)": "Strength",
    }
    df1 = df1.rename(columns=df1_rename)
    df1["Age"] = 28
    df1 = df1[target_columns].copy()
    df1["Source"] = "data103"

    # Dataset 2 (data714)
    df2_rename = {
        "Compressive strength of cement fce (MPa)": "fce",
        "Tensile strength of cement fct (MPa)": "fct",
        "Curing age (day)": "Age",
        "Dmax of crushed stone (mm)": "DmaxStone",
        "Stone powder content in sand (%)": "StonePowderPct",
        "Fineness modulus of sand": "SandFineness",
        "W/B": "W_B",
        "Water to cement ratio, mw/mc": "W_C",
        "Water (kg/m3)": "Water",
        "Sand ratio (%)": "SandRatioPct",
        "Slump (mm)": "Slump",
        "SP_C": "SP_C",
        "Slag": "Slag",
        "FlyAsh": "FlyAsh",
        "CoarseAgg": "CoarseAgg",
        "FineAgg": "FineAgg",
        "Compressive strength, fcu,t (MPa)": "Strength",
        "Splitting tensile  strength, fst,t (MPa)": "fst",
    }
    df2 = df2.rename(columns=df2_rename)
    df2 = df2[target_columns].copy()
    df2["Source"] = "data714"

    # Dataset 3 (data1133)
    df3_rename = {
        "Cement (kg in a m^3 mixture)": "Cement",
        "Blast Furnace Slag (kg in a m^3 mixture)": "Slag",
        "Fly Ash (kg in a m^3 mixture)": "FlyAsh",
        "Water (kg in a m^3 mixture)": "Water",
        "Superplasticizer (kg in a m^3 mixture)": "SP",
        "Coarse Aggregate (kg in a m^3 mixture)": "CoarseAgg",
        "Fine Aggregate (kg in a m^3 mixture)": "FineAgg",
        "Age (day)": "Age",
        "Concrete compressive strength": "Strength",
    }
    df3 = df3.rename(columns=df3_rename)
    df3["SP_C"] = df3["SP"] / df3["Cement"]
    df3["W_C"] = df3["Water"] / df3["Cement"]
    df3["W_B"] = df3["Water"] / (df3["Cement"] + df3["Slag"] + df3["FlyAsh"])
    df3 = df3[target_columns].copy()
    df3["Source"] = "data1133"

    # Merge only
    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    # Drop rows with any missing values in the unified feature set.
    df_all = df_all.dropna(subset=target_columns)
    out_path = "./data/tmp_data1950_merged.xlsx"
    df_all.to_excel(out_path, index=False)
    print(f"Merged shape (with Source): {df_all.shape}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


