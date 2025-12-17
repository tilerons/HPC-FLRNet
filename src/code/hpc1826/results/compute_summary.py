"""
读取 all_metrics.json 文件，计算并输出各指标的均值和标准差
"""
import json
import numpy as np

def compute_summary(json_path: str) -> dict:
    """读取JSON文件并计算均值±标准差"""
    with open(json_path, 'r', encoding='utf-8') as f:
        all_metrics = json.load(f)
    
    rmse_arr = np.array([m["rmse"] for m in all_metrics])
    r2_arr   = np.array([m["r2"]   for m in all_metrics])
    r_arr    = np.array([m["r"]    for m in all_metrics])
    mae_arr  = np.array([m["mae"]  for m in all_metrics])
    mape_arr = np.array([m["mape"] for m in all_metrics])

    summary = {
        "rmse_mean": rmse_arr.mean(),
        "rmse_std":  rmse_arr.std(ddof=1),
        "r2_mean":   r2_arr.mean(),
        "r2_std":    r2_arr.std(ddof=1),
        "r_mean":    r_arr.mean(),
        "r_std":     r_arr.std(ddof=1),
        "mae_mean":  mae_arr.mean(),
        "mae_std":   mae_arr.std(ddof=1),
        "mape_mean": mape_arr.mean(),
        "mape_std":  mape_arr.std(ddof=1),
    }
    return summary


def print_summary(summary: dict):
    """格式化打印统计结果"""
    print("=" * 50)
    print("各指标统计结果 (均值 ± 标准差)")
    print("=" * 50)
    print(f"RMSE: {summary['rmse_mean']:.3f}±{summary['rmse_std']:.3f}")
    print(f"R²:   {summary['r2_mean']:.3f}±{summary['r2_std']:.3f}")
    print(f"R:    {summary['r_mean']:.3f}±{summary['r_std']:.3f}")
    print(f"MAE:  {summary['mae_mean']:.3f}±{summary['mae_std']:.3f}")
    print(f"MAPE: {summary['mape_mean']:.3f}±{summary['mape_std']:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    json_path = "../../../data/results/significance/hpc1826/2025_12_14_17_16_HPC-FLRNet_all_metrics.json"
    summary = compute_summary(json_path)
    print_summary(summary)

