# -*- coding: utf-8 -*-
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
import os
init_random_state = 2025
RUN_TIME = time.strftime("%Y_%m_%d_%H_%M", time.localtime())


def save_results(obj, filename, out_dir=None):
    import json
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = filename if os.path.isabs(filename) else os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def scores_to_all_metrics(scores, base_seed, n_runs):
    all_metrics = []
    r2_list = scores.get("r2_list", [])
    rmse_list = scores.get("rmse_list", [])
    mae_list = scores.get("mae_list", [])
    mape_list = scores.get("mape_list", [])
    r_list = scores.get("r_list", [])

    for i in range(n_runs):
        all_metrics.append({
            "seed": int(base_seed + i),
            "rmse": float(rmse_list[i]),
            "r2": float(r2_list[i]),
            "r": float(r_list[i]) if len(r_list) > i else float(np.sqrt(r2_list[i])),
            "mae": float(mae_list[i]),
            "mape": float(mape_list[i]),
        })
    return all_metrics


def scores_to_summary(scores):
    return {
        "rmse_mean": float(scores.get("rmse_mean")),
        "rmse_std": float(scores.get("rmse_std")),
        "r2_mean": float(scores.get("r2_mean")),
        "r2_std": float(scores.get("r2_std")),
        "r_mean": float(scores.get("r_mean")),
        "r_std": float(scores.get("r_std")),
        "mae_mean": float(scores.get("mae_mean")),
        "mae_std": float(scores.get("mae_std")),
        "mape_mean": float(scores.get("mape_mean")),
        "mape_std": float(scores.get("mape_std")),
    }


def run_svr(random_state=0, poly_degree=1, kernel="rbf", C=1000, epsilon=0.04,
            gamma=0.5, problem="compressive", n_runs=10):
    """
    SVR avg±std
    """
    print("Running SVR for %s data with "
          "random_state=%s, poly_degree=%s, kernel=%s, C=%s, epsilon=%s, gamma=%s, n_runs=%s" %
          (problem, random_state, poly_degree, kernel, C, epsilon, gamma, n_runs))

    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, degree=poly_degree)
    if kernel == "poly":
        poly_degree = 1

    scores = Starts(random_state, poly_degree, model,
                    problem=problem, test_size=0.2, print_results=True, n_runs=n_runs)

    print("Finished running SVR for %s data with "
          "random_state=%s, poly_degree=%s, kernel=%s, C=%s, epsilon=%s, gamma=%s, n_runs=%s\n" %
          (problem, random_state, poly_degree, kernel, C, epsilon, gamma, n_runs))

    return scores


def run_mlp(random_state=0, poly_degree=3, hd_layer_1=300, hd_layer_2=100,
            solver="lbfgs", max_iter=1000, alpha=0, problem="compressive", n_runs=10):
    """
    MLP avg±std
    """
    print("Running MLP for %s data with "
          "random_state=%s, poly_degree=%s, hd_layer_1=%s, hd_layer_2=%s, "
          "solver=%s, max_iter=%s, alpha=%s, n_runs=%s" %
          (problem, random_state, poly_degree, hd_layer_1, hd_layer_2, solver, max_iter, alpha, n_runs))

    hd_layers = (hd_layer_1, hd_layer_2,)
    if hd_layer_2 < 0:
        hd_layers = (hd_layer_1,)

    model = MLPRegressor(
        warm_start=False,
        random_state=random_state,
        hidden_layer_sizes=hd_layers,
        solver=solver,
        max_iter=max_iter,
        alpha=alpha
    )

    scores = Starts(random_state, poly_degree, model,
                    problem=problem, test_size=0.2, print_results=True, n_runs=n_runs)

    print("Finished running MLP for %s data with "
          "random_state=%s, poly_degree=%s, hd_layer_1=%s, hd_layer_2=%s, "
          "solver=%s, max_iter=%s, alpha=%s, n_runs=%s\n" %
          (problem, random_state, poly_degree, hd_layer_1, hd_layer_2, solver, max_iter, alpha, n_runs))

    return scores


def run_gbr(random_state=0, poly_degree=2, n_estimators=1000,
            max_depth=5, learning_rate=0.1, loss="huber",
            problem="compressive", n_runs=10):
    """
    GBR avg±std
    """
    print("Running GBR for %s data with random_state=%s, poly_degree=%s, "
          "n_estimators=%s, max_depth=%s, learning_rate=%s, loss=%s, "
          "n_runs=%s" %
          (problem, random_state, poly_degree, n_estimators, max_depth,
           learning_rate, loss, n_runs))

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        loss=loss,
        random_state=random_state
    )

    scores = Starts(random_state, poly_degree, model,
                    problem=problem, test_size=0.2, print_results=True, n_runs=n_runs)

    print("Finished running GBR for %s data with random_state=%s, poly_degree=%s, "
          "n_estimators=%s, max_depth=%s, learning_rate=%s, loss=%s, "
          "n_runs=%s\n" %
          (problem, random_state, poly_degree, n_estimators, max_depth,
           learning_rate, loss, n_runs))

    return scores


def run_xgb(random_state=0, poly_degree=1, n_estimators=1000, max_depth=4,
            learning_rate=0.2, objective="reg:logistic", problem="compressive", n_runs=10):
    """
    XGB avg±std
    """
    print("Running XGB for %s data with random_state=%s, poly_degree=%s, "
          "n_estimators=%s, max_depth=%s, learning_rate=%s, objective=%s, n_runs=%s" %
          (problem, random_state, poly_degree, n_estimators, max_depth,
           learning_rate, objective, n_runs))

    model = XGBRegressor(
        poly_degree=poly_degree,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective=objective,
        random_state=random_state
    )

    scores = Starts(random_state, poly_degree, model,
                    problem=problem, test_size=0.2, print_results=True, n_runs=n_runs)

    print("Finished running XGB for %s data with random_state=%s, poly_degree=%s, "
          "n_estimators=%s, max_depth=%s, learning_rate=%s, objective=%s, n_runs=%s\n" %
          (problem, random_state, poly_degree, n_estimators, max_depth,
           learning_rate, objective, n_runs))

    return scores


def set_seed(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_absolute_percentage_error(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


def Starts(random_state,
           poly_degree,
           model,
           problem="hpc1826.xlsx",
           test_size=0.2,
           print_results=True,
           n_runs=10):
    set_seed(init_random_state)

    data = pd.read_excel('./data/' + problem,
                         sheet_name='Sheet1',
                         engine='openpyxl')
    data = data.values
    n_data_cols = np.shape(data)[1]
    n_features = n_data_cols - 1

    X = np.array(data[:, :n_features])
    y = np.array(data[:, n_features:])

    r2_list = []
    r_list = []
    rmse_list = []
    mae_list = []
    mape_list = []

    for i in range(n_runs):
        seed = random_state + i
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            shuffle=True
        )

        X_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)

        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train_scaled = y_scaler.fit_transform(y_train)

        if poly_degree >= 1:
            poly = PolynomialFeatures(degree=poly_degree)
            X_train_scaled = poly.fit_transform(X_train_scaled)
            X_test_scaled = poly.transform(X_test_scaled)

        model_run = clone(model)
        if hasattr(model_run, "random_state"):
            model_run.random_state = seed

        # train
        model_run.fit(X_train_scaled, y_train_scaled.ravel())

        # predict
        y_pred_scaled = model_run.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

        r2 = r2_score(y_test, y_pred)
        r_lcc = r2 ** 0.5
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        r2_list.append(r2)
        r_list.append(r_lcc)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)

        if print_results:
            print("=" * 60)
            print(f"Run {i + 1}/{n_runs} | seed={seed}")
            print("R²:  {0:.5f}".format(r2))
            print("R:   {0:.5f}".format(r_lcc))
            print("RMSE (MPa): {0:.5f}".format(rmse))
            print("MAE  (MPa): {0:.5f}".format(mae))
            print("MAPE (%):   {0:.5f}".format(mape))

    # ddof=1
    r2_mean = np.mean(r2_list)
    r2_std = np.std(r2_list, ddof=1) if n_runs > 1 else 0.0

    r_mean = np.mean(r_list)
    r_std = np.std(r_list, ddof=1) if n_runs > 1 else 0.0

    rmse_mean = np.mean(rmse_list)
    rmse_std = np.std(rmse_list, ddof=1) if n_runs > 1 else 0.0

    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list, ddof=1) if n_runs > 1 else 0.0

    mape_mean = np.mean(mape_list)
    mape_std = np.std(mape_list, ddof=1) if n_runs > 1 else 0.0

    if print_results:
        print("\n" + "#" * 60)
        print(f"Summary over {n_runs} runs (problem = {problem}):")
        print("R²:   {:.3f}±{:.3f}".format(r2_mean, r2_std))
        print("R:   {:.3f}±{:.3f}".format(r_mean, r_std))
        print("RMSE: {:.3f}±{:.3f} MPa".format(rmse_mean, rmse_std))
        print("MAE:  {:.3f}±{:.3f} MPa".format(mae_mean, mae_std))
        print("MAPE: {:.3f}±{:.3f} %".format(mape_mean, mape_std))
        print("#" * 60 + "\n")

    scores = {
        "r2_list": r2_list,
        "rmse_list": rmse_list,
        "mae_list": mae_list,
        "mape_list": mape_list,
        "r2_mean": float(r2_mean),
        "r2_std": float(r2_std),
        "rmse_mean": float(rmse_mean),
        "rmse_std": float(rmse_std),
        "mae_mean": float(mae_mean),
        "mae_std": float(mae_std),
        "mape_mean": float(mape_mean),
        "mape_std": float(mape_std),
        "r_list": r_list,
        "r_mean": float(r_mean),
        "r_std": float(r_std)
    }

    return scores


if __name__ == '__main__':
    problem = "hpc1826.xlsx"
    set_seed(init_random_state)
    n_runs = 10

    svr_scores = run_svr(random_state=init_random_state,
                         poly_degree=1,
                         kernel="rbf",
                         C=1000,
                         epsilon=0.04,
                         gamma=0.5,
                         problem=problem,
                         n_runs=n_runs)

    mlp_scores = run_mlp(random_state=init_random_state,
                         hd_layer_1=300,
                         hd_layer_2=100,
                         max_iter=1000,
                         alpha=0,
                         problem=problem,
                         n_runs=n_runs)

    gbr_scores = run_gbr(random_state=init_random_state,
                         poly_degree=2,
                         n_estimators=1000,
                         max_depth=5,
                         learning_rate=0.2,
                         loss="huber",
                         problem=problem,
                         n_runs=n_runs)

    xgb_scores = run_xgb(random_state=init_random_state,
                         poly_degree=2,
                         n_estimators=1000,
                         max_depth=6,
                         learning_rate=0.2,
                         objective="reg:logistic",
                         problem=problem,
                         n_runs=n_runs)

    svr_all_metrics = scores_to_all_metrics(svr_scores, base_seed=init_random_state, n_runs=n_runs)
    mlp_all_metrics = scores_to_all_metrics(mlp_scores, base_seed=init_random_state, n_runs=n_runs)
    gbr_all_metrics = scores_to_all_metrics(gbr_scores, base_seed=init_random_state, n_runs=n_runs)
    xgb_all_metrics = scores_to_all_metrics(xgb_scores, base_seed=init_random_state, n_runs=n_runs)

    save_results(svr_all_metrics, f"{RUN_TIME}_SVR_all_metrics.json")
    save_results(mlp_all_metrics, f"{RUN_TIME}_MLP_all_metrics.json")
    save_results(gbr_all_metrics, f"{RUN_TIME}_GBR_all_metrics.json")
    save_results(xgb_all_metrics, f"{RUN_TIME}_XGB_all_metrics.json")