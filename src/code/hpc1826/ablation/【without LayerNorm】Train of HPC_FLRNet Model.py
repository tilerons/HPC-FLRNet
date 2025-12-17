import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import logging
import time
import random
from sklearn.model_selection import train_test_split

def set_logger(log_path):
    directory = os.path.dirname(log_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def set_seed(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


RUN_TIME = time.strftime("%Y_%m_%d_%H_%M", time.localtime())


def mean_absolute_percentage_error(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    non_zero_mask = y_test != 0
    return np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100

set_logger('./' + (os.path.splitext(os.path.basename(__file__))[0]) + '_' + RUN_TIME + '.log')

CONFIG = {
    "random_state": 2025,
    "epochs": 4000,
    "hidden_dim": 0,
    "dropout_rate": 0.1,
    "early_stopping_patience": 400,
    "lr": '',
    "weight_decay": 1e-4,
    "scheduler_patience": 40,
    "scheduler_factor": 0.6,
    "batch_size": 8
}

#========recommended configuration===========
hidden_dim = 256
lr = 0.00027
base_seed = 2436
#========recommended configuration===========

logging.info("CONFIG：" + str(CONFIG))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"(Using device): {device}")

class ConcreteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class InputProjectionModule(nn.Module):
    """IPM"""

    def __init__(self, input_dim, hidden_dim):
        super(InputProjectionModule, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.linear(x))


class FeatureRefinementModule(nn.Module):
    """FRM"""

    def __init__(self, dim, dropout_rate):
        super(FeatureRefinementModule, self).__init__()
        #self.norm = nn.LayerNorm(dim) without LayerNorm layer
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return residual + x


class CompressionRegressionModule(nn.Module):
    """CRM"""

    def __init__(self, hidden_dim, dropout_rate):
        super(CompressionRegressionModule, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x)


class HPC_FLRNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, dropout_rate=0.2):
        super(HPC_FLRNet, self).__init__()
        self.input_projection = InputProjectionModule(input_dim, hidden_dim)
        self.feature_refinement = FeatureRefinementModule(hidden_dim, dropout_rate)
        self.regression_head = CompressionRegressionModule(hidden_dim, dropout_rate)
    def forward(self, x):
        x = self.input_projection(x)
        x = self.feature_refinement(x)
        return self.regression_head(x)

def train_and_eval(model, rawX_train, rawX_test, rawY_train, rawY_test, config):
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_factor'],
                                                     patience=config['scheduler_patience'], verbose=False)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(rawX_train)
    X_test = X_scaler.transform(rawX_test)
    Y_train = y_scaler.fit_transform(rawY_train)

    train_loader = DataLoader(ConcreteDataset(X_train, Y_train),
                              batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(ConcreteDataset(X_test, rawY_test),
                             batch_size=config['batch_size'], shuffle=False, num_workers=0)

    epochs = config['epochs']
    early_stopping_patience = config['early_stopping_patience']
    early_stop_counter = 0

    best_rmse = float('inf')
    best_preds = None
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_inverse_preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                val_inverse_preds.append(outputs.cpu().numpy())
        val_inverse_preds = y_scaler.inverse_transform(np.concatenate(val_inverse_preds))

        r2 = r2_score(rawY_test, val_inverse_preds)
        r_lcc = np.sqrt(r2)
        rmse = np.sqrt(mean_squared_error(rawY_test, val_inverse_preds))
        mae = mean_absolute_error(rawY_test, val_inverse_preds)
        mape = mean_absolute_percentage_error(rawY_test, val_inverse_preds)
        scheduler.step(rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            early_stop_counter = 0
            best_preds = val_inverse_preds
            logging.info(f"Get best in Epoch: {epoch + 1} | R²: {r2:.4f} | R: {r_lcc:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    final_pred = best_preds
    r2 = r2_score(rawY_test, final_pred)
    r_lcc = np.sqrt(r2)
    rmse = np.sqrt(mean_squared_error(rawY_test, final_pred))
    mae = mean_absolute_error(rawY_test, final_pred)
    mape = mean_absolute_percentage_error(rawY_test, final_pred)
    logging.info(f"result loging: R²: {r2:.4f} | R: {r_lcc:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.4f}")
    return r2, r_lcc, rmse, mae, mape


def Starts(rawX, rawY, test_size=0.2, random_state=42, n_runs=10):
    logging.info("\n\nrepeat runs")

    rawX_train, rawX_test, rawY_train, rawY_test = train_test_split(
        rawX, rawY,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    logging.info(f"hidden_dim={hidden_dim}, lr={lr}, base_seed={base_seed}")
    all_metrics = []

    for i in range(n_runs):
        seed = base_seed + i
        set_seed(seed)

        config = CONFIG.copy()
        config['hidden_dim'] = hidden_dim
        config['lr'] = lr

        logging.info(f"\n[Run {i+1}/{n_runs}] seed={seed}")

        model = HPC_FLRNet(
            input_dim=rawX_train.shape[1],
            hidden_dim=config['hidden_dim'],
            dropout_rate=config['dropout_rate']
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        model.to(device)

        r2, r_lcc, rmse, mae, mape = train_and_eval(
            model, rawX_train, rawX_test, rawY_train, rawY_test, config
        )

        logging.info(
            f"[Run {i+1}] RMSE={rmse:.4f}, R²={r2:.4f}, R={r_lcc:.4f}, "
            f"MAE={mae:.4f}, MAPE={mape:.4f}"
        )

        all_metrics.append({
            "seed": seed,
            "rmse": rmse,
            "r2": r2,
            "r": r_lcc,
            "mae": mae,
            "mape": mape
        })

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

    logging.info("\nresults")
    logging.info(
        "RMSE = {:.4f} ± {:.4f}, R² = {:.4f} ± {:.4f}, R = {:.4f} ± {:.4f}, "
        "MAE = {:.4f} ± {:.4f}, MAPE = {:.4f} ± {:.4f}".format(
            summary["rmse_mean"], summary["rmse_std"],
            summary["r2_mean"], summary["r2_std"],
            summary["r_mean"], summary["r_std"],
            summary["mae_mean"], summary["mae_std"],
            summary["mape_mean"], summary["mape_std"],
        )
    )
    return all_metrics, summary

def save_results(results, filename='results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    set_seed(CONFIG['random_state'])
    data_file = "hpc1826.xlsx"
    x_cols = []
    y_cols = []
    if 'data103' in data_file:
        x_cols = ['Cement', 'Slag', 'Fly ash', 'Water', 'W/C', 'W/B', 'SP', 'SP/C', 'Coarse Aggr.', 'Fine Aggr.']
        y_cols = ['Compressive Strength (28-day)(Mpa)']
    elif 'data714' in data_file:
        x_cols = ['Compressive strength of cement fce (MPa)','Curing age (day)','Dmax of crushed stone (mm)','Stone powder content in sand (%)', 'Fineness modulus of sand', 'W/B', 'Water to cement ratio, mw/mc', 'Water (kg/m3)', 'Sand ratio (%)']
        y_cols = ['Compressive strength,fcu,t(MPa)']
    elif 'data1133' in data_file:
        x_cols = ['Cement (kg in a m^3 mixture)', 'Blast Furnace Slag (kg in a m^3 mixture)', 'Fly Ash (kg in a m^3 mixture)', 'Water (kg in a m^3 mixture)', 'Superplasticizer (kg in a m^3 mixture)', 'Coarse Aggregate (kg in a m^3 mixture)', 'Fine Aggregate (kg in a m^3 mixture)', 'Age (day)']
        y_cols = ['Concrete compressive strength']
    elif 'hpc1826' in data_file:
        x_cols = ['Cement', 'Slag', 'FlyAsh', 'Water', 'W_B', 'W_C', 'SP_C', 'CoarseAgg', 'FineAgg', 'Age']
        y_cols = ['Strength']

    data = pd.read_excel('../data/' + data_file, sheet_name='Sheet1', engine='openpyxl')
    data = data.values

    all_cols = x_cols + y_cols
    data_df = pd.DataFrame(data)
    data_df.columns = all_cols

    logging.info(f"{data_file}: input feature {len(x_cols)}: {x_cols}")

    rawX = data_df[x_cols].values
    rawY = data_df[y_cols].values

    # start
    all_metrics, summary = Starts(rawX, rawY, test_size=0.2, random_state=CONFIG['random_state'], n_runs=10)
    save_results(all_metrics, RUN_TIME + '_all_metrics' + '.json')
    save_results(summary, RUN_TIME + '_summary' + '.json')
