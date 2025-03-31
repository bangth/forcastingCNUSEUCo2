# === File: src/model/SVR/utils.py ===
import os
import numpy as np
import pandas as pd


def load_data(dataset, horizon):
    # Load từ data/CN, data/EU, data/US - cấu trúc bạn đã có
    data_path = f"data/{dataset}/prepared_data_h{horizon}.npz"
    data = np.load(data_path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


def save_metrics(model_name, dataset, horizon, mae, mape, rmse, rmspe, r2):
    result_path = "results/metrics_results.csv"
    os.makedirs("results", exist_ok=True)
    result = pd.DataFrame([{
        "model": model_name,
        "dataset": dataset,
        "horizon": horizon,
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "RMSPE": rmspe,
        "R2": r2
    }])
    if os.path.exists(result_path):
        result.to_csv(result_path, mode='a', header=False, index=False)
    else:
        result.to_csv(result_path, index=False)
