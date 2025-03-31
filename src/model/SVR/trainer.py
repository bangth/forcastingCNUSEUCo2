# === File: src/model/SVR/trainer.py ===
import numpy as np
from model import SVRModel
from utils import load_data, save_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate(dataset, horizon):
    X_train, y_train, X_test, y_test = load_data(dataset, horizon)

    model = SVRModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100

    save_metrics("SVR", dataset, horizon, mae, mape, rmse, rmspe, r2)