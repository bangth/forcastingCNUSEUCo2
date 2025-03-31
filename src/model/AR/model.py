import numpy as np
from statsmodels.tsa.ar_model import AutoReg


class ARModel:
    def __init__(self, lags=7):
        self.lags = lags
        self.model = None
        self.fitted_model = None

    def fit(self, train_series):
        """
        train_series: 1D numpy array (e.g., one node's emission)
        """
        self.model = AutoReg(train_series, lags=self.lags, old_names=False)
        self.fitted_model = self.model.fit()

    def predict(self, steps=1):
        """
        steps: number of steps to predict (e.g., 1 or 3)
        """
        return self.fitted_model.predict(start=len(self.fitted_model.model.endog), end=len(self.fitted_model.model.endog) + steps - 1)

    def evaluate(self, y_true, y_pred):
        """
        Compute evaluation metrics
        """
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        return {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'RMSPE': rmspe,
            'R2': r2
        }
