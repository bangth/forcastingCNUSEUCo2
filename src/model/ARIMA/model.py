import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    def __init__(self, order=(5, 1, 0)):
        """
        ARIMA(p,d,q): p = autoregressive lags, d = differencing, q = moving average lags
        """
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, train_series):
        """
        train_series: 1D numpy array (e.g., emissions of one node)
        """
        self.model = ARIMA(train_series, order=self.order)
        self.fitted_model = self.model.fit()

    def predict(self, steps=1):
        """
        steps: number of steps to predict
        """
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def evaluate(self, y_true, y_pred):
        """
        Compute evaluation metrics
        """
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmspe = np.sqrt(np.mean(((y_true - y_pred) / (y_true + 1e-6)) ** 2)) * 100
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-6)

        return {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'RMSPE': rmspe,
            'R2': r2
        }
