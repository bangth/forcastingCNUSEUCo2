# === File: src/model/SVR/model.py ===
from sklearn.svm import SVR

class SVRModel:
    def __init__(self):
        self.model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
