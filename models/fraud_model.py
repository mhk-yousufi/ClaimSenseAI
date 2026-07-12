import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class FraudDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Mock training data to fit the model for the prototype
        X_mock = np.random.rand(100, 4) * [10000, 80, 5, 10] # Amount, Age, Prev Claims, Vehicle Age
        y_mock = np.random.randint(0, 2, 100)
        self.model.fit(X_mock, y_mock)

    def predict_risk(self, amount, age, prev_claims, vehicle_age):
        features = np.array([[amount, age, prev_claims, vehicle_age]])
        risk_prob = self.model.predict_proba(features)[0][1]
        return risk_prob * 100, features