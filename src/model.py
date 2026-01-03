import xgboost as xgb
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

class CryptoModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.1):
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )

    def train(self, X_train, y_train):
        
        print("Training XGBoost model...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {acc:.4f}")
        print("Detailed Report:")
        print(classification_report(y_test, y_pred))
        return acc

    def predict(self, X):
        
        return self.model.predict(X)

    def predict_proba(self, X):

        # [:, 1] gives the probability of class 1 (Up)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to: {path}")

    def load(self, path: str):

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model = joblib.load(path)
        print(f"Model loaded from: {path}")