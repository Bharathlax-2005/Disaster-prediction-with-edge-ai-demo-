"""
Model Training Module
Trains separate ML models for each disaster type
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import json
from typing import Dict, Tuple


class DisasterModelTrainer:
    """Train and evaluate disaster prediction models"""
    
    def __init__(self, processed_data_dir: str = "data/processed", models_dir: str = "models"):
        self.processed_data_dir = Path(processed_data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.evaluation_results = {}
    
    def load_data(self, disaster_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and split processed data"""
        data_path = self.processed_data_dir / f"{disaster_type}_processed.npz"
        data = np.load(data_path)
        X, y = data['X'], data['y']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_flood_model(self) -> RandomForestClassifier:
        """Train Random Forest model for flood prediction"""
        print("\n🌊 Training Flood Prediction Model...")
        
        X_train, X_test, y_train, y_test = self.load_data('flood')
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        self._evaluate_model('flood', y_test, y_pred)
        
        # Save model
        model_path = self.models_dir / "flood_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.models['flood'] = model
        print(f"✅ Flood model saved to {model_path}")
        
        return model
    
    def train_earthquake_model(self) -> xgb.XGBClassifier:
        """Train XGBoost model for earthquake prediction"""
        print("\n🌋 Training Earthquake Prediction Model...")
        
        X_train, X_test, y_train, y_test = self.load_data('earthquake')
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        self._evaluate_model('earthquake', y_test, y_pred)
        
        # Save model
        model_path = self.models_dir / "earthquake_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.models['earthquake'] = model
        print(f"✅ Earthquake model saved to {model_path}")
        
        return model
    
    def train_cyclone_model(self) -> GradientBoostingClassifier:
        """Train Gradient Boosting model for cyclone prediction"""
        print("\n🌪️ Training Cyclone Prediction Model...")
        
        X_train, X_test, y_train, y_test = self.load_data('cyclone')
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        self._evaluate_model('cyclone', y_test, y_pred)
        
        # Save model
        model_path = self.models_dir / "cyclone_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.models['cyclone'] = model
        print(f"✅ Cyclone model saved to {model_path}")
        
        return model
    
    def _evaluate_model(self, disaster_type: str, y_true: np.ndarray, y_pred: np.ndarray):
        """Evaluate model performance"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        self.evaluation_results[disaster_type] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        print(f"\n📊 {disaster_type.upper()} Model Evaluation:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
    
    def train_all_models(self):
        """Train all disaster prediction models"""
        print("🚀 Starting Multi-Disaster Model Training Pipeline...\n")
        
        self.train_flood_model()
        self.train_earthquake_model()
        self.train_cyclone_model()
        
        # Save evaluation results
        results_path = self.models_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"\n✅ All models trained successfully!")
        print(f"📋 Evaluation results saved to {results_path}")
        
        return self.models


if __name__ == "__main__":
    trainer = DisasterModelTrainer()
    trainer.train_all_models()