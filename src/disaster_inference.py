"""
Unified Disaster Inference Engine
Routes data to appropriate models and returns predictions
"""

import tensorflow as tf
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple
import time


class UnifiedDisasterInferenceEngine:
    """Unified inference engine for all disaster types"""
    
    def __init__(self, models_dir: str = "models", use_tflite: bool = True):
        self.models_dir = Path(models_dir)
        self.use_tflite = use_tflite
        
        # Load configuration
        with open("configs/model_mapping.json", 'r') as f:
            self.config = json.load(f)
        
        # Load scalers
        self.scalers = self._load_scalers()
        
        # Load models
        self.models = self._load_models()
        
        print("✅ Unified Inference Engine initialized")
        print(f"   Model type: {'TFLite (Edge)' if use_tflite else 'Original (Full)'}")
    
    def _load_scalers(self) -> Dict:
        """Load feature scalers for all disaster types"""
        scalers = {}
        scaler_dir = Path("data/processed")
        
        for disaster in ['flood', 'earthquake', 'cyclone']:
            scaler_path = scaler_dir / f"{disaster}_scaler.pkl"
            with open(scaler_path, 'rb') as f:
                scalers[disaster] = pickle.load(f)
        
        return scalers
    
    def _load_models(self) -> Dict:
        """Load inference models (TFLite or original)"""
        models = {}
        
        for disaster in ['flood', 'earthquake', 'cyclone']:
            if self.use_tflite:
                # Load TFLite model
                tflite_path = self.models_dir / f"{disaster}_tflite_model.tflite"
                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                interpreter.allocate_tensors()
                models[disaster] = interpreter
            else:
                # Load original pickle model
                pkl_path = self.models_dir / f"{disaster}_model.pkl"
                with open(pkl_path, 'rb') as f:
                    models[disaster] = pickle.load(f)
        
        return models
    
    def _predict_tflite(self, interpreter, input_data: np.ndarray) -> float:
        """Run inference on TFLite model"""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        input_data = input_data.astype(np.float32).reshape(1, -1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        return float(output[0][0])
    
    def predict_disaster(self, disaster_type: str, raw_data: Dict) -> Tuple[float, float, str]:
        """
        Predict disaster risk
        
        Args:
            disaster_type: 'flood', 'earthquake', or 'cyclone'
            raw_data: Dictionary with feature values
        
        Returns:
            (risk_score, inference_time_ms, risk_level)
        """
        start_time = time.time()
        
        # Extract features based on disaster type
        features = self.config[disaster_type]['features']
        feature_values = np.array([raw_data[f] for f in features])
        
        # Scale features
        scaled_features = self.scalers[disaster_type].transform(feature_values.reshape(1, -1))
        
        # Predict
        if self.use_tflite:
            risk_score = self._predict_tflite(self.models[disaster_type], scaled_features[0])
        else:
            prediction_proba = self.models[disaster_type].predict_proba(scaled_features)
            risk_score = float(prediction_proba[0][1])
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Determine risk level
        risk_level = self._get_risk_level(disaster_type, risk_score)
        
        return risk_score, inference_time, risk_level
    
    def _get_risk_level(self, disaster_type: str, risk_score: float) -> str:
        """Determine risk level based on thresholds"""
        with open("configs/thresholds.json", 'r') as f:
            thresholds = json.load(f)[disaster_type]
        
        if risk_score < thresholds['low']:
            return "SAFE"
        elif risk_score < thresholds['medium']:
            return "LOW"
        elif risk_score < thresholds['high']:
            return "MEDIUM"
        elif risk_score < thresholds['critical']:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def batch_predict(self, disaster_type: str, data_batch: list) -> list:
        """Predict for multiple data points"""
        results = []
        
        for data_point in data_batch:
            risk_score, inference_time, risk_level = self.predict_disaster(
                disaster_type, data_point
            )
            results.append({
                'risk_score': risk_score,
                'inference_time_ms': inference_time,
                'risk_level': risk_level
            })
        
        return results


if __name__ == "__main__":
    # Test the inference engine
    engine = UnifiedDisasterInferenceEngine(use_tflite=True)
    
    # Test flood prediction
    flood_data = {
        'rainfall_mm': 150.0,
        'humidity_percent': 85.0,
        'pressure_hpa': 1008.0,
        'river_level_m': 6.5,
        'soil_moisture': 0.8
    }
    
    risk, time_ms, level = engine.predict_disaster('flood', flood_data)
    print(f"\n🌊 Flood Prediction:")
    print(f"   Risk Score: {risk:.4f}")
    print(f"   Risk Level: {level}")
    print(f"   Inference Time: {time_ms:.2f} ms")
    
    # Test earthquake prediction
    earthquake_data = {
        'magnitude': 6.5,
        'depth_km': 25.0,
        'latitude': 35.0,
        'longitude': -120.0,
        'gap': 120.0,
        'dmin': 0.5,
        'rms': 0.8
    }
    
    risk, time_ms, level = engine.predict_disaster('earthquake', earthquake_data)
    print(f"\n🌋 Earthquake Prediction:")
    print(f"   Risk Score: {risk:.4f}")
    print(f"   Risk Level: {level}")
    print(f"   Inference Time: {time_ms:.2f} ms")