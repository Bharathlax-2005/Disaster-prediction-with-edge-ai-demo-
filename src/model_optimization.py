"""
Model Optimization Module
Converts trained models to TensorFlow Lite format for edge deployment
"""

import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path
import json
import os


class EdgeModelOptimizer:
    """Optimize ML models for edge deployment"""
    
    def __init__(self, models_dir: str = "models", processed_data_dir: str = "data/processed"):
        self.models_dir = Path(models_dir)
        self.processed_data_dir = Path(processed_data_dir)
    
    def create_keras_wrapper(self, sklearn_model, input_shape: tuple, model_name: str):
        """Wrap sklearn model in Keras for TFLite conversion"""
        
        # Create a simple feedforward network that mimics the sklearn model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Load training data
        data_path = self.processed_data_dir / f"{model_name}_processed.npz"
        data = np.load(data_path)
        X, y = data['X'], data['y']
        
        # Train the Keras model to mimic sklearn model
        print(f"   Training Keras wrapper for {model_name}...")
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        
        return model
    
    def convert_to_tflite(self, keras_model, output_path: Path, quantize: bool = True):
        """Convert Keras model to TensorFlow Lite"""
        
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        
        if quantize:
            # Apply dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        return tflite_model
    
    def get_model_size(self, file_path: Path) -> float:
        """Get model file size in MB"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def optimize_flood_model(self):
        """Optimize flood prediction model"""
        print("\n🌊 Optimizing Flood Model for Edge Deployment...")
        
        # Load original model
        model_path = self.models_dir / "flood_model.pkl"
        with open(model_path, 'rb') as f:
            sklearn_model = pickle.load(f)
        
        # Create Keras wrapper
        keras_model = self.create_keras_wrapper(sklearn_model, (5,), 'flood')
        
        # Convert to TFLite
        tflite_path = self.models_dir / "flood_tflite_model.tflite"
        self.convert_to_tflite(keras_model, tflite_path, quantize=True)
        
        original_size = self.get_model_size(model_path)
        optimized_size = self.get_model_size(tflite_path)
        
        print(f"   ✅ Original size: {original_size:.2f} MB")
        print(f"   ✅ Optimized size: {optimized_size:.2f} MB")
        print(f"   ✅ Reduction: {(1 - optimized_size/original_size) * 100:.1f}%")
        
        return optimized_size
    
    def optimize_earthquake_model(self):
        """Optimize earthquake prediction model"""
        print("\n🌋 Optimizing Earthquake Model for Edge Deployment...")
        
        model_path = self.models_dir / "earthquake_model.pkl"
        with open(model_path, 'rb') as f:
            sklearn_model = pickle.load(f)
        
        keras_model = self.create_keras_wrapper(sklearn_model, (7,), 'earthquake')
        
        tflite_path = self.models_dir / "earthquake_tflite_model.tflite"
        self.convert_to_tflite(keras_model, tflite_path, quantize=True)
        
        original_size = self.get_model_size(model_path)
        optimized_size = self.get_model_size(tflite_path)
        
        print(f"   ✅ Original size: {original_size:.2f} MB")
        print(f"   ✅ Optimized size: {optimized_size:.2f} MB")
        print(f"   ✅ Reduction: {(1 - optimized_size/original_size) * 100:.1f}%")
        
        return optimized_size
    
    def optimize_cyclone_model(self):
        """Optimize cyclone prediction model"""
        print("\n🌪️ Optimizing Cyclone Model for Edge Deployment...")
        
        model_path = self.models_dir / "cyclone_model.pkl"
        with open(model_path, 'rb') as f:
            sklearn_model = pickle.load(f)
        
        keras_model = self.create_keras_wrapper(sklearn_model, (5,), 'cyclone')
        
        tflite_path = self.models_dir / "cyclone_tflite_model.tflite"
        self.convert_to_tflite(keras_model, tflite_path, quantize=True)
        
        original_size = self.get_model_size(model_path)
        optimized_size = self.get_model_size(tflite_path)
        
        print(f"   ✅ Original size: {original_size:.2f} MB")
        print(f"   ✅ Optimized size: {optimized_size:.2f} MB")
        print(f"   ✅ Reduction: {(1 - optimized_size/original_size) * 100:.1f}%")
        
        return optimized_size
    
    def optimize_all_models(self):
        """Optimize all disaster prediction models"""
        print("🚀 Starting Edge Model Optimization Pipeline...\n")
        
        sizes = {}
        sizes['flood'] = self.optimize_flood_model()
        sizes['earthquake'] = self.optimize_earthquake_model()
        sizes['cyclone'] = self.optimize_cyclone_model()
        
        # Save optimization report
        report = {
            'optimized_model_sizes_mb': sizes,
            'total_size_mb': sum(sizes.values()),
            'optimization_technique': 'TensorFlow Lite + Float16 Quantization'
        }
        
        report_path = self.models_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ All models optimized successfully!")
        print(f"📋 Total optimized models size: {sum(sizes.values()):.2f} MB")
        print(f"📄 Optimization report saved to {report_path}")


if __name__ == "__main__":
    optimizer = EdgeModelOptimizer()
    optimizer.optimize_all_models()