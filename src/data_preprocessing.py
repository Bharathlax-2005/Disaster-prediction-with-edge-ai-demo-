"""
Data Preprocessing Module
Handles loading, cleaning, and preprocessing of multi-disaster datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
from sklearn.preprocessing import StandardScaler
import pickle


class MultiDisasterPreprocessor:
    """Unified preprocessor for flood, earthquake, and cyclone datasets"""
    
    def __init__(self, data_dir: str = "data", config_path: str = "configs/model_mapping.json"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.scalers = {}
    
    def generate_synthetic_flood_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic flood prediction dataset"""
        np.random.seed(42)
        
        data = {
            'rainfall_mm': np.random.gamma(2, 50, n_samples),  # Skewed distribution
            'humidity_percent': np.random.normal(70, 15, n_samples).clip(30, 100),
            'pressure_hpa': np.random.normal(1013, 10, n_samples),
            'river_level_m': np.random.exponential(3, n_samples),
            'soil_moisture': np.random.beta(2, 5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target: flood risk (0-1)
        df['flood_risk'] = (
            0.4 * (df['rainfall_mm'] / 200) +
            0.2 * (df['humidity_percent'] / 100) +
            0.3 * (df['river_level_m'] / 10) +
            0.1 * df['soil_moisture'] +
            np.random.normal(0, 0.05, n_samples)
        ).clip(0, 1)
        
        # Binary classification
        df['flood_occurred'] = (df['flood_risk'] > 0.6).astype(int)
        
        return df
    
    def generate_synthetic_earthquake_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic earthquake dataset"""
        np.random.seed(43)
        
        data = {
            'magnitude': np.random.gamma(2, 1.5, n_samples).clip(2, 9),
            'depth_km': np.random.exponential(50, n_samples).clip(0, 700),
            'latitude': np.random.uniform(-90, 90, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples),
            'gap': np.random.uniform(20, 300, n_samples),
            'dmin': np.random.exponential(1, n_samples),
            'rms': np.random.exponential(0.5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target: earthquake severity risk
        df['earthquake_risk'] = (
            0.5 * (df['magnitude'] / 9) +
            0.2 * (1 - df['depth_km'] / 700) +  # Shallow earthquakes more dangerous
            0.15 * (df['rms'] / 2) +
            0.15 * np.random.random(n_samples)
        ).clip(0, 1)
        
        df['severe_earthquake'] = (df['earthquake_risk'] > 0.65).astype(int)
        
        return df
    
    def generate_synthetic_cyclone_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic cyclone dataset"""
        np.random.seed(44)
        
        data = {
            'wind_speed_kmh': np.random.gamma(3, 30, n_samples).clip(20, 300),
            'sea_pressure_hpa': np.random.normal(1000, 15, n_samples).clip(950, 1020),
            'temperature_c': np.random.normal(28, 3, n_samples),
            'humidity_percent': np.random.normal(80, 10, n_samples).clip(50, 100),
            'wave_height_m': np.random.exponential(2, n_samples).clip(0, 15)
        }
        
        df = pd.DataFrame(data)
        
        # Create target: cyclone intensity risk
        df['cyclone_risk'] = (
            0.4 * (df['wind_speed_kmh'] / 300) +
            0.25 * (1 - (df['sea_pressure_hpa'] - 950) / 70) +  # Low pressure = high risk
            0.2 * (df['wave_height_m'] / 15) +
            0.15 * (df['humidity_percent'] / 100) +
            np.random.normal(0, 0.05, n_samples)
        ).clip(0, 1)
        
        df['cyclone_severe'] = (df['cyclone_risk'] > 0.6).astype(int)
        
        return df
    
    def preprocess_flood_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess flood data"""
        features = self.config['flood']['features']
        X = df[features].values
        y = df['flood_occurred'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['flood'] = scaler
        
        return X_scaled, y
    
    def preprocess_earthquake_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess earthquake data"""
        features = self.config['earthquake']['features']
        X = df[features].values
        y = df['severe_earthquake'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['earthquake'] = scaler
        
        return X_scaled, y
    
    def preprocess_cyclone_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess cyclone data"""
        features = self.config['cyclone']['features']
        X = df[features].values
        y = df['cyclone_severe'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['cyclone'] = scaler
        
        return X_scaled, y
    
    def save_processed_data(self, disaster_type: str, X: np.ndarray, y: np.ndarray):
        """Save processed data and scaler"""
        save_path = self.processed_dir / f"{disaster_type}_processed.npz"
        np.savez(save_path, X=X, y=y)
        
        scaler_path = self.processed_dir / f"{disaster_type}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers[disaster_type], f)
        
        print(f"✅ Saved processed {disaster_type} data to {save_path}")
    
    def load_processed_data(self, disaster_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load processed data"""
        load_path = self.processed_dir / f"{disaster_type}_processed.npz"
        data = np.load(load_path)
        return data['X'], data['y']
    
    def process_all_datasets(self):
        """Process all three disaster datasets"""
        print("🔄 Generating synthetic datasets...")
        
        # Generate data
        flood_df = self.generate_synthetic_flood_data()
        earthquake_df = self.generate_synthetic_earthquake_data()
        cyclone_df = self.generate_synthetic_cyclone_data()
        
        # Save raw data
        flood_df.to_csv(self.data_dir / "flood" / "flood_data.csv", index=False)
        earthquake_df.to_csv(self.data_dir / "earthquake" / "earthquake_data.csv", index=False)
        cyclone_df.to_csv(self.data_dir / "cyclone" / "cyclone_data.csv", index=False)
        
        print("✅ Raw datasets saved")
        
        # Preprocess
        print("\n🔄 Preprocessing datasets...")
        
        X_flood, y_flood = self.preprocess_flood_data(flood_df)
        self.save_processed_data('flood', X_flood, y_flood)
        
        X_earthquake, y_earthquake = self.preprocess_earthquake_data(earthquake_df)
        self.save_processed_data('earthquake', X_earthquake, y_earthquake)
        
        X_cyclone, y_cyclone = self.preprocess_cyclone_data(cyclone_df)
        self.save_processed_data('cyclone', X_cyclone, y_cyclone)
        
        print("\n✅ All datasets preprocessed successfully!")
        print(f"   - Flood: {len(y_flood)} samples")
        print(f"   - Earthquake: {len(y_earthquake)} samples")
        print(f"   - Cyclone: {len(y_cyclone)} samples")


if __name__ == "__main__":
    preprocessor = MultiDisasterPreprocessor()
    preprocessor.process_all_datasets()