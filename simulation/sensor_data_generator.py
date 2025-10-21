"""
Sensor Data Generator
Simulates real-time environmental sensor data streams
"""

import numpy as np
import time
from typing import Dict, Generator
from datetime import datetime


class MultiSensorSimulator:
    """Simulates multi-disaster sensor data streams"""
    
    def __init__(self, noise_factor: float = 0.1):
        self.noise_factor = noise_factor
        self.time_step = 0
        
    def generate_flood_sensor_data(self) -> Dict:
        """Generate realistic flood sensor readings"""
        # Base values with temporal patterns
        hour = (self.time_step % 24)
        
        # Rainfall varies by time (higher in evening)
        rainfall_base = 50 + 30 * np.sin(hour * np.pi / 12)
        rainfall = max(0, rainfall_base + np.random.normal(0, 20 * self.noise_factor))
        
        # Humidity correlates with rainfall
        humidity = min(100, 60 + 20 * (rainfall / 100) + np.random.normal(0, 10 * self.noise_factor))
        
        # Pressure inversely correlates with rainfall
        pressure = 1013 - (rainfall / 10) + np.random.normal(0, 5 * self.noise_factor)
        
        # River level increases with rainfall (with lag)
        river_level = 2 + (rainfall / 30) + np.random.normal(0, 0.5 * self.noise_factor)
        
        # Soil moisture
        soil_moisture = min(1.0, 0.3 + (rainfall / 200) + np.random.normal(0, 0.1 * self.noise_factor))
        
        return {
            'rainfall_mm': float(rainfall),
            'humidity_percent': float(humidity),
            'pressure_hpa': float(pressure),
            'river_level_m': float(river_level),
            'soil_moisture': float(soil_moisture),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_earthquake_sensor_data(self) -> Dict:
        """Generate realistic earthquake sensor readings"""
        # Most readings are normal, occasional spikes
        is_event = np.random.random() < 0.05  # 5% chance of seismic event
        
        if is_event:
            magnitude = np.random.uniform(4.0, 7.5)
            depth = np.random.exponential(40)
        else:
            magnitude = np.random.uniform(2.0, 3.5)
            depth = np.random.uniform(5, 100)
        
        # Location (simulate specific region)
        latitude = 35.0 + np.random.normal(0, 2)
        longitude = -120.0 + np.random.normal(0, 2)
        
        # Seismic parameters
        gap = np.random.uniform(30, 250)
        dmin = np.random.exponential(1)
        rms = np.random.exponential(0.5)
        
        return {
            'magnitude': float(magnitude),
            'depth_km': float(depth),
            'latitude': float(latitude),
            'longitude': float(longitude),
            'gap': float(gap),
            'dmin': float(dmin),
            'rms': float(rms),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_cyclone_sensor_data(self) -> Dict:
        """Generate realistic cyclone sensor readings"""
        # Cyclone intensity varies
        is_cyclone_active = np.random.random() < 0.3  # 30% chance of active cyclone
        
        if is_cyclone_active:
            wind_speed = np.random.gamma(3, 40)  # Higher wind speeds
            sea_pressure = np.random.normal(985, 15)  # Lower pressure
        else:
            wind_speed = np.random.gamma(2, 15)  # Normal wind
            sea_pressure = np.random.normal(1010, 5)
        
        # Temperature and humidity
        temperature = 28 + np.random.normal(0, 3 * self.noise_factor)
        humidity = min(100, 70 + 15 * (wind_speed / 150) + np.random.normal(0, 10 * self.noise_factor))
        
        # Wave height correlates with wind speed
        wave_height = (wind_speed / 50) + np.random.exponential(1)
        
        return {
            'wind_speed_kmh': float(wind_speed),
            'sea_pressure_hpa': float(sea_pressure),
            'temperature_c': float(temperature),
            'humidity_percent': float(humidity),
            'wave_height_m': float(wave_height),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_all_sensors(self) -> Dict[str, Dict]:
        """Generate data from all sensor types"""
        self.time_step += 1
        
        return {
            'flood': self.generate_flood_sensor_data(),
            'earthquake': self.generate_earthquake_sensor_data(),
            'cyclone': self.generate_cyclone_sensor_data()
        }
    
    def stream_data(self, interval_seconds: float = 1.0) -> Generator[Dict, None, None]:
        """Stream sensor data continuously"""
        while True:
            data = self.generate_all_sensors()
            yield data
            time.sleep(interval_seconds)


if __name__ == "__main__":
    # Test the sensor simulator
    simulator = MultiSensorSimulator(noise_factor=0.15)
    
    print("🔄 Starting Sensor Data Stream...\n")
    
    for i, sensor_data in enumerate(simulator.stream_data(interval_seconds=2.0)):
        print(f"📊 Reading #{i+1}:")
        print(f"   🌊 Flood Sensors: Rainfall={sensor_data['flood']['rainfall_mm']:.1f}mm, "
              f"River={sensor_data['flood']['river_level_m']:.1f}m")
        print(f"   🌋 Earthquake Sensors: Magnitude={sensor_data['earthquake']['magnitude']:.2f}, "
              f"Depth={sensor_data['earthquake']['depth_km']:.1f}km")
        print(f"   🌪️ Cyclone Sensors: Wind={sensor_data['cyclone']['wind_speed_kmh']:.1f}km/h, "
              f"Pressure={sensor_data['cyclone']['sea_pressure_hpa']:.1f}hPa")
        print()
        
        if i >= 5:  # Stop after 5 readings for demo
            break