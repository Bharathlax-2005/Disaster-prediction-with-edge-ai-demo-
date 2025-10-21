# 🌍 Edge AI-Powered Multi-Disaster Prediction & Alert Simulation System

A unified Edge-AI simulation platform that predicts multiple disasters (Floods, Earthquakes, Cyclones) using lightweight ML models optimized for Edge inference, with real-time sensor simulation and risk alert dashboards.

## 🎯 Project Overview

This system demonstrates:
- **Multi-Disaster Prediction**: Simultaneous monitoring of floods, earthquakes, and cyclones
- **Edge AI Optimization**: TensorFlow Lite models with quantization for resource-constrained devices
- **Real-Time Simulation**: Synthetic sensor data streams mimicking real environmental conditions
- **Interactive Dashboard**: Streamlit-based GUI with color-coded risk indicators
- **Alert System**: Automated risk alerts with visual, audio, and logging capabilities

## 📁 Project Structure

```
multi_disaster_edge_ai/
│
├── data/                          # Dataset storage
│   ├── flood/                     # Flood prediction data
│   ├── earthquake/                # Earthquake data
│   ├── cyclone/                   # Cyclone data
│   └── processed/                 # Cleaned and normalized data
│
├── models/                        # Trained models
│   ├── *_model.pkl               # Original models
│   ├── *_tflite_model.tflite     # Edge-optimized models
│   └── evaluation_results.json    # Performance metrics
│
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data cleaning and feature engineering
│   ├── model_training.py          # Model training pipeline
│   ├── model_optimization.py      # TFLite conversion and quantization
│   ├── disaster_inference.py      # Unified inference engine
│   ├── alert_system.py            # Alert management
│   └── gui_dashboard.py           # Streamlit dashboard
│
├── simulation/                    # Simulation modules
│   └── sensor_data_generator.py   # Synthetic sensor data
│
├── configs/                       # Configuration files
│   ├── thresholds.json            # Risk level definitions
│   ├── model_mapping.json         # Model-disaster mappings
│   └── ui_config.yaml             # Dashboard settings
│
├── reports/                       # Output reports and logs
│   ├── alert_log.csv              # Alert history
│   └── visualization_plots/       # Generated charts
│
├── app_launcher.py                # Master launcher script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Features

### 1. Multi-Disaster Coverage
- **Flood Prediction**: Rainfall, humidity, pressure, river levels, soil moisture
- **Earthquake Detection**: Magnitude, depth, location, seismic parameters
- **Cyclone Monitoring**: Wind speed, sea pressure, temperature, wave height

### 2. Edge AI Optimization
- TensorFlow Lite conversion with float16 quantization
- Model size reduction: 70-90% smaller than original
- Inference latency: <10ms per prediction
- Suitable for Raspberry Pi, Jetson Nano, mobile devices

### 3. Real-Time Simulation
- Synthetic sensor data generation with realistic temporal patterns
- Configurable noise and variability
- Multi-threaded parallel streams for all disaster types
- Adjustable simulation speed (0.5-5 Hz)

### 4. Interactive Dashboard
- Real-time risk gauges for each disaster type
- Time-series trend visualization
- Color-coded alerts (Green → Yellow → Orange → Red)
- Live sensor readings display
- Alert acknowledgment system

### 5. Alert System
- 5-level risk classification: SAFE, LOW, MEDIUM, HIGH, CRITICAL
- Automated alert triggering based on thresholds
- CSV logging for historical analysis
- Alert statistics and analytics

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- Operating System: Windows, macOS, or Linux

## ⚙️ Installation & Setup

### Step 1: Clone or Download the Project
```bash
# If you have git
git clone <repository-url>
cd name

# Or extract the ZIP file and navigate to it
cd name
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Using the launcher (recommended)
python app_launcher.py --setup

# Or manually
pip install -r requirements.txt
```

## 🔧 Usage Guide

### Quick Start (Complete Pipeline)

Run the entire pipeline in one command:
```bash
python app_launcher.py --full
```

This will:
1. ✅ Install dependencies
2. ✅ Generate and preprocess datasets (30,000 samples total)
3. ✅ Train all three disaster prediction models
4. ✅ Optimize models for edge deployment
5. ✅ Test inference engine

### Launch Dashboard

After completing the pipeline:
```bash
python app_launcher.py --dashboard
```

The dashboard will open in your browser at `http://localhost:8501`

### Individual Operations

**Preprocess Data Only:**
```bash
python app_launcher.py --preprocess
```

**Train Models Only:**
```bash
python app_launcher.py --train
```

**Optimize Models Only:**
```bash
python app_launcher.py --optimize
```

**Test Inference Engine:**
```bash
python app_launcher.py --test-inference
```

**Test Sensor Simulation:**
```bash
python app_launcher.py --test-sensors
```

## 📊 Dashboard Features

### Main View
The dashboard displays three real-time monitoring panels:

1. **🌊 Flood Monitor**
   - Risk gauge (0-100%)
   - Current sensor readings
   - Historical trend chart
   - Alert status

2. **🌋 Earthquake Monitor**
   - Seismic risk gauge
   - Magnitude and depth readings
   - Location tracking
   - Risk trend visualization

3. **🌪️ Cyclone Monitor**
   - Storm intensity gauge
   - Wind speed and pressure
   - Wave height monitoring
   - Cyclone risk trends

### Alert System
- **🟢 SAFE**: No action required
- **🟡 LOW**: Monitor conditions
- **🟠 MEDIUM**: Prepare emergency supplies
- **🔴 HIGH**: Take immediate protective action
- **🚨 CRITICAL**: Emergency evacuation required

### Sidebar Controls
- Toggle Edge AI models (TFLite) vs. Original models
- Adjust simulation speed (0.5-5 Hz)
- Clear all alerts
- View alert statistics

## 🧪 Testing & Validation

### Model Performance Metrics

Expected performance after training:

| Disaster Type | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| Flood        | 85-90%   | 82-88%    | 83-89% | 83-88%   |
| Earthquake   | 87-92%   | 85-90%    | 86-91% | 86-90%   |
| Cyclone      | 84-89%   | 81-87%    | 82-88% | 82-87%   |

### Edge Optimization Results

| Model        | Original Size | TFLite Size | Reduction | Inference Time |
|--------------|--------------|-------------|-----------|----------------|
| Flood        | ~2.5 MB      | ~0.3 MB     | 88%       | 3-5 ms         |
| Earthquake   | ~3.2 MB      | ~0.4 MB     | 87%       | 4-6 ms         |
| Cyclone      | ~2.8 MB      | ~0.35 MB    | 87%       | 3-5 ms         |

## 🔬 How It Works

### 1. Data Generation
Synthetic datasets are generated with realistic distributions:
- **Flood**: Gamma distribution for rainfall, exponential for river levels
- **Earthquake**: Exponential depth distribution, realistic magnitude ranges
- **Cyclone**: Gamma distribution for wind speed, inverse pressure correlation

### 2. Feature Engineering
- Standardization using StandardScaler
- Feature correlation analysis
- Target variable generation based on multi-factor risk models

### 3. Model Training
- **Flood**: Random Forest (100 trees, max_depth=15)
- **Earthquake**: XGBoost (100 estimators, learning_rate=0.1)
- **Cyclone**: Gradient Boosting (100 estimators, subsample=0.8)

### 4. Edge Optimization
- Keras wrapper creation for sklearn models
- TensorFlow Lite conversion
- Float16 quantization for size reduction
- Validation of accuracy retention

### 5. Real-Time Inference
- Multi-threaded sensor data generation
- Parallel inference across all disaster types
- Risk level classification using configurable thresholds
- Alert triggering based on risk levels

## 📈 Customization

### Adjust Risk Thresholds
Edit `configs/thresholds.json`:
```json
{
  "flood": {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8,
    "critical": 0.95
  }
}
```

### Modify Sensor Parameters
Edit `simulation/sensor_data_generator.py` to adjust:
- Sensor reading ranges
- Noise factors
- Temporal patterns
- Event probabilities

### Change Dashboard Appearance
Edit `configs/ui_config.yaml`:
```yaml
risk_colors:
  safe: "#00FF00"
  critical: "#FF0000"
  
dashboard:
  refresh_interval: 2
  max_history_points: 100
```

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Dashboard won't start
**Solution**: Make sure Streamlit is installed and no other process uses port 8501
```bash
pip install streamlit
streamlit run src/gui_dashboard.py --server.port 8502
```

### Issue: Models not found
**Solution**: Run the complete pipeline first
```bash
python app_launcher.py --full
```

### Issue: TensorFlow errors
**Solution**: Install compatible TensorFlow version
```bash
pip install tensorflow==2.15.0
```

## 📝 File Descriptions

### Core Modules

**data_preprocessing.py**
- Generates synthetic disaster datasets
- Performs feature scaling and normalization
- Saves processed data and scalers

**model_training.py**
- Trains individual models for each disaster type
- Evaluates model performance
- Saves trained models and metrics

**model_optimization.py**
- Converts models to TensorFlow Lite
- Applies quantization for size reduction
- Validates optimized model accuracy

**disaster_inference.py**
- Unified inference engine for all disasters
- Handles model loading and routing
- Provides prediction API

**gui_dashboard.py**
- Streamlit-based web interface
- Real-time visualization and monitoring
- Interactive alert management

**alert_system.py**
- Alert generation and management
- CSV logging for audit trails
- Statistics and analytics

**sensor_data_generator.py**
- Realistic sensor data simulation
- Temporal pattern generation
- Multi-disaster data streams

## 🎓 Educational Value

This project demonstrates:
- End-to-end ML pipeline development
- Edge AI model optimization techniques
- Real-time data simulation
- Multi-model system integration
- Interactive dashboard development
- Alert and monitoring systems

## 🔮 Future Enhancements

Potential additions:
- [ ] Integration with real sensor hardware (Arduino, Raspberry Pi)
- [ ] Deep learning models (LSTM, CNN) for time-series prediction
- [ ] Geographic information system (GIS) integration
- [ ] Mobile app for alerts
- [ ] Cloud deployment with Docker
- [ ] Historical data analysis and trend prediction
- [ ] Multi-language support
- [ ] Email/SMS alert notifications

## 📄 License

This project is for educational purposes. Feel free to modify and extend it.

**Built with ❤️ for disaster resilience and Edge AI education**
