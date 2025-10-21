"""
Streamlit Dashboard for Multi-Disaster Monitoring
Real-time visualization and alert interface
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.disaster_inference import UnifiedDisasterInferenceEngine
from simulation.sensor_data_generator import MultiSensorSimulator
from src.alert_system import DisasterAlertSystem


# Page configuration
st.set_page_config(
    page_title="🌍 Multi-Disaster Edge AI Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-critical {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        animation: blink 1s linear infinite;
    }
    .alert-high {
        background-color: #ff8c00;
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = UnifiedDisasterInferenceEngine(use_tflite=True)
    st.session_state.sensor_simulator = MultiSensorSimulator(noise_factor=0.15)
    st.session_state.alert_system = DisasterAlertSystem()
    st.session_state.history = {
        'flood': {'time': [], 'risk': []},
        'earthquake': {'time': [], 'risk': []},
        'cyclone': {'time': [], 'risk': []}
    }
    st.session_state.max_history = 50


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level"""
    colors = {
        'SAFE': '#00FF00',
        'LOW': '#FFFF00',
        'MEDIUM': '#FFA500',
        'HIGH': '#FF4500',
        'CRITICAL': '#FF0000'
    }
    return colors.get(risk_level, '#CCCCCC')


def create_gauge_chart(risk_score: float, risk_level: str, title: str):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_risk_color(risk_level)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#E8F5E9'},
                {'range': [30, 60], 'color': '#FFF9C4'},
                {'range': [60, 80], 'color': '#FFECB3'},
                {'range': [80, 100], 'color': '#FFCDD2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_time_series_chart(history_data: dict, title: str):
    """Create time series chart for risk history"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(history_data['risk']))),
        y=history_data['risk'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Steps",
        yaxis_title="Risk Score",
        height=200,
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def main():
    # Header
    st.title("🌍 Multi-Disaster Edge AI Monitor")
    st.markdown("Real-time disaster prediction and alert system powered by Edge AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        use_tflite = st.checkbox("Use Edge Models (TFLite)", value=True)
        simulation_speed = st.slider("Simulation Speed (Hz)", 0.5, 5.0, 1.0, 0.5)
        
        st.markdown("---")
        st.header("📊 System Status")
        st.success("✅ All sensors operational")
        st.info(f"🔄 Update Rate: {simulation_speed} Hz")
        
        st.markdown("---")
        if st.button("🗑️ Clear All Alerts"):
            st.session_state.alert_system.clear_all_alerts()
            st.success("Alerts cleared!")
        
        if st.button("📊 View Alert Statistics"):
            stats = st.session_state.alert_system.get_alert_statistics()
            st.json(stats)
    
    # Main content
    placeholder = st.empty()
    
    # Start monitoring loop
    while True:
        with placeholder.container():
            # Generate sensor data
            sensor_data = st.session_state.sensor_simulator.generate_all_sensors()
            
            # Run predictions
            predictions = {}
            for disaster_type in ['flood', 'earthquake', 'cyclone']:
                risk_score, inference_time, risk_level = st.session_state.inference_engine.predict_disaster(
                    disaster_type, sensor_data[disaster_type]
                )
                predictions[disaster_type] = {
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'inference_time': inference_time
                }
                
                # Update history
                st.session_state.history[disaster_type]['time'].append(datetime.now())
                st.session_state.history[disaster_type]['risk'].append(risk_score)
                
                # Keep only recent history
                if len(st.session_state.history[disaster_type]['risk']) > st.session_state.max_history:
                    st.session_state.history[disaster_type]['time'].pop(0)
                    st.session_state.history[disaster_type]['risk'].pop(0)
                
                # Check for alerts
                alert = st.session_state.alert_system.create_alert(
                    disaster_type, risk_score, risk_level
                )
            
            # Display active alerts at top
            active_alerts = st.session_state.alert_system.get_active_alerts()
            if active_alerts:
                st.markdown("### 🚨 ACTIVE ALERTS")
                for disaster_type, alert in active_alerts.items():
                    if alert['risk_level'] == 'CRITICAL':
                        st.markdown(f'<div class="alert-critical">{alert["emoji"]} {alert["message"]}</div>', 
                                  unsafe_allow_html=True)
                    elif alert['risk_level'] == 'HIGH':
                        st.markdown(f'<div class="alert-high">{alert["emoji"]} {alert["message"]}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.warning(f"{alert['emoji']} {alert['message']}")
                st.markdown("---")
            
            # Create three columns for each disaster type
            col1, col2, col3 = st.columns(3)
            
            # Flood Monitor
            with col1:
                st.subheader("🌊 Flood Monitor")
                flood_pred = predictions['flood']
                
                # Gauge chart
                st.plotly_chart(
                    create_gauge_chart(flood_pred['risk_score'], flood_pred['risk_level'], "Flood Risk"),
                    use_container_width=True
                )
                
                # Metrics
                st.metric("Risk Level", flood_pred['risk_level'])
                st.metric("Inference Time", f"{flood_pred['inference_time']:.2f} ms")
                
                # Current sensor readings
                with st.expander("📡 Sensor Readings"):
                    flood_data = sensor_data['flood']
                    st.write(f"💧 Rainfall: {flood_data['rainfall_mm']:.1f} mm")
                    st.write(f"💨 Humidity: {flood_data['humidity_percent']:.1f}%")
                    st.write(f"🌡️ Pressure: {flood_data['pressure_hpa']:.1f} hPa")
                    st.write(f"🌊 River Level: {flood_data['river_level_m']:.1f} m")
                
                # Time series
                st.plotly_chart(
                    create_time_series_chart(st.session_state.history['flood'], "Flood Risk Trend"),
                    use_container_width=True
                )
            
            # Earthquake Monitor
            with col2:
                st.subheader("🌋 Earthquake Monitor")
                eq_pred = predictions['earthquake']
                
                # Gauge chart
                st.plotly_chart(
                    create_gauge_chart(eq_pred['risk_score'], eq_pred['risk_level'], "Earthquake Risk"),
                    use_container_width=True
                )
                
                # Metrics
                st.metric("Risk Level", eq_pred['risk_level'])
                st.metric("Inference Time", f"{eq_pred['inference_time']:.2f} ms")
                
                # Current sensor readings
                with st.expander("📡 Sensor Readings"):
                    eq_data = sensor_data['earthquake']
                    st.write(f"📊 Magnitude: {eq_data['magnitude']:.2f}")
                    st.write(f"📏 Depth: {eq_data['depth_km']:.1f} km")
                    st.write(f"🌍 Location: ({eq_data['latitude']:.2f}, {eq_data['longitude']:.2f})")
                    st.write(f"📉 RMS: {eq_data['rms']:.3f}")
                
                # Time series
                st.plotly_chart(
                    create_time_series_chart(st.session_state.history['earthquake'], "Earthquake Risk Trend"),
                    use_container_width=True
                )
            
            # Cyclone Monitor
            with col3:
                st.subheader("🌪️ Cyclone Monitor")
                cyclone_pred = predictions['cyclone']
                
                # Gauge chart
                st.plotly_chart(
                    create_gauge_chart(cyclone_pred['risk_score'], cyclone_pred['risk_level'], "Cyclone Risk"),
                    use_container_width=True
                )
                
                # Metrics
                st.metric("Risk Level", cyclone_pred['risk_level'])
                st.metric("Inference Time", f"{cyclone_pred['inference_time']:.2f} ms")
                
                # Current sensor readings
                with st.expander("📡 Sensor Readings"):
                    cyclone_data = sensor_data['cyclone']
                    st.write(f"💨 Wind Speed: {cyclone_data['wind_speed_kmh']:.1f} km/h")
                    st.write(f"🌡️ Pressure: {cyclone_data['sea_pressure_hpa']:.1f} hPa")
                    st.write(f"🌡️ Temperature: {cyclone_data['temperature_c']:.1f}°C")
                    st.write(f"🌊 Wave Height: {cyclone_data['wave_height_m']:.1f} m")
                
                # Time series
                st.plotly_chart(
                    create_time_series_chart(st.session_state.history['cyclone'], "Cyclone Risk Trend"),
                    use_container_width=True
                )
        
        # Update interval
        time.sleep(1.0 / simulation_speed)


if __name__ == "__main__":
    main()