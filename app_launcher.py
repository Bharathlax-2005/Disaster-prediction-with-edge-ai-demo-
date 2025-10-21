"""
Master Application Launcher
Orchestrates the complete disaster prediction pipeline
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(command, description):
    """Run a command and display progress"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(command, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error during: {description}")
        sys.exit(1)
    
    print(f"\n✅ Completed: {description}")


def setup_environment():
    """Install required packages"""
    run_command(
        "pip install --quiet --no-input -r requirements.txt",
        "Installing Python dependencies"
    )


def preprocess_data():
    """Run data preprocessing"""
    run_command(
        "python src/data_preprocessing.py",
        "Preprocessing disaster datasets"
    )


def train_models():
    """Train all disaster prediction models"""
    run_command(
        "python src/model_training.py",
        "Training disaster prediction models"
    )


def optimize_models():
    """Optimize models for edge deployment"""
    run_command(
        "python src/model_optimization.py",
        "Optimizing models for Edge AI"
    )


def launch_dashboard():
    """Launch Streamlit dashboard"""
    print(f"\n{'='*60}")
    print("🌍 Launching Multi-Disaster Edge AI Dashboard")
    print(f"{'='*60}\n")
    print("📱 Dashboard will open in your browser")
    print("🔴 Press Ctrl+C to stop the dashboard\n")
    
    subprocess.run("streamlit run src/gui_dashboard.py", shell=True)


def test_inference():
    """Test inference engine"""
    run_command(
        "python src/disaster_inference.py",
        "Testing unified inference engine"
    )


def test_sensors():
    """Test sensor simulation"""
    run_command(
        "python simulation/sensor_data_generator.py",
        "Testing sensor data generation"
    )


def full_pipeline():
    """Run complete pipeline"""
    print("\n" + "="*60)
    print("🌍 MULTI-DISASTER EDGE AI SYSTEM")
    print("="*60)
    print("\nStarting complete pipeline...")
    
    setup_environment()
    preprocess_data()
    train_models()
    optimize_models()
    test_inference()
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print("\n📊 Summary:")
    print("   ✓ Data preprocessed and ready")
    print("   ✓ All models trained successfully")
    print("   ✓ Models optimized for edge deployment")
    print("   ✓ Inference engine tested and operational")
    print("\n🚀 Ready to launch dashboard!")
    print("   Run: python app_launcher.py --dashboard")


def main():
    parser = argparse.ArgumentParser(
        description='🌍 Multi-Disaster Edge AI System Launcher'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup environment and install dependencies'
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Preprocess disaster datasets'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train all disaster prediction models'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimize models for edge deployment'
    )
    
    parser.add_argument(
        '--test-inference',
        action='store_true',
        help='Test inference engine'
    )
    
    parser.add_argument(
        '--test-sensors',
        action='store_true',
        help='Test sensor data generation'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch Streamlit dashboard'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run complete pipeline (setup, preprocess, train, optimize)'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python app_launcher.py --full        # Run complete pipeline")
        print("   python app_launcher.py --dashboard   # Launch dashboard")
        return
    
    # Execute requested operations
    if args.setup:
        setup_environment()
    
    if args.preprocess:
        preprocess_data()
    
    if args.train:
        train_models()
    
    if args.optimize:
        optimize_models()
    
    if args.test_inference:
        test_inference()
    
    if args.test_sensors:
        test_sensors()
    
    if args.full:
        full_pipeline()
    
    if args.dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
    