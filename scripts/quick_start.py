#!/usr/bin/env python3
"""
Quick start script for Drug-Target Affinity prediction.

This script provides a simplified interface for common tasks.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from configs.config import config
from main import preprocess_data, train_model, evaluate_model


def quick_start(data_path):
    """
    Run the complete pipeline: preprocess -> train -> evaluate
    
    Args:
        data_path (str): Path to CSV data file
    """
    print("🚀 Starting DTA Prediction Pipeline")
    print("=" * 50)
    
    # Step 1: Preprocess data
    print("\n📊 Step 1: Data Preprocessing")
    preprocess_data(data_path, config)
    
    # Step 2: Train model
    print("\n🧠 Step 2: Model Training")
    train_model(config)
    
    # Step 3: Evaluate model
    print("\n📈 Step 3: Model Evaluation")
    evaluate_model(config)
    
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/quick_start.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    quick_start(data_path)
