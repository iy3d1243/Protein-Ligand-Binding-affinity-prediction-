#!/usr/bin/env python3
"""
Main script for Drug-Target Affinity (DTA) prediction.

This script provides a complete pipeline for:
1. Data preprocessing and caching
2. Model training with multi-GPU support
3. Model evaluation and testing

Usage:
    python main.py --mode preprocess --data_path /path/to/data.csv
    python main.py --mode train
    python main.py --mode evaluate
"""

import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from configs.config import config
from src.data.preprocessing import preprocess_and_cache_data
from src.utils.data_utils import load_cached_data, create_dataloaders
from src.utils.model_utils import init_weights, count_parameters
from src.models import DTAModel
from src.training import Trainer


def preprocess_data(data_path, config):
    """
    Preprocess data and save to cache.
    
    Args:
        data_path (str): Path to CSV data file
        config: Configuration object
    """
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # Validate data path
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Preprocess and cache data
    cached_data = preprocess_and_cache_data(data_path, config)
    
    print("✓ Data preprocessing completed!")
    return cached_data


def train_model(config):
    """
    Train the DTA prediction model.
    
    Args:
        config: Configuration object
    """
    print("=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)
    
    # Load cached data
    cached_data = load_cached_data(config)
    
    # Print GPU information
    print("\n" + "=" * 80)
    print("GPU INFORMATION")
    print("=" * 80)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print("=" * 80 + "\n")
    
    # Create data loaders
    train_loader, val_loader, test_loader, datasets = create_dataloaders(cached_data, config)
    
    # Initialize model
    print("Initializing model...")
    model = DTAModel(config)
    
    # Initialize weights
    model.apply(init_weights)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Create trainer
    trainer = Trainer(
        model, train_loader, val_loader, config,
        pki_mean=cached_data['pki_mean'],
        pki_std=cached_data['pki_std']
    )
    
    # Train model
    trainer.train()
    
    # Final test evaluation
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION")
    print("=" * 80)
    test_loss, test_r2, test_rmse, test_mae = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print("=" * 80)
    
    return model, trainer


def evaluate_model(config):
    """
    Evaluate a trained model.
    
    Args:
        config: Configuration object
    """
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Load cached data
    cached_data = load_cached_data(config)
    
    # Create data loaders
    train_loader, val_loader, test_loader, datasets = create_dataloaders(cached_data, config)
    
    # Load trained model
    checkpoint_path = config.CHECKPOINTS_DIR / 'best_model_checkpoint.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Initialize model and load weights
    model = DTAModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Best validation loss: {checkpoint['val_loss']:.4f}")
    print(f"✓ Best validation R²: {checkpoint['val_r2']:.4f}")
    
    # Create trainer for evaluation
    trainer = Trainer(
        model, train_loader, val_loader, config,
        pki_mean=cached_data['pki_mean'],
        pki_std=cached_data['pki_std']
    )
    
    # Evaluate on all splits
    print("\nEvaluating on all data splits...")
    
    # Train set evaluation
    train_loss, train_r2, train_rmse, train_mae = trainer.evaluate(train_loader)
    print(f"\nTrain Set:")
    print(f"  Loss: {train_loss:.4f}")
    print(f"  R²: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    
    # Validation set evaluation
    val_loss, val_r2, val_rmse, val_mae = trainer.evaluate(val_loader)
    print(f"\nValidation Set:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  R²: {val_r2:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  MAE: {val_mae:.4f}")
    
    # Test set evaluation
    test_loss, test_r2, test_rmse, test_mae = trainer.evaluate(test_loader)
    print(f"\nTest Set:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  R²: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Drug-Target Affinity Prediction")
    parser.add_argument(
        "--mode", 
        choices=["preprocess", "train", "evaluate"], 
        required=True,
        help="Mode to run: preprocess, train, or evaluate"
    )
    parser.add_argument(
        "--data_path", 
        type=str,
        help="Path to CSV data file (required for preprocess mode)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to custom config file (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    config.validate_config()
    
    # Print configuration
    config.print_config()
    
    # Create necessary directories
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run selected mode
    if args.mode == "preprocess":
        if not args.data_path:
            raise ValueError("--data_path is required for preprocess mode")
        preprocess_data(args.data_path, config)
        
    elif args.mode == "train":
        train_model(config)
        
    elif args.mode == "evaluate":
        evaluate_model(config)


if __name__ == "__main__":
    main()
