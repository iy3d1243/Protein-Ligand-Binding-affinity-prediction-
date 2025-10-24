"""
Data preprocessing utilities for Drug-Target Affinity prediction.
"""

import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from ..data.dataset import DTADataset, smiles_to_graph


def preprocess_and_cache_data(csv_path, config):
    """
    Preprocess data and cache it to disk for faster subsequent loading.
    
    Args:
        csv_path (str): Path to the CSV file containing the dataset
        config: Configuration object with data processing settings
        
    Returns:
        dict: Cached data dictionary containing processed datasets
    """
    # Create cache directory
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = config.CACHE_DIR / config.PROCESSED_DATA_FILE
    
    # Check if cached data already exists
    if cache_file.exists():
        print("=" * 80)
        print("CACHED DATA FOUND - Loading from disk...")
        print("=" * 80)
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print("✓ Loaded cached data successfully!")
        return cached_data
    
    print("=" * 80)
    print("PREPROCESSING DATA - This will be saved to disk")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Load and clean data
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    # Clean data - remove invalid pKi values
    original_len = len(df)
    df = df[~df['pKi'].isna() & ~np.isinf(df['pKi'])]
    if len(df) < original_len:
        print(f"⚠ Removed {original_len - len(df)} invalid pKi rows")
    
    # Split data into train/validation/test sets
    train_val_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_SEED
    )
    
    print(f"Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}\n")
    
    # Load ESM-2 tokenizer
    print("Loading ESM-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.ESM_MODEL, trust_remote_code=True)
    
    # Process datasets
    print("\nProcessing TRAIN dataset...")
    train_dataset = DTADataset(train_df, tokenizer, config.ESM_MAX_LENGTH)
    
    print("\nProcessing VAL dataset...")
    val_dataset = DTADataset(
        val_df, tokenizer, config.ESM_MAX_LENGTH,
        pki_mean=train_dataset.pki_mean,
        pki_std=train_dataset.pki_std
    )
    
    print("\nProcessing TEST dataset...")
    test_dataset = DTADataset(
        test_df, tokenizer, config.ESM_MAX_LENGTH,
        pki_mean=train_dataset.pki_mean,
        pki_std=train_dataset.pki_std
    )
    
    # Save processed data to disk
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA TO DISK...")
    cached_data = {
        'train_cache': train_dataset.get_cache_data(),
        'val_cache': val_dataset.get_cache_data(),
        'test_cache': test_dataset.get_cache_data(),
        'pki_mean': train_dataset.pki_mean,
        'pki_std': train_dataset.pki_std,
        'tokenizer_name': config.ESM_MODEL
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    
    print(f"✓ Data saved to: {cache_file}")
    print("✓ You can now train without reprocessing, even after crashes!")
    print("=" * 80)
    
    return cached_data
