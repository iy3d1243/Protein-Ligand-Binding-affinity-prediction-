"""
Data utility functions.
"""

import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..data import DTADataset, collate_fn


def load_cached_data(config):
    """
    Load cached data from disk.
    
    Args:
        config: Configuration object
        
    Returns:
        dict: Cached data dictionary
    """
    cache_file = config.CACHE_DIR / config.PROCESSED_DATA_FILE
    
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No cached data found at {cache_file}! "
            "Run data preprocessing first."
        )
    
    print("=" * 80)
    print("LOADING CACHED DATA")
    print("=" * 80)
    
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    print("✓ Cached data loaded!")
    return cached_data


def create_dataloaders(cached_data, config):
    """
    Create data loaders from cached data.
    
    Args:
        cached_data (dict): Cached data dictionary
        config: Configuration object
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, datasets)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cached_data['tokenizer_name'], 
        trust_remote_code=True
    )
    
    # Create datasets from cache
    train_dataset = DTADataset(
        pd.DataFrame(), tokenizer, config.ESM_MAX_LENGTH,
        normalize_pki=True,
        pki_mean=cached_data['pki_mean'],
        pki_std=cached_data['pki_std'],
        cache_data=cached_data['train_cache']
    )
    
    val_dataset = DTADataset(
        pd.DataFrame(), tokenizer, config.ESM_MAX_LENGTH,
        normalize_pki=True,
        pki_mean=cached_data['pki_mean'],
        pki_std=cached_data['pki_std'],
        cache_data=cached_data['val_cache']
    )
    
    test_dataset = DTADataset(
        pd.DataFrame(), tokenizer, config.ESM_MAX_LENGTH,
        normalize_pki=True,
        pki_mean=cached_data['pki_mean'],
        pki_std=cached_data['pki_std'],
        cache_data=cached_data['test_cache']
    )
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)
