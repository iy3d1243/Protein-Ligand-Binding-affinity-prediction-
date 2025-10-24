"""
Configuration module for Drug-Target Affinity (DTA) prediction model.
Optimized for 2x T4 GPUs with data persistence and memory efficiency.
"""

import torch
from pathlib import Path


class Config:
    """Centralized configuration for DTA prediction model"""
    
    # ============================================================================
    # MODEL ARCHITECTURE - OPTIMIZED FOR MEMORY EFFICIENCY
    # ============================================================================
    
    # Embedding dimensions
    LIGAND_EMBED_DIM = 192  # Reduced from 256 for memory efficiency
    PROTEIN_EMBED_DIM = 192  # Reduced from 256 for memory efficiency
    
    # GNN (Graph Neural Network) settings
    GNN_HIDDEN_DIM = 96      # Reduced from 128
    GNN_NUM_LAYERS = 2       # Reduced from 3
    GNN_INPUT_DIM = 5        # Atom features: atomic_num, degree, charge, hybridization, aromatic
    
    # Cross-attention settings
    ATTENTION_HEADS = 6      # Reduced from 8
    ATTENTION_DROPOUT = 0.15
    
    # MLP (Multi-Layer Perceptron) settings
    MLP_DIMS = [576, 384, 192, 64, 1]  # 3*192 (ligand + protein + cross-attention)
    MLP_DROPOUT = 0.2
    
    # ============================================================================
    # ESM-2 PROTEIN ENCODING SETTINGS
    # ============================================================================
    
    ESM_MODEL = "facebook/esm2_t12_35M_UR50D"  # Smaller model for memory efficiency
    ESM_MAX_LENGTH = 800     # Reduced from 1024
    FREEZE_ESM = True        # Freeze ESM weights to save memory (can unfreeze later)
    
    # ============================================================================
    # TRAINING SETTINGS - OPTIMIZED FOR 2x T4 GPUs
    # ============================================================================
    
    # Batch size and gradient accumulation
    BATCH_SIZE = 4           # Per GPU (effective = 4 * 2 = 8)
    GRADIENT_ACCUMULATION = 2  # Effective batch = 8 * 2 = 16
    
    # Optimization
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 1.0
    
    # Training schedule
    NUM_EPOCHS = 100
    PATIENCE = 15             # Early stopping patience
    
    # ============================================================================
    # DATA SETTINGS
    # ============================================================================
    
    # Data splits
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_SEED = 42
    
    # DataLoader settings
    NUM_WORKERS = 2           # Reduced for stability
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False
    
    # ============================================================================
    # PATHS AND PERSISTENCE
    # ============================================================================
    
    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    
    # Data paths
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CACHE_DIR = DATA_DIR / "cache"
    
    # Output paths
    CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
    LOGS_DIR = OUTPUTS_DIR / "logs"
    RESULTS_DIR = OUTPUTS_DIR / "results"
    
    # Cache settings
    PROCESSED_DATA_FILE = "processed_datasets.pkl"
    
    # ============================================================================
    # DEVICE AND GPU SETTINGS
    # ============================================================================
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_MULTI_GPU = torch.cuda.device_count() > 1
    
    # ============================================================================
    # LOGGING AND MONITORING
    # ============================================================================
    
    LOG_LEVEL = "INFO"
    SAVE_ATTENTION_WEIGHTS = False  # Set to True to save attention visualizations
    
    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================
    
    @classmethod
    def validate_config(cls):
        """Validate configuration parameters"""
        assert cls.LIGAND_EMBED_DIM > 0, "LIGAND_EMBED_DIM must be positive"
        assert cls.PROTEIN_EMBED_DIM > 0, "PROTEIN_EMBED_DIM must be positive"
        assert cls.GNN_HIDDEN_DIM > 0, "GNN_HIDDEN_DIM must be positive"
        assert cls.GNN_NUM_LAYERS > 0, "GNN_NUM_LAYERS must be positive"
        assert cls.ATTENTION_HEADS > 0, "ATTENTION_HEADS must be positive"
        assert 0 <= cls.ATTENTION_DROPOUT <= 1, "ATTENTION_DROPOUT must be between 0 and 1"
        assert 0 <= cls.MLP_DROPOUT <= 1, "MLP_DROPOUT must be between 0 and 1"
        assert cls.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
        assert cls.GRADIENT_ACCUMULATION > 0, "GRADIENT_ACCUMULATION must be positive"
        assert cls.LEARNING_RATE > 0, "LEARNING_RATE must be positive"
        assert cls.WEIGHT_DECAY >= 0, "WEIGHT_DECAY must be non-negative"
        assert cls.NUM_EPOCHS > 0, "NUM_EPOCHS must be positive"
        assert cls.PATIENCE > 0, "PATIENCE must be positive"
        assert 0 < cls.TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"
        assert 0 < cls.VAL_SIZE < 1, "VAL_SIZE must be between 0 and 1"
        assert cls.VAL_SIZE + cls.TEST_SIZE < 1, "VAL_SIZE + TEST_SIZE must be less than 1"
        
        print("✓ Configuration validation passed!")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 80)
        print("DTA PREDICTION MODEL CONFIGURATION")
        print("=" * 80)
        print(f"Model Architecture:")
        print(f"  - Ligand Embedding Dim: {cls.LIGAND_EMBED_DIM}")
        print(f"  - Protein Embedding Dim: {cls.PROTEIN_EMBED_DIM}")
        print(f"  - GNN Hidden Dim: {cls.GNN_HIDDEN_DIM}")
        print(f"  - GNN Layers: {cls.GNN_NUM_LAYERS}")
        print(f"  - Attention Heads: {cls.ATTENTION_HEADS}")
        print(f"  - MLP Dimensions: {cls.MLP_DIMS}")
        print()
        print(f"Training Settings:")
        print(f"  - Batch Size: {cls.BATCH_SIZE} (per GPU)")
        print(f"  - Gradient Accumulation: {cls.GRADIENT_ACCUMULATION}")
        print(f"  - Effective Batch Size: {cls.BATCH_SIZE * (2 if cls.USE_MULTI_GPU else 1) * cls.GRADIENT_ACCUMULATION}")
        print(f"  - Learning Rate: {cls.LEARNING_RATE}")
        print(f"  - Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"  - Max Epochs: {cls.NUM_EPOCHS}")
        print(f"  - Patience: {cls.PATIENCE}")
        print()
        print(f"Data Settings:")
        print(f"  - Test Size: {cls.TEST_SIZE}")
        print(f"  - Validation Size: {cls.VAL_SIZE}")
        print(f"  - Random Seed: {cls.RANDOM_SEED}")
        print()
        print(f"Device Settings:")
        print(f"  - Device: {cls.DEVICE}")
        print(f"  - Multi-GPU: {cls.USE_MULTI_GPU}")
        if torch.cuda.is_available():
            print(f"  - GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        print("=" * 80)


# Create default config instance
config = Config()
