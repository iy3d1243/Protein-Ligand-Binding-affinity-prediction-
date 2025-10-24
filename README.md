# Drug-Target Affinity (DTA) Prediction

A deep learning model for predicting drug-target binding affinity using Graph Neural Networks (GNN), ESM-2 protein encoding, and bidirectional cross-attention mechanisms.

## 🧬 Model Architecture

This project implements a state-of-the-art model that combines:

- **Graph Neural Network (GNN)**: Encodes molecular structures from SMILES strings
- **ESM-2 (Evolutionary Scale Modeling)**: Encodes protein sequences using transformer architecture
- **Bidirectional Cross-Attention**: Models interactions between ligands and proteins
- **Multi-GPU Training**: Optimized for 2x T4 GPUs with data persistence

## 🚀 Features

- **Memory Efficient**: Optimized for 2x T4 GPUs (16GB each) with reduced model dimensions
- **Data Persistence**: Preprocessed data is cached to disk for faster subsequent training
- **Multi-GPU Support**: Automatic DataParallel for multi-GPU training
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Accumulation**: Effective larger batch sizes
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Comprehensive Metrics**: R², RMSE, MAE evaluation

## 📁 Project Structure

```
dta_prediction_project/
├── src/                          # Source code
│   ├── models/                   # Model components
│   │   ├── dta_model.py         # Complete DTA model
│   │   ├── ligand_gnn.py         # GNN for ligand encoding
│   │   ├── protein_encoder.py    # ESM-2 protein encoder
│   │   └── cross_attention.py    # Bidirectional attention
│   ├── data/                     # Data processing
│   │   ├── dataset.py           # Dataset classes
│   │   └── preprocessing.py     # Data preprocessing
│   ├── training/                 # Training utilities
│   │   └── trainer.py           # Trainer class
│   └── utils/                    # Utility functions
│       ├── model_utils.py       # Model utilities
│       └── data_utils.py        # Data utilities
├── configs/                      # Configuration
│   └── config.py                # Model configuration
├── data/                         # Data directories
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed data
│   └── cache/                   # Cached data
├── outputs/                      # Output directories
│   ├── checkpoints/             # Model checkpoints
│   ├── logs/                    # Training logs
│   └── results/                  # Results
├── scripts/                      # Utility scripts
├── tests/                        # Test files
├── main.py                      # Main execution script
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: 2x T4 GPUs)
- 16GB+ GPU memory

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd dta_prediction_project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch Geometric (if needed):**
   ```bash
   pip install torch-geometric
   ```

## 📊 Data Format

The model expects CSV data with the following columns:

- `Ligand SMILES`: SMILES string of the ligand molecule
- `pKi`: Binding affinity value (target variable)
- Protein sequence column (one of):
  - `BindingDB Target Chain Sequence`
  - `BindingDB Target Chain Sequence 1`
  - `Protein Sequence`
  - `protein_sequence`
  - `sequence`

## 🚀 Usage

### 1. Data Preprocessing

First, preprocess your data and cache it for faster training:

```bash
python main.py --mode preprocess --data_path /path/to/your/data.csv
```

This will:
- Load and clean the data
- Convert SMILES to molecular graphs
- Tokenize protein sequences
- Split into train/validation/test sets
- Cache processed data to disk

### 2. Model Training

Train the model using cached data:

```bash
python main.py --mode train
```

This will:
- Load cached data (no reprocessing needed)
- Initialize the model with optimized weights
- Train with multi-GPU support
- Save best model checkpoints
- Evaluate on test set

### 3. Model Evaluation

Evaluate a trained model:

```bash
python main.py --mode evaluate
```

This will:
- Load the best model checkpoint
- Evaluate on train/validation/test sets
- Print comprehensive metrics

## ⚙️ Configuration

The model configuration is centralized in `configs/config.py`. Key parameters:

### Model Architecture
- `LIGAND_EMBED_DIM`: Ligand embedding dimension (192)
- `PROTEIN_EMBED_DIM`: Protein embedding dimension (192)
- `GNN_HIDDEN_DIM`: GNN hidden dimension (96)
- `ATTENTION_HEADS`: Number of attention heads (6)

### Training Settings
- `BATCH_SIZE`: Batch size per GPU (4)
- `GRADIENT_ACCUMULATION`: Gradient accumulation steps (2)
- `LEARNING_RATE`: Learning rate (3e-5)
- `NUM_EPOCHS`: Maximum epochs (100)
- `PATIENCE`: Early stopping patience (15)

### Data Settings
- `TEST_SIZE`: Test set fraction (0.15)
- `VAL_SIZE`: Validation set fraction (0.15)
- `ESM_MAX_LENGTH`: Maximum protein length (800)

## 🔧 Advanced Usage

### Custom Configuration

Create a custom config file:

```python
# custom_config.py
from configs.config import Config

class CustomConfig(Config):
    LIGAND_EMBED_DIM = 256  # Larger embeddings
    BATCH_SIZE = 8          # Larger batch size
    LEARNING_RATE = 1e-4    # Higher learning rate
```

### Multi-GPU Training

The model automatically detects and uses multiple GPUs:

```python
# Check GPU availability
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

### Memory Optimization

For limited GPU memory:

```python
# Reduce model dimensions
config.LIGAND_EMBED_DIM = 128
config.PROTEIN_EMBED_DIM = 128
config.GNN_HIDDEN_DIM = 64
config.BATCH_SIZE = 2
config.GRADIENT_ACCUMULATION = 4
```

## 📈 Performance

### Memory Usage (2x T4 GPUs)
- **Model Size**: ~50M parameters
- **GPU Memory**: ~12GB per GPU
- **Effective Batch Size**: 16 (4 × 2 GPUs × 2 accumulation)

## 🙏 Acknowledgments

- **ESM-2**: Facebook AI Research for protein language models
- **PyTorch Geometric**: For graph neural network utilities
- **RDKit**: For molecular informatics
- **Transformers**: Hugging Face for transformer models

