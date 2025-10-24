"""
Test cases for model components.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from configs.config import config
from src.models import DTAModel, LigandGNN, ProteinEncoder, BiDirectionalCrossAttention


def test_ligand_gnn():
    """Test LigandGNN model."""
    model = LigandGNN(
        input_dim=5,
        hidden_dim=32,
        output_dim=64,
        num_layers=2
    )
    
    # Create dummy graph data
    from torch_geometric.data import Data, Batch
    x = torch.randn(10, 5)  # 10 atoms, 5 features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Forward pass
    output = model(data)
    
    assert output.shape == (1, 64)  # 1 molecule, 64 features
    assert not torch.isnan(output).any()


def test_protein_encoder():
    """Test ProteinEncoder model."""
    model = ProteinEncoder(
        model_name="facebook/esm2_t6_8M_UR50D",  # Smaller model for testing
        output_dim=64,
        freeze=True
    )
    
    # Create dummy protein data
    batch_size = 2
    seq_len = 100
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    output, hidden_states = model(input_ids, attention_mask)
    
    assert output.shape == (batch_size, 64)
    assert hidden_states.shape == (batch_size, seq_len, model.esm.config.hidden_size)
    assert not torch.isnan(output).any()


def test_cross_attention():
    """Test BiDirectionalCrossAttention."""
    model = BiDirectionalCrossAttention(
        embed_dim=64,
        num_heads=4,
        dropout=0.1
    )
    
    batch_size = 2
    embed_dim = 64
    
    ligand_embed = torch.randn(batch_size, embed_dim)
    protein_embed = torch.randn(batch_size, embed_dim)
    
    # Forward pass
    lig_attn, prot_attn, attn_weights = model(ligand_embed, protein_embed)
    
    assert lig_attn.shape == (batch_size, embed_dim)
    assert prot_attn.shape == (batch_size, embed_dim)
    assert len(attn_weights) == 2  # Two attention weight tensors
    assert not torch.isnan(lig_attn).any()
    assert not torch.isnan(prot_attn).any()


def test_dta_model():
    """Test complete DTA model."""
    # Create smaller config for testing
    test_config = config
    test_config.LIGAND_EMBED_DIM = 64
    test_config.PROTEIN_EMBED_DIM = 64
    test_config.GNN_HIDDEN_DIM = 32
    test_config.MLP_DIMS = [192, 64, 1]  # 3*64
    
    model = DTAModel(test_config)
    
    # Create dummy data
    from torch_geometric.data import Data, Batch
    
    # Graph data
    x = torch.randn(10, 5)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    graph_data = Data(x=x, edge_index=edge_index, batch=batch)
    graph_batch = Batch.from_data_list([graph_data, graph_data])
    
    # Protein data
    batch_size = 2
    seq_len = 50
    protein_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    protein_attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    pki_pred, attn_weights = model(graph_batch, protein_input_ids, protein_attention_mask)
    
    assert pki_pred.shape == (batch_size,)
    assert len(attn_weights) == 2
    assert not torch.isnan(pki_pred).any()


if __name__ == "__main__":
    # Run tests
    test_ligand_gnn()
    print("✓ LigandGNN test passed")
    
    test_protein_encoder()
    print("✓ ProteinEncoder test passed")
    
    test_cross_attention()
    print("✓ BiDirectionalCrossAttention test passed")
    
    test_dta_model()
    print("✓ DTAModel test passed")
    
    print("\n🎉 All tests passed!")
