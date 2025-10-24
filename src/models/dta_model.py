"""
Complete Drug-Target Affinity prediction model.
"""

import torch
import torch.nn as nn

from .ligand_gnn import LigandGNN
from .protein_encoder import ProteinEncoder
from .cross_attention import BiDirectionalCrossAttention


class DTAModel(nn.Module):
    """
    Complete Drug-Target Affinity prediction model.
    
    This model combines:
    1. GNN for ligand encoding
    2. ESM-2 for protein encoding  
    3. Bidirectional cross-attention for interaction modeling
    4. MLP for final affinity prediction
    """
    
    def __init__(self, config):
        """
        Initialize DTA model.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        
        # Ligand encoder (GNN)
        self.ligand_encoder = LigandGNN(
            input_dim=config.GNN_INPUT_DIM,
            hidden_dim=config.GNN_HIDDEN_DIM,
            output_dim=config.LIGAND_EMBED_DIM,
            num_layers=config.GNN_NUM_LAYERS
        )
        
        # Protein encoder (ESM-2)
        self.protein_encoder = ProteinEncoder(
            model_name=config.ESM_MODEL,
            output_dim=config.PROTEIN_EMBED_DIM,
            freeze=config.FREEZE_ESM
        )
        
        # Cross-attention mechanism
        self.cross_attention = BiDirectionalCrossAttention(
            embed_dim=config.LIGAND_EMBED_DIM,
            num_heads=config.ATTENTION_HEADS,
            dropout=config.ATTENTION_DROPOUT
        )
        
        # Final prediction MLP
        self._build_mlp(config)
    
    def _build_mlp(self, config):
        """
        Build the final prediction MLP.
        
        Args:
            config: Configuration object with MLP parameters
        """
        mlp_layers = []
        
        for i in range(len(config.MLP_DIMS) - 1):
            # Linear layer
            mlp_layers.append(nn.Linear(config.MLP_DIMS[i], config.MLP_DIMS[i+1]))
            
            # Activation (except for last layer)
            if i < len(config.MLP_DIMS) - 2:
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(config.MLP_DROPOUT))
            else:
                mlp_layers.append(nn.Identity())  # No activation for output layer
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, graph_batch, protein_input_ids, protein_attention_mask):
        """
        Forward pass through the complete model.
        
        Args:
            graph_batch: Batched molecular graphs
            protein_input_ids (torch.Tensor): Tokenized protein sequences
            protein_attention_mask (torch.Tensor): Protein attention masks
        
        Returns:
            tuple: (pki_prediction, attention_weights)
                - pki_prediction: [batch_size] - predicted pKi values
                - attention_weights: tuple of attention weights for visualization
        """
        # Encode ligand (molecule)
        ligand_embed = self.ligand_encoder(graph_batch)
        
        # Encode protein
        protein_embed, _ = self.protein_encoder(protein_input_ids, protein_attention_mask)
        
        # Cross-attention between ligand and protein
        lig_attn, prot_attn, attn_weights = self.cross_attention(ligand_embed, protein_embed)
        
        # Combine all representations
        # Concatenate: ligand_attended + protein_attended + original_protein
        combined = torch.cat([lig_attn, prot_attn, protein_embed], dim=1)
        
        # Final prediction
        pki_pred = self.mlp(combined).squeeze(-1)
        
        return pki_pred, attn_weights
