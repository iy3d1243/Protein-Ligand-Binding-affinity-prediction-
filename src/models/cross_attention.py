"""
Bidirectional cross-attention mechanism for ligand-protein interaction modeling.
"""

import torch
import torch.nn as nn


class BiDirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention mechanism.
    
    This module implements bidirectional attention between ligand and protein
    embeddings to model their interactions. It allows both ligand-to-protein
    and protein-to-ligand attention flows.
    """
    
    def __init__(self, embed_dim=192, num_heads=6, dropout=0.15):
        """
        Initialize BiDirectionalCrossAttention.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        
        # Ligand-to-protein attention
        self.ligand_to_protein = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Protein-to-ligand attention
        self.protein_to_ligand = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for residual connections
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, ligand_embed, protein_embed):
        """
        Forward pass through bidirectional cross-attention.
        
        Args:
            ligand_embed (torch.Tensor): Ligand embeddings [batch_size, embed_dim]
            protein_embed (torch.Tensor): Protein embeddings [batch_size, embed_dim]
        
        Returns:
            tuple: (ligand_attended, protein_attended, attention_weights)
                - ligand_attended: [batch_size, embed_dim]
                - protein_attended: [batch_size, embed_dim]
                - attention_weights: tuple of (ligand_weights, protein_weights)
        """
        # Reshape for attention (add sequence dimension)
        ligand_query = ligand_embed.unsqueeze(1)  # [batch_size, 1, embed_dim]
        protein_query = protein_embed.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Ligand-to-protein attention
        lig_attn, lig_weights = self.ligand_to_protein(
            query=ligand_query,
            key=protein_query,
            value=protein_query
        )
        # Residual connection and normalization
        lig_attn = self.norm1(lig_attn.squeeze(1) + ligand_embed)
        
        # Protein-to-ligand attention
        prot_attn, prot_weights = self.protein_to_ligand(
            query=protein_query,
            key=ligand_query,
            value=ligand_query
        )
        # Residual connection and normalization
        prot_attn = self.norm2(prot_attn.squeeze(1) + protein_embed)
        
        return lig_attn, prot_attn, (lig_weights, prot_weights)
