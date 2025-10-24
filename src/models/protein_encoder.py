"""
ESM-2 based protein sequence encoder.
"""

import torch
import torch.nn as nn
from transformers import EsmModel


class ProteinEncoder(nn.Module):
    """
    ESM-2 based protein sequence encoder.
    
    This module uses the ESM-2 (Evolutionary Scale Modeling) model
    to encode protein sequences into fixed-size embeddings.
    """
    
    def __init__(self, model_name, output_dim=192, freeze=True):
        """
        Initialize ProteinEncoder.
        
        Args:
            model_name (str): Name of the ESM-2 model to use
            output_dim (int): Output embedding dimension
            freeze (bool): Whether to freeze ESM-2 parameters
        """
        super().__init__()
        
        # Load pre-trained ESM-2 model
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Freeze ESM-2 parameters if requested
        if freeze:
            for param in self.esm.parameters():
                param.requires_grad = False
        
        # Get ESM-2 hidden size
        esm_hidden_size = self.esm.config.hidden_size
        
        # Projection layer to match desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(esm_hidden_size, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the protein encoder.
        
        Args:
            input_ids (torch.Tensor): Tokenized protein sequences [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
        
        Returns:
            tuple: (protein_embedding, hidden_states)
                - protein_embedding: [batch_size, output_dim]
                - hidden_states: [batch_size, seq_len, hidden_size]
        """
        # Get ESM-2 outputs
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        protein_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project to desired output dimension
        protein_embedding = self.projection(protein_embedding)
        
        return protein_embedding, outputs.last_hidden_state
