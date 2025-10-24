"""
Graph Neural Network for ligand (molecule) encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class LigandGNN(nn.Module):
    """
    Graph Neural Network for encoding molecular structures.
    
    This module processes molecular graphs to extract ligand embeddings
    using Graph Convolutional Networks (GCN) with residual connections.
    """
    
    def __init__(self, input_dim=5, hidden_dim=96, output_dim=192, num_layers=2):
        """
        Initialize LigandGNN.
        
        Args:
            input_dim (int): Input feature dimension (atom features)
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output embedding dimension
            num_layers (int): Number of GCN layers
        """
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers with batch normalization
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection with layer normalization
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, data):
        """
        Forward pass through the GNN.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment [num_nodes]
        
        Returns:
            torch.Tensor: Ligand embedding [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # GCN layers with residual connections
        for conv, bn in zip(self.convs, self.batch_norms):
            x_residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            # Residual connection
            x = x + x_residual
        
        # Global pooling and output projection
        x = global_mean_pool(x, batch)
        x = self.output_projection(x)
        
        return x
