"""
Dataset classes for Drug-Target Affinity prediction.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from rdkit import Chem
from tqdm import tqdm


def smiles_to_graph(smiles):
    """
    Convert SMILES string to PyTorch Geometric graph.
    
    Args:
        smiles (str): SMILES string representation of molecule
        
    Returns:
        Data: PyTorch Geometric Data object or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),           # Atomic number
            atom.GetDegree(),              # Number of bonds
            atom.GetFormalCharge(),        # Formal charge
            atom.GetHybridization().real,  # Hybridization type
            int(atom.GetIsAromatic())      # Aromaticity (0 or 1)
        ]
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Extract bond information
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        # Add both directions for undirected graph
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([bond_type, bond_type])
    
    # Handle molecules with no bonds
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float).unsqueeze(1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class DTADataset(Dataset):
    """
    Drug-Target Affinity Dataset with disk caching support.
    
    This dataset handles:
    - SMILES to molecular graph conversion
    - Protein sequence tokenization
    - pKi value normalization
    - Data caching for faster subsequent loading
    """
    
    def __init__(self, df, tokenizer, max_length=800, normalize_pki=True, 
                 pki_mean=None, pki_std=None, cache_data=None):
        """
        Initialize DTA dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            tokenizer: ESM-2 tokenizer for protein sequences
            max_length (int): Maximum protein sequence length
            normalize_pki (bool): Whether to normalize pKi values
            pki_mean (float): Mean pKi for normalization (if None, computed from data)
            pki_std (float): Std pKi for normalization (if None, computed from data)
            cache_data (dict): Pre-processed data from cache
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.normalize_pki = normalize_pki
        
        # Set up normalization statistics
        if normalize_pki:
            if pki_mean is None:
                self.pki_mean = df['pKi'].mean()
                self.pki_std = df['pKi'].std()
            else:
                self.pki_mean = pki_mean
                self.pki_std = pki_std
        
        # Load from cache or process data
        if cache_data is not None:
            print("✓ Loading from cache...")
            self.graphs = cache_data['graphs']
            self.protein_tokens = cache_data['protein_tokens']
            self.df = cache_data['df']
            print(f"✓ Loaded {len(self.df)} samples from cache")
        else:
            self._process_data()
    
    def _process_data(self):
        """Process SMILES and protein sequences (called only if no cache)"""
        valid_indices = []
        self.graphs = []
        
        print("Processing SMILES to molecular graphs...")
        for idx, smiles in enumerate(tqdm(self.df['Ligand SMILES'], desc="SMILES")):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                valid_indices.append(idx)
        
        # Filter out invalid molecules
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"✓ Valid molecules: {len(self.df)} / {len(self.df) + len(valid_indices)}")
        
        # Tokenize protein sequences
        print("Pre-tokenizing protein sequences...")
        self.protein_tokens = []
        
        for idx in tqdm(range(len(self.df)), desc="Proteins"):
            row = self.df.iloc[idx]
            sequence = self._get_protein_sequence(row)
            
            tokens = self.tokenizer(
                sequence,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            self.protein_tokens.append({
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0)
            })
        
        print(f"✓ Pre-processing complete!\n")
    
    def _get_protein_sequence(self, row):
        """
        Find protein sequence from various possible column names.
        
        Args:
            row (pd.Series): DataFrame row
            
        Returns:
            str: Protein sequence
            
        Raises:
            KeyError: If no protein sequence column is found
        """
        possible_names = [
            'BindingDB Target Chain Sequence',
            'BindingDB Target Chain Sequence 1',
            'Protein Sequence',
            'protein_sequence',
            'sequence'
        ]
        
        # Try known column names first
        for name in possible_names:
            if name in row.index:
                return row[name]
        
        # Try to find any column with 'sequence' in the name
        seq_cols = [col for col in row.index if 'sequence' in col.lower()]
        if seq_cols:
            return row[seq_cols[0]]
        
        raise KeyError(f"Cannot find protein sequence column. Available: {list(row.index)}")
    
    def get_cache_data(self):
        """
        Return data for caching.
        
        Returns:
            dict: Dictionary containing graphs, protein tokens, and dataframe
        """
        return {
            'graphs': self.graphs,
            'protein_tokens': self.protein_tokens,
            'df': self.df
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Dictionary containing graph, protein tokens, and pKi value
        """
        row = self.df.iloc[idx]
        
        graph = self.graphs[idx]
        protein_input_ids = self.protein_tokens[idx]['input_ids']
        protein_attention_mask = self.protein_tokens[idx]['attention_mask']
        
        # Get and normalize pKi value
        pki = row['pKi']
        if self.normalize_pki:
            pki = (pki - self.pki_mean) / self.pki_std
        pki = torch.tensor(pki, dtype=torch.float)
        
        return {
            'graph': graph,
            'protein_input_ids': protein_input_ids,
            'protein_attention_mask': protein_attention_mask,
            'pki': pki
        }


def collate_fn(batch):
    """
    Custom collate function for batching PyTorch Geometric data.
    
    Args:
        batch (list): List of samples from the dataset
        
    Returns:
        dict: Batched data dictionary
    """
    graphs = [item['graph'] for item in batch]
    graph_batch = Batch.from_data_list(graphs)
    
    protein_input_ids = torch.stack([item['protein_input_ids'] for item in batch])
    protein_attention_mask = torch.stack([item['protein_attention_mask'] for item in batch])
    pki = torch.stack([item['pki'] for item in batch])
    
    return {
        'graph_batch': graph_batch,
        'protein_input_ids': protein_input_ids,
        'protein_attention_mask': protein_attention_mask,
        'pki': pki
    }
