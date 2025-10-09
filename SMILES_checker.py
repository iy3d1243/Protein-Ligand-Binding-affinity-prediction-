import pandas as pd
from rdkit import Chem

df = pd.read_csv("Ki (nM)_only.csv")  # replace with your file path

smiles_column = "Ligand SMILES"  # replace with your actual column name

def is_valid_smiles(smi):
    return Chem.MolFromSmiles(smi) is not None

df['is_valid'] = df[smiles_column].apply(is_valid_smiles)

total = len(df)
valid_count = df['is_valid'].sum()
valid_percentage = (valid_count / total) * 100

print(f"Total SMILES: {total}")
print(f"Valid SMILES: {valid_count}")
print(f"Percentage valid: {valid_percentage:.2f}%")
