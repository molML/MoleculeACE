
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from typing import List


class ChemBertaDataset(Dataset):
    def __init__(self, smiles: List[str], labels: List[float]):
        """ Create a dataset for the ChemBerta transformer using a pretrained tokenizer """
        self.smiles = smiles
        self.labels = torch.tensor(labels).unsqueeze(1)
        self.chemical_tokenizer = AutoTokenizer.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')

        self.tokens = self.chemical_tokenizer(smiles, return_tensors='pt', padding=True, truncation=True,
                                              max_length=200)

    def __getitem__(self, idx):
        return self.tokens['input_ids'][idx], self.tokens['attention_mask'][idx], self.labels[idx]

    def __len__(self):
        return len(self.smiles)
