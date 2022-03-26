"""
Code to one-hot encode SMILES strings
Derek van Tilborg, Eindhoven University of Technology, March 2022

Adapted from:

Moret, M., Grisoni, F., Katzberger, P. & Schneider, G.
Perplexity-based molecule ranking and bias estimation of chemical language models.
ChemRxiv (2021) doi:10.26434/chemrxiv-2021-zv6f1-v2.
"""

import re
import numpy as np

from MoleculeACE.benchmark.utils import get_config
from MoleculeACE.benchmark.utils.const import CONFIG_PATH_SMILES
smiles_encoding = get_config(CONFIG_PATH_SMILES)


class OneHotEncode:
    def __init__(self, max_len_model, n_chars, indices_token, token_indices, pad_char, start_char, end_char):
        """Initialization"""
        self.max_len_model = max_len_model
        self.n_chars = n_chars

        self.pad_char = pad_char
        self.start_char = start_char
        self.end_char = end_char

        self.indices_token = indices_token
        self.token_indices = token_indices

    def one_hot_encode(self, token_list, n_chars):
        output = np.zeros((token_list.shape[0], n_chars))
        for j, token in enumerate(token_list):
            output[j, token] = 1
        return output

    def smi_to_int(self, smi: str):
        """
        this will turn a list of smiles in string format
        and turn them into a np array of int, with padding
        """
        token_list = smi_tokenizer(smi)
        token_list = [self.start_char] + token_list + [self.end_char]
        padding = [self.pad_char] * (self.max_len_model - len(token_list))
        token_list.extend(padding)
        int_list = [self.token_indices[x] for x in token_list]
        return np.asarray(int_list)

    def int_to_smile(self, array):
        """
        From an array of int, return a list of
        molecules in string smile format
        Note: remove the padding char
        """
        all_smi = []
        for seq in array:
            new_mol = [self.indices_token[int(x)] for x in seq]
            all_smi.append(''.join(new_mol).replace(self.pad_char, ''))
        return all_smi

    def smile_to_onehot(self, smiles: str):

        lines = smiles
        n_data = len(lines)

        x = np.empty((n_data, self.max_len_model, self.n_chars), dtype=int)

        for i, smi in enumerate(lines):
            # remove return line symbols
            smi = smi.replace('\n', '')
            # tokenize
            int_smi = self.smi_to_int(smi)
            # one hot encode
            x[i] = self.one_hot_encode(int_smi, self.n_chars)

        return x

    def generator_smile_to_onehot(self, smi: str):

        smi = smi.replace('\n', '')
        int_smi = self.smi_to_int(smi)
        one_hot = self.one_hot_encode(int_smi, self.n_chars)
        return one_hot


def onehot_to_smiles(array, indices_token: dict, start_char: str, end_char: str):
    """ Convert one-hot encoded array of SMILES back to a SMILES string

    Args:
        array: (np.array) one-hot encoded numpy array with shape (mol_length, vocab_length)
        indices_token: (dict) the location of each token in the vocab. {0: 'c', 1: 'C', 2: '(', etc}
        start_char: (str) character that signals the start of the sequence
        end_char: (str) character that signals the end of the sequence

    Returns: (str) SMILES string

    """
    smiles_string = ''
    for row in array:
        token = indices_token[int(np.where(row == 1)[0])]  # Find the corresponding token to this row
        if token == end_char:  # stop if you reach the end character
            break
        if token != start_char:
            smiles_string += token
    return smiles_string


def smi_tokenizer(smi: str):
    """
    Tokenize a SMILES
    """
    pattern = "(\[|\]|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Si|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    return tokens


def is_acceptable_smiles(smile: str, allowed_chars=smiles_encoding['indices_token'].values(),
                         min_len=smiles_encoding['min_smiles_len'], max_len=smiles_encoding['max_smiles_len']):
    """ Checks which smiles

    Args:
        smile: (str) smiles string
        allowed_chars: (lst) list of allowed smiles characters ['c', 'C', '(', ')', 'O', '1', '2', '=', 'N', ... ]
        min_len: (int) minimal smiles character length (default = 5)
        max_len: (int) minimal smiles character length (default = 200)

    Returns: (bool) True = allowed smile, False = weird smile

    """
    tokens = smi_tokenizer(smile)
    return len(tokens) >= min_len and len(tokens) <= max_len and all([tok in allowed_chars for tok in tokens])


