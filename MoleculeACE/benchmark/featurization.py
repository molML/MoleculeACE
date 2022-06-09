"""
Author: Derek van Tilborg -- TU/e -- 20-05-2022

All code for featurizing a molecules into molecular descriptors, molecular graphs, one-hot endode SMILES strings and
tokens.

    - Featurizer:                   Main molecule featurizer class that converts SMILES strings to computable info.
                                        example:    x_train = Featurizer().ecfp(list_of_smiles)
        - ecfp()
        - maccs()
        - whim()
        - physchem()
        - one_hot()
        - tokens()
        - graphs()

    - OneHotEncodeSMILES:           Class that can one-hot encode SMILES strings
    - featurize_graph():            Construct a molecular graph from a SMILES string
    - compute_whim():               Computes WHIM descriptors from a SMILES string
    - compute_physchem():           Computes physical-chemical properties a SMILES string

"""
from MoleculeACE.benchmark.const import RANDOM_SEED, Descriptors
from MoleculeACE.benchmark.utils import get_config, smi_tokenizer
from MoleculeACE.benchmark.const import CONFIG_PATH_SMILES
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from typing import List
from tqdm import tqdm
import numpy as np
import torch
import os.path
import pickle


smiles_encoding = get_config(CONFIG_PATH_SMILES)

x_map = {'symbol': ["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S", "Si", 'Se', 'Te'],
         'atomic_weight': float,
         'n_valence': int,
         'num_hs': int,
         'degree': list(range(0, 7)),
         'hybridization': ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'],
         'partial_charge': float,
         'is_aromatic': [True, False],
         'is_in_ring': [True, False]}

e_map = {'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
         'is_conjugated': [True, False]}


class Featurizer:
    def __init__(self):
        self.scaler_physchem = None
        self.scaler_whim = None

    @staticmethod
    def ecfp(smiles: List[str], radius: int = 2, nbits: int = 1024):
        """ Convert SMILES to ECFP fingerprints """
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

        fp = [GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in mols_from_smiles(smiles)]
        return rdkit_numpy_convert(fp)

    @staticmethod
    def maccs(smiles: List[str]):
        """ Convert SMILES to MACCs fingerprints"""
        from rdkit.Chem import MACCSkeys

        fp = [MACCSkeys.GenMACCSKeys(m) for m in mols_from_smiles(smiles)]
        return rdkit_numpy_convert(fp)

    def whim(self, smiles: List[str], scale: bool = False, scale_test_on_train: bool = False):
        """ Compute WHIM descriptors from SMILES """
        from sklearn.preprocessing import StandardScaler

        # Compute WHIM descriptors on SMILES strings
        X = compute_whim(smiles)

        if scale_test_on_train:
            if self.scaler_whim is None:
                raise ValueError('Did not see training data yet. The autoscaler from train does not exist')
            X = self.scaler_whim.transform(X)
        elif scale:
            self.scaler_whim = StandardScaler().fit(X)
            X = self.scaler_whim.transform(X)

        return X

    def physchem(self, smiles: List[str], scale: bool = False, scale_test_on_train: bool = False):
        """ Calculate drug-like physchem descriptors from a rdkit mol object. """
        from sklearn.preprocessing import StandardScaler

        # Compute physchem descriptors on SMILES strings
        X = compute_physchem(smiles)

        if scale_test_on_train:
            if self.scaler_physchem is None:
                raise ValueError('Did not see training data yet. The autoscaler from train does not exist')
            X = self.scaler_physchem.transform(X)
        elif scale:
            self.scaler_physchem = StandardScaler().fit(X)
            X = self.scaler_physchem.transform(X)

        return X

    @staticmethod
    def one_hot(smiles: List[str], truncate: bool = True):
        return OneHotEncodeSMILES()(smiles, truncate)

    @staticmethod
    def tokens(smiles: List[str], max_smiles_length: int = 200, padding: bool = True, truncation: bool = True,
               auto_tokenizer: str = 'seyonec/PubChem10M_SMILES_BPE_450k'):
        """ Tokenize SMILES for a ChemBerta Transformer

        :param max_smiles_length: (int) Maximal allowable SMILES string length
        :param padding: (bool) allow padding
        :param truncation: (bool) allow truncation (you will need this for heterogeneous SMILES strings)
        :param auto_tokenizer: (str) name of the auto tokenizer provided by HuggingFace
        :return: Dict['input_ids': tensor, 'attention_mask': tensor], tensors are of shape N x max_smiles_length
        """

        from transformers import AutoTokenizer

        chemical_tokenizer = AutoTokenizer.from_pretrained(auto_tokenizer)
        tokens = chemical_tokenizer(smiles, return_tensors='pt', padding=padding, truncation=truncation,
                                    max_length=max_smiles_length)

        return tokens

    @staticmethod
    def graphs(smiles: List[str]):
        return [featurize_graph(smi) for smi in smiles]

    def __call__(self, descriptor: Descriptors, **kwargs):
        if descriptor.name == 'ECFP':
            return self.ecfp(**kwargs)
        if descriptor.name == 'MACCS':
            return self.maccs(**kwargs)
        if descriptor.name == 'PHYSCHEM':
            return self.physchem(**kwargs)
        if descriptor.name == 'WHIM':
            return self.whim(**kwargs)
        if descriptor.name == 'SMILES':
            return self.one_hot(**kwargs)
        if descriptor.name == 'TOKENS':
            return self.tokens(**kwargs)
        if descriptor.name == 'GRAPH':
            return self.graphs( **kwargs)

    def __repr__(self):
        return "Molecule Featurizer that converts SMILES strings to: ecfp(), maccs(), whim(), physchem(), one_hot(), " \
               "tokens(), graphs()"


class OneHotEncodeSMILES:
    """ Code to one-hot encode SMILES strings

    Adapted (added some rarer elements) from:

    Moret, M., Grisoni, F., Katzberger, P. & Schneider, G.
    Perplexity-based molecule ranking and bias estimation of chemical language models.
    ChemRxiv (2021) doi:10.26434/chemrxiv-2021-zv6f1-v2.
    """

    def __init__(self, max_len_model=smiles_encoding['max_smiles_len'] + 2, n_chars=smiles_encoding['vocab_size'],
                 indices_token=smiles_encoding['indices_token'], token_indices=smiles_encoding['token_indices'],
                 pad_char=smiles_encoding['pad_char'], start_char=smiles_encoding['start_char'],
                 end_char=smiles_encoding['end_char']):

        self.max_len_model = max_len_model
        self.n_chars = n_chars
        self.pad_char = pad_char
        self.start_char = start_char
        self.end_char = end_char
        self.indices_token = indices_token
        self.token_indices = token_indices

    def encode_smiles(self, smiles: str, truncate: bool = True):
        token_ints = self._smi_to_int(smiles, truncate)
        return self._one_hot_encode(token_ints, self.n_chars)

    def _one_hot_encode(self, token_list, n_chars):
        output = np.zeros((token_list.shape[0], n_chars))
        for j, token in enumerate(token_list):
            output[j, token] = 1
        return output

    def _smi_to_int(self, smi: str, truncate: bool = True):
        """
        this will turn a list of smiles in string format
        and turn them into a np array of int, with padding
        """
        token_list = smi_tokenizer(smi)

        if truncate:
            token_list = token_list[:(self.max_len_model-2)]

        token_list = [self.start_char] + token_list + [self.end_char]
        padding = [self.pad_char] * (self.max_len_model - len(token_list))
        token_list.extend(padding)
        int_list = [self.token_indices[x] for x in token_list]

        return np.asarray(int_list)

    def __call__(self, smiles, truncate: bool = True):

        if type(smiles) is str:
            return self.encode_smiles(smiles, truncate)
        if type(smiles) is list:
            return np.array([self.encode_smiles(smi, truncate) for smi in smiles])


def mols_from_smiles(smiles: List[str]):
    """ Create a list of RDkit mol objects from a list of SMILES strings """
    from rdkit.Chem import MolFromSmiles
    return [MolFromSmiles(m) for m in smiles]


def rdkit_numpy_convert(fp):
    """ Convert a RDkit fingerprint object to simple numpy array """
    from rdkit.DataStructs import ConvertToNumpyArray
    output = []
    for f in fp:
        arr = np.zeros((1,))
        ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def sigmoid(number: float):
    """ numerically semi-stable sigmoid function to map charge between 0 and 1 """
    return 1.0 / (1.0 + float(np.exp(-number)))


def molecule_from_smiles(smiles: str):
    """ Sanitize a molecule from a SMILES string"""

    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    Chem.rdPartialCharges.ComputeGasteigerCharges(molecule)

    return molecule


class GenFeatures(object):
    """ Adapted by Luke Rossen """

    def __init__(self):
        self.symbols = [
            "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S", "Si", 'Se', 'Te'
        ]

        self.hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]

    def __call__(self, data):
        # Create rdkit mol object
        mol = molecule_from_smiles(data.smiles)

        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.

            atomic_weight = sigmoid(Chem.GetPeriodicTable().GetAtomicWeight(atom.GetSymbol()))

            n_valence = float(atom.GetTotalValence())
            n_hydrogens = float(atom.GetTotalNumHs())

            degree = [0.] * 7
            degree[atom.GetDegree()] = 1.

            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())] = 1.

            partial_charge = sigmoid(float(atom.GetProp('_GasteigerCharge')))

            aromatic = [0.] * 2
            if atom.GetIsAromatic():
                aromatic[0] = 1.
            else:
                aromatic[1] = 1.

            in_ring = [0.] * 2
            if atom.IsInRing():
                in_ring[0] = 1.
            else:
                in_ring[1] = 1.

            x = torch.tensor(symbol + [atomic_weight] + [n_valence] +
                             [n_hydrogens] + degree + hybridization +
                             [partial_charge] + aromatic + in_ring)
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.

            conjugation = [0.] * 2
            if bond.GetIsConjugated():
                conjugation[0] = 1.
            else:
                conjugation[1] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic] + conjugation)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data


def featurize_graph(smiles: str):
    # Create RDkit molobject
    mol = molecule_from_smiles(smiles)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['symbol'].index(atom.GetSymbol()))
        x.append(0)
        x.append(0)
        x.append(0)
        x.append(x_map['degree'].index(atom.GetDegree()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(0)
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, len(xs))

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    # y = torch.tensor(label, dtype=torch.float).view(1, -1)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    graph = GenFeatures()(graph)

    return graph


def compute_whim(smiles: List[str]):

    from rdkit.Chem import rdMolDescriptors, AllChem, AddHs, MolFromSmiles

    whim_database = {}
    if os.path.exists("whim_database.pkl"):
        with open("whim_database.pkl", 'rb') as handle:
            whim_database = pickle.load(handle)

    X = []
    for smi in tqdm(smiles):

        if smi in whim_database:
            X.append(whim_database[smi])
        else:

            # Construct mol object add hydrogens
            m = MolFromSmiles(smi)
            mh = AddHs(m)

            # Use distance geometry to obtain initial coordinates for a molecule
            embed = AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=True, randomSeed=RANDOM_SEED,
                                          maxAttempts=5)

            if embed == -1:
                print(f"failed first attempt for molecule {smi}, trying for more embedding attempts")
                embed = AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=True, randomSeed=RANDOM_SEED,
                                              clearConfs=False, enforceChirality=False, maxAttempts=45)

            if embed == -1:
                print(f"failed second attempt for molecule {smi}, trying embedding w/o using basic knowledge")
                embed = AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=False, randomSeed=RANDOM_SEED,
                                              clearConfs=True, enforceChirality=False, maxAttempts=1000)

            if embed == -1:
                raise RuntimeError(f"FAILED embedding {smi}")

            AllChem.MMFFOptimizeMolecule(mh, maxIters=1000, mmffVariant='MMFF94')

            # calculate WHIM 3D descriptor
            X.append(rdMolDescriptors.CalcWHIM(mh))

    return np.array(X)


def compute_physchem(smiles: List[str]):

    from rdkit.Chem import Descriptors
    from rdkit import Chem

    X = []
    for m in mols_from_smiles(smiles):
        weight = Descriptors.ExactMolWt(m)
        logp = Descriptors.MolLogP(m)
        h_bond_donor = Descriptors.NumHDonors(m)
        h_bond_acceptors = Descriptors.NumHAcceptors(m)
        rotatable_bonds = Descriptors.NumRotatableBonds(m)
        atoms = Chem.rdchem.Mol.GetNumAtoms(m)
        heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(m)
        molar_refractivity = Chem.Crippen.MolMR(m)
        topological_polar_surface_area = Chem.QED.properties(m).PSA
        formal_charge = Chem.rdmolops.GetFormalCharge(m)
        rings = Chem.rdMolDescriptors.CalcNumRings(m)

        X.append(np.array([weight, logp, h_bond_donor, h_bond_acceptors, rotatable_bonds, atoms, heavy_atoms,
                           molar_refractivity, topological_polar_surface_area, formal_charge, rings]))

    return np.array(X)
