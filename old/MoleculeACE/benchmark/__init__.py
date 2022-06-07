""" This file contains all functions you need to run a benchmark on our data or your own data

from MoleculeACE import benchmark

data = benchmark.load_data('CHEMBL2047_EC50', descriptor=Descriptors.ECFP)
# or with your own data:    data = benchmark.process_data('yourdata.csv', descriptor=Descriptors.ECFP)
model = benchmark.train_model(data, algorithm=Algorithms.RF)
pred = model.predict(data.x_test)
results = benchmark.evaluate(data=data, predictions=pred)

"""
from old.MoleculeACE.benchmark.utils import Cliffs
from old.MoleculeACE.benchmark.evaluation import evaluate
from old.MoleculeACE.benchmark.data_processing import Data
from old.MoleculeACE.benchmark.data_processing import prep_data, split_data, encode_smiles, augment, drug_like_descriptor, \
    whim_descriptor, \
    smiles_to_onehot, smiles_to_morgan2, smiles_to_rdkit, \
    smiles_to_atom_pair, smiles_to_topological_torsion, smiles_to_smarts, smiles_to_maccs, smiles_to_drug_like, \
    smiles_to_whim, smiles_to_canonical_graph, smiles_to_attentive_graph
from old.MoleculeACE.benchmark.data_processing import load_data
from old.MoleculeACE.benchmark.data_processing import process_data
from old.MoleculeACE.benchmark.models import Model
from old.MoleculeACE.benchmark.models import train_model
