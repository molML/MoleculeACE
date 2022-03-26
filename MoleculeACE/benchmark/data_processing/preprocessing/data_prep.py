"""
All data prep functions used to fetch CHEMBL data, clean it, cluster, split and encode it
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import pandas as pd
import random
from progress.bar import ShadyBar

from rdkit import Chem
from rdkit.Chem import AllChem

from MoleculeACE.benchmark import data_processing
from MoleculeACE.benchmark.utils.const import Descriptors, RANDOM_SEED


def fetch_data(chembl_targetid='CHEMBL2047', endpoints=['EC50']):
    """Download and prep the data from CHEMBL. Throws out duplicates, problematic molecules, and extreme outliers"""
    from MoleculeACE.benchmark.data_processing.preprocessing.data_fetching import main_curator
    import os

    # fetch + curate data
    data = main_curator.main(chembl_targetid=chembl_targetid, endpoints=endpoints)
    # write to Data directory
    filename = os.path.join('Data', f"{chembl_targetid}_{'_'.join(endpoints)}.csv")
    data.to_csv(filename)


def prep_data(filename, smiles_colname: str = 'smiles', y_colname: str = 'exp_mean [nM]', chembl_id_colname: str = None,
              remove_stereo: bool = True):
    """ Perform some initial prepping of bioactivity data obtained from fetch_data()"""
    print(f"Prepping: {filename}\n"
          f"- {'R' if remove_stereo else 'Not r'}emoving stereochemical redundancies")

    # Read the data
    df = pd.read_csv(filename)
    # # remove rows containing na's
    if chembl_id_colname is not None:
        df.dropna(axis=0, subset=[smiles_colname, y_colname, chembl_id_colname], inplace=True)
    else:
        df.dropna(axis=0, subset=[smiles_colname, y_colname], inplace=True)

    # get the data from the pd.dataframe as a list
    smiles = df[smiles_colname].to_list()
    activities = df[y_colname].to_list()
    chembl_id = None if chembl_id_colname is None else df[chembl_id_colname].to_list()

    # Remove stereochemical redundancies if needed
    if remove_stereo:
        bad_smiles_idx = find_stereochemical_siblings(smiles)
        for index in sorted(bad_smiles_idx, reverse=True):
            del smiles[index]
            del activities[index]
            if chembl_id_colname is not None:
                del chembl_id[index]
        print(f'Removed {len(bad_smiles_idx)} stereochemical redundancies from a total of {len(df)} molecules')

    return smiles, activities, chembl_id


def clusterfp(fps, clustering_cutoff: float = 0.4):
    from rdkit.ML.Cluster import Butina
    from rdkit import DataStructs
    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, clustering_cutoff, isDistData=True)
    return cs


def cluster_smiles(smiles: list, clustering_cutoff: float = 0.4):
    """ Cluster smiles based on their Murcko scaffold using the Butina algorithm:

    D Butina 'Unsupervised Database Clustering Based on Daylight's Fingerprint and Tanimoto Similarity:
    A Fast and Automated Way to Cluster Small and Large Data Sets', JCICS, 39, 747-750 (1999)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

    # Make Murcko scaffolds
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    mols = [GetScaffoldForMol(m) for m in mols]

    # Create fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]

    # Cluster fingerprints
    clusters = clusterfp(fps, clustering_cutoff=clustering_cutoff)

    return clusters


def cliff_split_clusters(clusters, smiles: list, cliff_compounds: list, test_split: float = 0.2):
    """ Splitting clusters of indices """
    from sklearn.model_selection import train_test_split
    random.seed(RANDOM_SEED)

    train_idx, test_idx = [], []
    for clust in clusters:
        # Get the smiles and cliffs for this cluster
        clust_smiles = [smiles[idx] for idx in clust]
        clust_cliff_cpd = [int(s in cliff_compounds) for s in clust_smiles]

        try:
            # If cluster is too small, add randomly to train/test according to the split ratio
            if len(clust) < (1 / test_split):
                for idx in clust:
                    if random.uniform(0, 1) >= test_split:
                        train_idx.append(idx)
                    else:
                        test_idx.append(idx)

            # If cluster has just one cliff, or all cliffs distribute it randomly
            elif sum(clust_cliff_cpd) < 2 or sum(clust_cliff_cpd) == len(clust_cliff_cpd):
                for idx in clust:
                    if random.uniform(0, 1) >= test_split:
                        train_idx.append(idx)
                    else:
                        test_idx.append(idx)

            else:
                smiles_train_cl, smiles_test_cl = train_test_split(clust, stratify=clust_cliff_cpd,
                                                                   test_size=test_split, random_state=RANDOM_SEED)
                train_idx.extend(smiles_train_cl)
                test_idx.extend(smiles_test_cl)
        except:
            # If for some reason this gives a bug, add it to the train/test
            for idx in clust:
                if random.uniform(0, 1) >= test_split:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)

    return train_idx, test_idx


def split_data(smiles: list, cliff_compounds: list, test_size: float = 0.20, clustering_cutoff: float = 0.4):
    """ Split a list of SMILES stings by Murcko scaffold with Butina clustering and the presence of activity cliffs"""
    clusters = cluster_smiles(smiles, clustering_cutoff=clustering_cutoff)
    train_idx, test_idx = cliff_split_clusters(clusters, smiles, cliff_compounds, test_split=test_size)

    return train_idx, test_idx


def split_smiles(smiles: list, test_split: float = 0.2, clustering_cutoff: float = 0.4):
    """ Split list of SMILES strings by their Murcko scaffold with Butina clustering"""
    random.seed(RANDOM_SEED)

    clusters = cluster_smiles(smiles, clustering_cutoff=clustering_cutoff)

    train_idx, test_idx = [], []
    for clust in clusters:
        # For every item in a cluster, distribute it among train/test according to the test_split ratio
        for idx in clust:
            if random.uniform(0, 1) >= test_split:
                train_idx.append(idx)
            else:
                test_idx.append(idx)

    return train_idx, test_idx


def split_binary_fingerprint_array(x, test_split: float = 0.2, clustering_cutoff: float = 0.4):
    """ Split a np array of binary fingerprints by their Murcko scaffold with Butina clustering """
    from rdkit import DataStructs
    random.seed(RANDOM_SEED)

    fps = []
    for arr in x:
        bitstring = "".join(arr.astype(str))
        fp = DataStructs.cDataStructs.CreateFromBitString(bitstring)
        fps.append(fp)

    clusters = clusterfp(fps, clustering_cutoff=clustering_cutoff)
    train_idx, test_idx = [], []

    for clust in clusters:
        for idx in clust:
            # For every item in a cluster, distribute it among train/test according to the test_split ratio
            if random.uniform(0, 1) >= test_split:
                train_idx.append(idx)
            else:
                test_idx.append(idx)

    return train_idx, test_idx


def encode_smiles(smiles: list, descriptor: Descriptors):
    """ Encode a list of SMILES strings

    Args:
        smiles: list of SMILES strings
        descriptor: Type of descriptor.

    Returns: x: (np.array) encoded smiles, failed_idx: (lst) list of failed molecules (indices)

    """

    pickable = [d for d in Descriptors]
    available = [d.value for d in Descriptors]

    if descriptor not in pickable:
        raise ValueError(f"Descriptor '{descriptor}' is not supported. Pick from: {available}")

    failed_idx = []
    x = None

    if descriptor == Descriptors.MORGAN2 or descriptor == Descriptors.ECFP or descriptor == Descriptors.CIRCULAR:
        x = data_processing.smiles_to_morgan2(smiles)
    if descriptor == Descriptors.RDKIT:
        x = data_processing.smiles_to_rdkit(smiles)
    if descriptor == Descriptors.ATOM_PAIR:
        x = data_processing.smiles_to_atom_pair(smiles)
    if descriptor == Descriptors.TOPOLOGICAL_TORSION:
        x = data_processing.smiles_to_topological_torsion(smiles)
    if descriptor == Descriptors.SMARTS:
        x = data_processing.smiles_to_smarts(smiles)
    if descriptor == Descriptors.MACCS:
        x = data_processing.smiles_to_maccs(smiles)
    if descriptor == Descriptors.PHYSCHEM or descriptor == Descriptors.DRUG_LIKE:
        x = data_processing.smiles_to_drug_like(smiles)
    if descriptor == Descriptors.WHIM:
        x, failed_idx = data_processing.smiles_to_whim(smiles)
    if descriptor == Descriptors.SMILES or descriptor == Descriptors.LSTM:
        x, failed_idx = data_processing.smiles_to_onehot(smiles)
    if descriptor == Descriptors.GRAPH:
        x = smiles
    if descriptor == Descriptors.CANONICAL_GRAPH:
        x, failed_idx = data_processing.smiles_to_canonical_graph(smiles)
    if descriptor == Descriptors.ATTENTIVE_GRAPH:
        x, failed_idx = data_processing.smiles_to_attentive_graph(smiles)

    return x, failed_idx


def find_stereochemical_siblings(smiles: list):
    """ Detects molecules that have different SMILES strings, but ecode for the same molecule with
    different stereochemistry

    Args:
        smiles: (lst) list of SMILES strings

    Returns: (lst) List of indices of SMILES having a similar molecule with different stereochemistry

    """
    # Make a dict of SMILES:fingerprint to save time
    fp_db = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
        fp_db[smi] = ''.join([str(int(i)) for i in fp])

    # Make a dict containing every smile string + all the other smiles that result in the same fingerprint
    # {SMILES:[(SMILES, fp), (SMILES2, fp)]
    fingerprint_brothers = {}
    for smi in smiles:
        brothers = []
        in_fp = fp_db[smi]
        for i in smiles:
            if fp_db[i] == in_fp:
                brothers.append((i, fp_db[i]))
        fingerprint_brothers[smi] = brothers
    # Find the smiles that have at least one other smiles that yields the same fingerprint
    duplicated_fp_smiles = []
    for i in fingerprint_brothers.values():
        if len(i) > 1:
            duplicated_fp_smiles.extend([j[0] for j in i])

    # return the indices of the smiles to remove
    return [smiles.index(i) for i in list(set(duplicated_fp_smiles))]


def augment(smiles, activity=None, chembl=None, augment_factor=10, max_smiles_len=200,
            shuffle_data=False, waitbar=True):
    """ Augment SMILES strings by adding non-canonical SMILES. Keeps corresponding activity values/CHEMBL IDs

    Args:
        smiles: (lst) List of SMILES stringss
        activity: (lst) List of bioactivity values matching the SMILES
        chembl: (lst) List of CHEMBL (or any other) IDs matching the SMILES
        augment_factor: (int) augmenting SMILES n times. Augmentation of 10 would mean getting from 100 to 1000 SMILES
        max_smiles_len: (int) Maximal length of the generated SMILES strings
        shuffle_data: (bool) Shuffle the lists (together)
        waitbar: (bool) Use a progress bar?

    Returns: (smiles, activity, chembl)

    """
    # set the random seed and initiate some empty lists
    random.seed(RANDOM_SEED)
    new_smiles = []
    new_activities = []
    new_chembl = []
    if chembl is None:
        chembl = [None] * len(smiles)
    if activity is None:
        activity = [None] * len(smiles)

    if waitbar:
        bar = ShadyBar(f'Augmenting SMILES {augment_factor}x', max=len(smiles), check_tty=False)

    # For every SMILES string, add n extra smiles
    for idx in range(len(smiles)):
        if waitbar:
            bar.next()
        generated = smile_augmentation(smiles[idx], augment_factor - 1, max_smiles_len)
        new_smiles.extend(generated)
        new_activities.extend([activity[idx] for idx in range(len(generated))])
        new_chembl.extend([chembl[idx] for idx in range(len(generated))])

    if waitbar:
        bar.finish()

    smiles.extend(new_smiles)
    activity.extend(new_activities)
    chembl.extend(new_chembl)

    if shuffle_data:
        c = list(zip(smiles, activity, chembl))  # Shuffle all lists together
        random.shuffle(c)
        smiles, activity, chembl = zip(*c)

    return smiles, activity, chembl


def random_smiles(mol):
    """ Generate a random non-canonical SMILES string from a molecule"""
    # https://github.com/michael1788/virtual_libraries/blob/master/experiments/do_data_processing.py
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol)


def smile_augmentation(smile, augmentation, max_len):
    """Generate n random non-canonical SMILES strings from a SMILES string with length constraints"""
    # https://github.com/michael1788/virtual_libraries/blob/master/experiments/do_data_processing.py
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for i in range(1000):
        smiles = random_smiles(mol)
        if len(smiles) <= max_len:
            s.add(smiles)
            if len(s) == augmentation:
                break

    return list(s)
