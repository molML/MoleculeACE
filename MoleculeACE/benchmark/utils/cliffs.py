"""
This class computes activity cliffs on all compounds
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import numpy as np
from progress.bar import ShadyBar
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from Levenshtein import distance as levenshtein

from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


class Cliffs:
    """Find activity cliffs for a list of molecules and their bioactivity
        We use a three-part activity cliff definition consisting of Tanimoto similarity, generic Murcko scaffold
        similarity, and Levenshtein similarity. Compound pairs that have a similarity >= 0.7 and a fold-change
        in bioactivity of >= 10x, are considered a activity cliff. Because activity cliffs can be defined from
        different angles, we use a soft-consensus to determine if a compound is an activity cliff compound"""

    def __init__(self, smiles: list, activity: list, test_smiles: list = None, test_activity: list = None,
                 in_log10: bool = True, fold_threshold: int = 10, similarity_threshold: float = 0.9):
        """
            smiles: (lst) List of SMILES strings
            activity: (lst) List of bioactivities (standard in log10 [nM])
            test_smiles: (lst) List of SMILES strings for compounds in the test set (optional)
            test_activity: (lst) List of bioactivities (standard in log10 [nM]) for compounds in the test set (optional)
            in_log10: (bool) is the bioactivity provided in log10? If True (default) it will be deconverted to determine
            the fold change.
            fold_threshold: (int) how much must the fold change be before being an activity cliff? (default=10)
            similarity_threshold: (float) how similar must two compounds be before being an activity cliff (default=0.7)
        """
        self.in_log10 = in_log10
        self.smiles = smiles
        self.test_smiles = test_smiles
        if test_smiles is None:
            self.test_smiles = []
        self.all_smiles = self.smiles + self.test_smiles

        self.activity = activity
        self.test_activity = test_activity
        if test_activity is None:
            self.test_activity = []
        self.all_activity = self.activity + self.test_activity

        self.train_idx = list(range(len(smiles)))
        self.test_idx = list(range(len(smiles), len(self.all_smiles)))

        # Initiate a shitload of empty variables
        self.fc = None
        self.tanimoto_sim = None
        self.levenshtein_sim = None
        self.scaffold_sim = None
        self.tanimoto = None
        self.levenshtein = None
        self.scaffold = None
        self.soft_consensus = None
        self.tanimoto_tr = None
        self.tanimoto_tst = None
        self.scaffold_tr = None
        self.scaffold_tst = None
        self.levenshtein_tr = None
        self.levenshtein_tst = None
        self.soft_consensus_tr = None
        self.soft_consensus_tst = None
        self.db_fp = None
        self.db_scaf = None
        self.fold_threshold = fold_threshold
        self.similarity_threshold = similarity_threshold
        self.cliff_mols_tanimoto = None
        self.cliff_mols_tanimoto_tr = None
        self.cliff_mols_tanimoto_tst = None
        self.cliff_mols_scaffold = None
        self.cliff_mols_scaffold_tr = None
        self.cliff_mols_scaffold_tst = None
        self.cliff_mols_levenshtein = None
        self.cliff_mols_levenshtein_tr = None
        self.cliff_mols_levenshtein_tst = None
        self.cliff_mols_soft_consensus = None
        self.cliff_mols_soft_consensus_tr = None
        self.cliff_mols_soft_consensus_tst = None
        self.stats = None

        # Find cliffs, find which compounds are activity cliff compounds and subset the results if there is test data
        self.find_cliffs(fold_threshold, similarity_threshold)
        self.find_cliff_compounds()
        self.train_test_cliffs(self.train_idx, self.test_idx)
        self.get_stats()

    def find_fc(self, a: float, b: float):
        """Get the fold change of to bioactivities (deconvert from log10 if needed)"""

        if self.in_log10:
            a, b = 10**a, 10**b
        return max([a, b]) / min([a, b])

    def get_fc(self, waitbar: bool = True):
        """ Calculates the pairwise fold difference in compound activity given a list of activities"""

        act_len = len(self.all_activity)
        m = np.zeros([act_len, act_len])
        if waitbar:
            bar = ShadyBar('Calculating pairwise fold change                        ', max=act_len, check_tty=False)
        # Calculate upper triangle of matrix
        for i in range(act_len):
            if waitbar:
                bar.next()
            for j in range(i, act_len):
                m[i, j] = self.find_fc(self.all_activity[i], self.all_activity[j])

        if waitbar:
            bar.finish()

        # Fill in the lower triangle without having to loop (saves ~50% of time)
        m = m + m.T - np.diag(np.diag(m))
        # Fill the diagonal with 0's
        np.fill_diagonal(m, 0)

        self.fc = m

    def get_levenshtein_matrix(self, waitbar: bool = True, normalize: bool = True):
        """ Calculates a matrix of levenshtein similarity scores for a list of SMILES string"""

        smiles = self.all_smiles
        smi_len = len(smiles)
        if waitbar:
            bar = ShadyBar('Calculating pairwise Levenshtein similarity (normalized)', max=smi_len, check_tty=False)
        m = np.zeros([smi_len, smi_len])
        # Calculate upper triangle of matrix
        for i in range(smi_len):
            if waitbar:
                bar.next()
            for j in range(i, smi_len):
                if normalize:
                    m[i, j] = levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j]))
                else:
                    m[i, j] = levenshtein(smiles[i], smiles[j])
        if waitbar:
            bar.finish()
        # Fill in the lower triangle without having to loop (saves ~50% of time)
        m = m + m.T - np.diag(np.diag(m))
        # Get from a distance to a similarity
        m = 1 - m

        # Fill the diagonal with 0's
        np.fill_diagonal(m, 0)

        self.levenshtein_sim = m

    def get_tanimoto_matrix(self, waitbar: bool = True, radius: int = 2, nBits: int = 1024):
        """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string"""

        # Make a fingerprint database
        self.db_fp = {}
        for smi in self.all_smiles:
            m = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
            self.db_fp[smi] = fp

        smi_len = len(self.all_smiles)
        if waitbar:
            bar = ShadyBar('Calculating pairwise Tanimoto similarity                ', max=smi_len, check_tty=False)
        m = np.zeros([smi_len, smi_len])
        # Calculate upper triangle of matrix
        for i in range(smi_len):
            if waitbar:
                bar.next()
            for j in range(i, smi_len):
                m[i, j] = DataStructs.TanimotoSimilarity(self.db_fp[self.all_smiles[i]],
                                                         self.db_fp[self.all_smiles[j]])
        if waitbar:
            bar.finish()
        # Fill in the lower triangle without having to loop (saves ~50% of time)
        m = m + m.T - np.diag(np.diag(m))
        # Fill the diagonal with 0's
        np.fill_diagonal(m, 0)

        self.tanimoto_sim = m

    def get_scaffold_matrix(self, waitbar: bool = True, radius: int = 2, nBits: int = 1024):
        """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string """

        # Make scaffold database
        self.db_scaf = {}
        for smi in self.all_smiles:
            m = Chem.MolFromSmiles(smi)
            try:
                skeleton = GraphFramework(m)
            except Exception:  # In the very rare case this doesnt work, use a normal scaffold
                print(f"Could not create a generic scaffold of {smi}, used a normal scaffold instead")
                skeleton = GetScaffoldForMol(m)
            skeleton_fp = AllChem.GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)
            self.db_scaf[smi] = skeleton_fp

        smi_len = len(self.all_smiles)
        if waitbar:
            bar = ShadyBar('Calculating pairwise generic scaffold similarity        ', max=smi_len, check_tty=False)
        m = np.zeros([smi_len, smi_len])
        # Calculate upper triangle of matrix
        for i in range(smi_len):
            if waitbar:
                bar.next()
            for j in range(i, smi_len):
                m[i, j] = DataStructs.TanimotoSimilarity(self.db_scaf[self.all_smiles[i]],
                                                         self.db_scaf[self.all_smiles[j]])
        if waitbar:
            bar.finish()
        # Fill in the lower triangle without having to loop (saves ~50% of time)
        m = m + m.T - np.diag(np.diag(m))
        # Fill the diagonal with 0's
        np.fill_diagonal(m, 0)

        self.scaffold_sim = m

    def find_cliffs(self, fold_threshold: int = 10, similarity_threshold: float = 0.90):
        """Find Tanimoto, scaffold, and Levenshtein activity cliffs """

        # Calculate fold change and similarity
        self.get_fc()
        self.get_tanimoto_matrix()
        self.get_scaffold_matrix()
        self.get_levenshtein_matrix()

        # Determine which compound pairs are cliffs
        self.tanimoto = np.logical_and(self.fc > fold_threshold,
                                       self.tanimoto_sim > similarity_threshold).astype(int)

        self.levenshtein = np.logical_and(self.fc > fold_threshold,
                                          self.levenshtein_sim > similarity_threshold).astype(int)

        self.scaffold = np.logical_and(self.fc > fold_threshold,
                                       self.scaffold_sim > similarity_threshold).astype(int)

        # If a compound pair is a cliff in at least 1 method for soft consensus
        self.soft_consensus = self.tanimoto + self.scaffold + self.levenshtein
        self.soft_consensus[self.soft_consensus > 0] = 1

    def find_cliff_compounds(self):
        """ Find activity cliff compounds (having at least 1 cliff with any compound) for the different cliff types"""
        self.cliff_mols_tanimoto = [s for i, s in enumerate(self.all_smiles) if sum(self.tanimoto[i]) > 0]
        self.cliff_mols_scaffold = [s for i, s in enumerate(self.all_smiles) if sum(self.scaffold[i]) > 0]
        self.cliff_mols_levenshtein = [s for i, s in enumerate(self.all_smiles) if sum(self.levenshtein[i]) > 0]
        self.cliff_mols_soft_consensus = [s for i, s in enumerate(self.all_smiles) if sum(self.soft_consensus[i]) > 0]

    def train_test_cliffs(self, train_idx: list, test_idx: list):
        """ Subset the activity cliff matrices with the train/test indices"""

        # Tanimoto
        self.tanimoto_tr = self.tanimoto[np.ix_(train_idx, train_idx)]
        self.tanimoto_tst = self.tanimoto[np.ix_(test_idx, test_idx)]
        self.cliff_mols_tanimoto_tr = [s for s in self.cliff_mols_tanimoto if s in self.smiles]
        self.cliff_mols_tanimoto_tst = [s for s in self.cliff_mols_tanimoto if s in self.test_smiles]

        # Scaffold
        self.scaffold_tr = self.scaffold[np.ix_(train_idx, train_idx)]
        self.scaffold_tst = self.scaffold[np.ix_(test_idx, test_idx)]
        self.cliff_mols_scaffold_tr = [s for s in self.cliff_mols_scaffold if s in self.smiles]
        self.cliff_mols_scaffold_tst = [s for s in self.cliff_mols_scaffold if s in self.test_smiles]

        # Levenshtein
        self.levenshtein_tr = self.levenshtein[np.ix_(train_idx, train_idx)]
        self.levenshtein_tst = self.levenshtein[np.ix_(test_idx, test_idx)]
        self.cliff_mols_levenshtein_tr = [s for s in self.cliff_mols_levenshtein if s in self.smiles]
        self.cliff_mols_levenshtein_tst = [s for s in self.cliff_mols_levenshtein if s in self.test_smiles]

        # consensus
        self.soft_consensus_tr = self.soft_consensus[np.ix_(train_idx, train_idx)]
        self.soft_consensus_tst = self.soft_consensus[np.ix_(test_idx, test_idx)]
        self.cliff_mols_soft_consensus_tr = [s for s in self.cliff_mols_soft_consensus if s in self.smiles]
        self.cliff_mols_soft_consensus_tst = [s for s in self.cliff_mols_soft_consensus if s in self.test_smiles]

    def get_stats(self):
        """ Calculate various stats on the found activity cliffs"""
        self.stats = {
            "Fold change threshold":
                self.fold_threshold,
            "Structural similarity threshold":
                self.similarity_threshold,
            "n_tanimoto_cliffs":
                0 if self.tanimoto is None else sum(np.triu(self.tanimoto).flatten()),
            "n_scaffold_cliffs":
                0 if self.scaffold is None else sum(np.triu(self.scaffold).flatten()),
            "n_levenstein_cliffs":
                0 if self.levenshtein is None else sum(np.triu(self.levenshtein).flatten()),
            "n_soft_consensus_cliffs":
                0 if self.soft_consensus is None else sum(np.triu(self.soft_consensus).flatten()),
            "n_tanimoto_cliffs_train":
                0 if self.tanimoto_tr is None else sum(np.triu(self.tanimoto_tr).flatten()),
            "n_scaffold_cliffs_train":
                0 if self.scaffold_tr is None else sum(np.triu(self.scaffold_tr).flatten()),
            "n_levenstein_cliffs_train":
                0 if self.levenshtein_tr is None else sum(np.triu(self.levenshtein_tr).flatten()),
            "n_soft_consensus_cliffs_train":
                0 if self.soft_consensus_tr is None else sum(np.triu(self.soft_consensus_tr).flatten()),
            "n_tanimoto_cliffs_test":
                0 if self.tanimoto_tst is None else sum(np.triu(self.tanimoto_tst).flatten()),
            "n_scaffold_cliffs_test":
                0 if self.scaffold_tst is None else sum(np.triu(self.scaffold_tst).flatten()),
            "n_levenstein_cliffs_test":
                0 if self.levenshtein_tst is None else sum(np.triu(self.levenshtein_tst).flatten()),
            "n_soft_consensus_cliffs_test":
                0 if self.soft_consensus_tst is None else sum(np.triu(self.soft_consensus_tst).flatten()),
            "n_compounds":
                len(self.all_smiles),
            "n_compounds_train":
                len(self.all_smiles)-len(self.test_smiles),
            "n_compounds_test":
                len(self.test_smiles),
            "n_tanimoto_cliff_compounds":
                0 if self.cliff_mols_tanimoto is None else len(self.cliff_mols_tanimoto),
            "n_scaffold_cliff_compounds":
                0 if self.cliff_mols_scaffold is None else len(self.cliff_mols_scaffold),
            "n_levenshtein_cliff_compounds":
                0 if self.cliff_mols_levenshtein is None else len(self.cliff_mols_levenshtein),
            "n_soft_consensus_cliff_compounds":
                0 if self.cliff_mols_soft_consensus is None else len(self.cliff_mols_soft_consensus),
            "n_tanimoto_cliff_compounds_train":
                0 if self.cliff_mols_tanimoto_tr is None else len(self.cliff_mols_tanimoto_tr),
            "n_scaffold_cliff_compounds_train":
                0 if self.cliff_mols_scaffold_tr is None else len(self.cliff_mols_scaffold_tr),
            "n_levenshtein_cliff_compounds_train":
                0 if self.cliff_mols_levenshtein_tr is None else len(self.cliff_mols_levenshtein_tr),
            "n_soft_consensus_cliff_compounds_train":
                0 if self.cliff_mols_soft_consensus_tr is None else len(self.cliff_mols_soft_consensus_tr),
            "n_tanimoto_cliff_compounds_test":
                0 if self.cliff_mols_tanimoto_tst is None else len(self.cliff_mols_tanimoto_tst),
            "n_scaffold_cliff_compounds_test":
                0 if self.cliff_mols_scaffold_tst is None else len(self.cliff_mols_scaffold_tst),
            "n_levenshtein_cliff_compounds_test":
                0 if self.cliff_mols_levenshtein_tst is None else len(self.cliff_mols_levenshtein_tst),
            "n_soft_consensus_cliff_compounds_test":
                0 if self.cliff_mols_soft_consensus_tst is None else len(self.cliff_mols_soft_consensus_tst),
        }

    def __repr__(self):
        self.get_stats()
        out = ''
        for i in list(self.stats.keys()):
            if len(self.test_smiles) == 0:  # if no test data is provided, don't show test stats (they will be 0)
                if not i.endswith('train') and not i.endswith('test'):
                    out += f'{i}: {self.stats[i]}\n'
            else:
                out += f'{i}: {self.stats[i]}\n'
        return out

