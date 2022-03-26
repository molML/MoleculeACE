"""
Class that holds the results: used for evaluating model performance on activity cliff compounds
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import os
import numpy as np

from MoleculeACE.benchmark.utils.const import Algorithms
from .metrics import calc_rmse, calc_q2f3


class Results:
    def __init__(self, predictions=None, reference=None, y_train=None, data=None,
                 tanimoto_cliff_compounds=None, scaffold_cliff_compounds=None, levenshtein_cliff_compounds=None,
                 soft_consensus_cliff_compounds=None):

        self.predictions = predictions
        self.reference = reference
        self.y_train = y_train

        self.tanimoto_cliff_compounds = tanimoto_cliff_compounds
        self.scaffold_cliff_compounds = scaffold_cliff_compounds
        self.levenshtein_cliff_compounds = levenshtein_cliff_compounds
        self.soft_consensus_cliff_compounds = soft_consensus_cliff_compounds

        self.data = data
        self.rmse = np.inf
        self.q2f3 = 0

        self.tanimoto_cliff_rmse = np.inf
        self.scaffold_cliff_rmse = np.inf
        self.levenshtein_cliff_rmse = np.inf
        self.soft_consensus_cliff_rmse = np.inf

    def calc_rmse(self, reference=None, predictions=None):
        """ Calculate the rmse from two lists of reference and predicted bioactivity"""
        if reference is not None:
            self.reference = reference
        if predictions is not None:
            self.predictions = predictions

        # calculate the rmsd
        self.rmse = calc_rmse(self.reference, self.predictions)

        return self.rmse

    def calc_q2f3(self, reference=None, predictions=None, y_train=None):
        """ Calculates the Q2 F3 score (best according to Todeschini et al. 2016)

        Args:
            reference: (1d array-like shape) true test values (float)
            predictions: (1d array-like shape) predicted test values (float)
            y_train: (1d array-like shape) true train values (float)

        Returns: Q2F3 score
        """
        if reference is not None:
            self.reference = reference
        if predictions is not None:
            self.predictions = predictions
        if y_train is not None:
            self.y_train = y_train

        # calculate the q2f3
        self.q2f3 = calc_q2f3(self.reference, self.predictions, self.y_train)

        return self.q2f3

    def calc_cliff_rmse(self, reference=None, predictions=None, tanimoto_cliff_compounds=None,
                        scaffold_cliff_compounds=None, levenshtein_cliff_compounds=None,
                        soft_consensus_cliff_compounds=None):
        """ Calculate the rmse of only cliff compounds

        Args:
            levenshtein_cliff_compounds: (lst) Binary list of cliff compounds (same length as predictions)
            tanimoto_cliff_compounds: (lst) Binary list of cliff compounds (same length as predictions)
            scaffold_cliff_compounds: (lst) Binary list of cliff compounds (same length as predictions)
            consensus_cliff_compounds: (lst) Binary list of cliff compounds (same length as predictions)
            soft_consensus_cliff_compounds: (lst) Binary list of cliff compounds (same length as predictions)
            reference: (lst) true bioactivity values
            predictions: (lst) predicted bioactivity values
            cliff_compounds: (lst) binary list describing if a compound is a cliff compound (1 == cliff, 0 == no cliff)

        Returns: (float) rmse

        """
        if reference is not None:
            self.reference = reference
        if predictions is not None:
            self.predictions = predictions
        if tanimoto_cliff_compounds is not None:
            self.tanimoto_cliff_compounds = tanimoto_cliff_compounds
        if scaffold_cliff_compounds is not None:
            self.scaffold_cliff_compounds = scaffold_cliff_compounds
        if levenshtein_cliff_compounds is not None:
            self.levenshtein_cliff_compounds = levenshtein_cliff_compounds
        if soft_consensus_cliff_compounds is not None:
            self.soft_consensus_cliff_compounds = soft_consensus_cliff_compounds

        if self.tanimoto_cliff_compounds is not None:
            # Subset only reference and predicted values of the cliff compounds, then calculate cliff rmse
            clf_ref = [self.reference[idx] for idx, clf in enumerate(self.tanimoto_cliff_compounds) if clf == 1]
            clf_prd = [self.predictions[idx] for idx, clf in enumerate(self.tanimoto_cliff_compounds) if clf == 1]
            self.tanimoto_cliff_rmse = calc_rmse(clf_ref, clf_prd)

        if self.scaffold_cliff_compounds is not None:
            # Subset only reference and predicted values of the cliff compounds, then calculate cliff rmse
            clf_ref = [self.reference[idx] for idx, clf in enumerate(self.scaffold_cliff_compounds) if clf == 1]
            clf_prd = [self.predictions[idx] for idx, clf in enumerate(self.scaffold_cliff_compounds) if clf == 1]
            self.scaffold_cliff_rmse = calc_rmse(clf_ref, clf_prd)

        if self.levenshtein_cliff_compounds is not None:
            # Subset only reference and predicted values of the cliff compounds, then calculate cliff rmse
            clf_ref = [self.reference[idx] for idx, clf in enumerate(self.levenshtein_cliff_compounds) if clf == 1]
            clf_prd = [self.predictions[idx] for idx, clf in enumerate(self.levenshtein_cliff_compounds) if clf == 1]
            self.levenshtein_cliff_rmse = calc_rmse(clf_ref, clf_prd)

        if self.soft_consensus_cliff_compounds is not None:
            # Subset only reference and predicted values of the cliff compounds, then calculate cliff rmse
            clf_ref = [self.reference[idx] for idx, clf in enumerate(self.soft_consensus_cliff_compounds) if clf == 1]
            clf_prd = [self.predictions[idx] for idx, clf in enumerate(self.soft_consensus_cliff_compounds) if clf == 1]
            self.soft_consensus_cliff_rmse = calc_rmse(clf_ref, clf_prd)

        return {'tanimoto_cliff_rmse': self.tanimoto_cliff_rmse, 'scaffold_cliff_rmse': self.scaffold_cliff_rmse,
                'levenshtein_cliff_rmse': self.levenshtein_cliff_rmse,
                'soft_consensus_cliff_rmse': self.soft_consensus_cliff_rmse}

    def to_csv(self, filename, algorithm: Algorithms = None):

        # Create output file if it doesnt exist
        if self.data is not None:
            if not os.path.isfile(filename):
                with open(filename, 'w') as f:
                    f.write('dataset,'
                            'algorithm,'
                            'descriptor,'
                            'augmentation,'
                            'rmse,'
                            'cliff_rmse,'
                            'n_compounds,'
                            'n_cliff_compounds,'
                            'n_compounds_train,'
                            'n_cliff_compounds_train,'
                            'n_compounds_test,'
                            'n_cliff_compounds_test\n')

            with open(filename, 'a') as f:
                f.write(f'{self.data.name},'
                        f'{algorithm.value},'
                        f'{self.data.descriptor.value},'
                        f'{self.data.augmentation},'
                        f'{self.rmse},'
                        f'{self.soft_consensus_cliff_rmse},'
                        f'{self.data.cliffs.stats["n_compounds"]},'
                        f'{self.data.cliffs.stats["n_soft_consensus_cliff_compounds"]},'
                        f'{self.data.cliffs.stats["n_compounds_train"]},'
                        f'{self.data.cliffs.stats["n_soft_consensus_cliff_compounds_train"]},'
                        f'{self.data.cliffs.stats["n_compounds_test"]},'
                        f'{self.data.cliffs.stats["n_soft_consensus_cliff_compounds_test"]}\n')

    def __repr__(self):
        return f"RMSE:    {self.rmse:.4f}\n" \
               f"Q2F3:    {self.q2f3:.4f}\n" \
               f"AC-RMSE: {self.soft_consensus_cliff_rmse:.4f}\n"
