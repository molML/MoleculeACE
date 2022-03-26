"""
Some random functions used for graph featurization
Alisa Alenicheva, Jetbrains research, Februari 2022
"""

from dgllife.utils import CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, CanonicalBondFeaturizer, \
    AttentiveFPBondFeaturizer

from MoleculeACE.benchmark.utils import Descriptors


def get_atom_feat_size(descriptor):
    if descriptor == Descriptors.CANONICAL_GRAPH:
        featurizer = CanonicalAtomFeaturizer()
    elif descriptor == Descriptors.ATTENTIVE_GRAPH:
        featurizer = AttentiveFPAtomFeaturizer()
    elif descriptor == Descriptors.PRETRAINED_GRAPH:
        return None
    else:
        raise ValueError('Unexpected descriptor: {}'.format(descriptor.value))
    return featurizer.feat_size()


def get_bond_feat_size(descriptor):
    if descriptor == Descriptors.CANONICAL_GRAPH:
        featurizer = CanonicalBondFeaturizer(self_loop=True)
    elif descriptor == Descriptors.ATTENTIVE_GRAPH:
        featurizer = AttentiveFPBondFeaturizer(self_loop=True)
    else:
        raise ValueError('Unexpected descriptor: {}'.format(descriptor.value))
    return featurizer.feat_size()
