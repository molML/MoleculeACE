"""
Functions to calculate drug-like physchem descriptors and WHIM descriptors
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

from MoleculeACE.benchmark.utils.const import RANDOM_SEED


def drug_like_descriptor(m):
    """
    Calculate drug-like physchem descriptors from a rdkit mol object.

    descriptors were inspired by:
    https://sharifsuliman1.medium.com/understanding-drug-likeness-filters-with-rdkit-and-exploring-the-withdrawn-database-ebd6b8b2921e
    """
    from rdkit.Chem import Descriptors

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

    descr = np.array([weight, logp, h_bond_donor, h_bond_acceptors, rotatable_bonds, atoms, heavy_atoms,
                      molar_refractivity, topological_polar_surface_area, formal_charge, rings])
    return descr


def whim_descriptor(m):
    """ Compute a WHIM descriptor from a RDkit molecule object"""
    from rdkit.Chem import rdMolDescriptors
    # Add hydrogens
    mh = Chem.AddHs(m)
    # Use distance geometry to obtain initial coordinates for a molecule
    AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=True, randomSeed=RANDOM_SEED,
                          clearConfs=True, maxAttempts=5)

    AllChem.MMFFOptimizeMolecule(mh, maxIters=1000, mmffVariant='MMFF94')

    # calculate WHIM 3D descriptor
    whim = rdMolDescriptors.CalcWHIM(mh)

    return np.array(whim)
