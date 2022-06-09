from rdkit import Chem
from rdkit.Chem import AllChem

"""Code to perform molecule standardization, and curation."""


def main(smiles, remove_salts=True, sanitize=True, neutralize=True):
    # initialize the outputs
    salt = []
    failed_sanit = []
    neutralized = []

    # performs the molecule preparation based on the flags
    if remove_salts and smiles is not None:
        if "." in smiles:  # checks if salts
            salt = True
        else:
            salt = False

    if sanitize is True:
        smiles, failed_sanit = sanitize_mol(smiles)

    if neutralize is True and failed_sanit is False:
        smiles, neutralized = neutralize_mol(smiles)

    return smiles, salt, failed_sanit, neutralized


def _InitialiseNeutralisationReactions():
    """ adapted from the rdkit contribution of Hans de Winter """
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


def sanitize_mol(smiles):
    """ Sanitizes a molecule using rdkit """
    # init
    failed_sanit = False

    # == basic checks on SMILES validity
    mol = Chem.MolFromSmiles(smiles)

    # flags: Kekulize, check valencies, set aromaticity, conjugation and hybridization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL

    # check if the conversion to mol was successful, return otherwise
    if mol is None:
        failed_sanit = True
    # sanitization based on the flags (san_opt)
    else:
        sanitize_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanitize_fail:
            failed_sanit = True
            raise ValueError(sanitize_fail)  # returns if failed

    return smiles, failed_sanit


# ====== neutralizes charges based on the patterns specified in _InitialiseNeutralisationReactions
def neutralize_mol(smiles):
    neutralized = False
    mol = Chem.MolFromSmiles(smiles)

    # retrieves the transformations
    transfm = _InitialiseNeutralisationReactions()  # set of transformations

    # applies the transformations
    for i, (reactant, product) in enumerate(transfm):
        while mol.HasSubstructMatch(reactant):
            neutralized = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]

    # converts back the molecule to smiles
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    return smiles, neutralized
