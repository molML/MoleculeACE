"""
Class to evaluate a models performance on activity cliff compounds
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

from .results import Results


def evaluate(predictions, data, verbose=True, tanimoto=True, scaffold=True, levenshtein=True, soft_consensus=True):
    """ Evaluate the perdictions results: Calculates rmse, q2f3 and cliff-rmse

    Args:
        predictions: (lst) predicted bioactivity values
        data: (MoleculeACE.benchmark.Data) Data object
        verbose: (bool) printout results?
        tanimoto: (bool) calculate tanimoto cliff rmse
        scaffold: (bool) calculate scaffold cliff rmse
        levenshtein: (bool) calculate levenshtein cliff rmse
        soft_consensus: (bool) calculate soft consensus cliff rmse

    Returns: MoleculeACE.benchmark.Results object

    """

    y_test = data.y_test
    y_train = data.y_train
    if data.cliffs is None:
        print("Data object has no pre-computed activity cliffs\nLooking for them now with default thresholds...")
        data.get_cliffs()

    if tanimoto:
        tanimoto_cliff_cpds = [1 if i in data.cliffs.cliff_mols_tanimoto else 0 for i in data.smiles_test]
    else:
        tanimoto_cliff_cpds = None
    if scaffold:
        scaffold_cliff_cpds = [1 if i in data.cliffs.cliff_mols_scaffold else 0 for i in data.smiles_test]
    else:
        scaffold_cliff_cpds = None
    if levenshtein:
        levenshtein_cliff_cpds = [1 if i in data.cliffs.cliff_mols_levenshtein else 0 for i in data.smiles_test]
    else:
        levenshtein_cliff_cpds = None
    if soft_consensus:
        soft_consensus_cliff_cpds = [1 if i in data.cliffs.cliff_mols_soft_consensus else 0 for i in data.smiles_test]
    else:
        soft_consensus_cliff_cpds = None

    results = Results(predictions=predictions, reference=y_test, y_train=y_train,
                      tanimoto_cliff_compounds=tanimoto_cliff_cpds,
                      scaffold_cliff_compounds=scaffold_cliff_cpds,
                      levenshtein_cliff_compounds=levenshtein_cliff_cpds,
                      soft_consensus_cliff_compounds=soft_consensus_cliff_cpds,
                      data=data)

    # Calculate the rmse
    results.calc_rmse()
    # Calculate the q2f3 if the train labels are available
    results.calc_q2f3()
    # Calculate the rmse for cliff compounds if they are known
    results.calc_cliff_rmse()

    if verbose:
        print(results)

    return results