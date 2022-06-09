# main code to wrap up all the steps from data fetching to curation
import pandas as pd
from MoleculeACE.benchmark.const import WORKING_DIR


def main(chembl_targetid='CHEMBL2047', bar=True, endpoints=["EC50"], assign_class=True, thr_class=100,
         working_dir: str = WORKING_DIR, rm_high_std=True):
    from benchmark.data_fetching import fetch_chembl
    import os
    from benchmark.data_fetching.exp_property_curation import curate_dataframe as curate_exp

    # fetches the data and saves
    table = fetch_chembl.retrieve_data(target_id=chembl_targetid, waitbar=bar, endpoints=endpoints)
    filename = os.path.join(working_dir, 'data_fetching', 'data_raw', chembl_targetid + '_raw_all.csv')
    table.to_csv(filename)

    # performs structure curation and saves
    print('Curating the structures...')
    table_curated = curate_struct(table)
    filename = os.path.join(working_dir, 'data_fetching', 'data_raw', chembl_targetid + '_raw_structure_filtered.csv')
    table_curated.to_csv(filename)

    # experimental property curation and saves
    print('Curating the experimental properties...')
    table_final = curate_exp(table_curated, rm_high_std=rm_high_std)
    filename = os.path.join(working_dir, 'data_fetching', 'data_curated', chembl_targetid + '_data_curated.csv')
    table_final.to_csv(filename)

    # and we're done :)
    print('...Done.')
    n_comp = len(table_final)
    print('Total number of compounds: ' + str(n_comp))

    # assigns a class based on the specified inputs
    if assign_class:
        table_class = class_vector(table_final, thr_class)

        # some printing, just for fun
        n_a = sum(table_class['class'])
        print(' - active compounds: ' + str(n_a))
        print(' - inactive compounds: ' + str(n_comp - n_a))
        print(' - % active compounds: ' + str(round(n_a / n_comp * 100, 2)))

    else:
        table_class = table_final.copy()

    filename = os.path.join(working_dir, 'data_fetching', 'data_final', chembl_targetid + '_data_final.csv')
    table_class.to_csv(filename)

    return table_class


def curate_struct(table):
    from benchmark.data_fetching.molecule_preparation import main as prepare_structures
    table_curated = table.copy()  # copy for further editing

    # structure curation
    for index, row in table_curated.iterrows():
        smiles = row['smiles']
        if smiles is not None:
            smiles, salt, failed_sanit, neutralized = prepare_structures(smiles, remove_salts=True, sanitize=True,
                                                                         neutralize=True)
        else:
            failed_sanit = True
            salt = False
            neutralized = True
            smiles = 'missing'

        table_curated.loc[index, 'IsSalt'] = salt
        table_curated.loc[index, 'FailedSanit'] = failed_sanit
        table_curated.loc[index, 'Neutralized'] = neutralized
        table_curated.loc[index, 'CuratedSmiles'] = smiles

    # "Cleans" the new table (salts & failures in sanitization)
    # removes salts
    table_curated.drop(table_curated.loc[table_curated['IsSalt'] == True].index, inplace=True)
    # removes failed sanitization
    table_curated.drop(table_curated.loc[table_curated['FailedSanit'] == True].index, inplace=True)
    # removes entries with warning flags
    table_curated.drop(table_curated.loc[table_curated['warning_flag'] == True].index, inplace=True)

    return table_curated


def class_vector(table_final, thr_class):
    smiles = pd.DataFrame(table_final['smiles'])
    table_class = pd.DataFrame(smiles)
    table_class['chembl_id'] = table_final['chembl_id']
    table_class['exp_mean [nM]'] = table_final['exp_mean']
    table_class['class'] = table_final.exp_mean < thr_class

    return table_class
