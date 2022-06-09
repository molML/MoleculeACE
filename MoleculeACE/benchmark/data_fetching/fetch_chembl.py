from chembl_webresource_client.new_client import new_client
import pandas as pd

def retrieve_data(target_id='CHEMBL3979', waitbar=True, endpoints = ["Ki", "Kd"], relations = ["="]):
    """
    Function to retrieve the data from ChEMBL, starting from a given ChEMBL ID (target)
    :param target_id (str): ChEMBL target ID
    :param waitbar (bool): display the wait bar. With Jupyter, this might give some display troubles, so set to False
    :return: df (pd dataframe): contains the raw data retrieved from ChEMBL based on the criteria

    """
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_id)  # only binding assays

    # initialization
    df = pd.DataFrame()

    print("Collecting data from ChEMBL...")

    # checks data and collects them in a dataframe

    try:
        while res:  # runs over the retrieved entries
            entry = res.next()
            # checks whether the compounds satisfy the criteria (endpoints and relation type), if so, stores them
            if entry['standard_type'] in endpoints and entry["relation"] in relations:

                # checks whether some warning flags are present for the activity or the entry
                if entry["data_validity_comment"] is not None:
                    warning_flag = True  # explicit flag in case there are comments on the reliability
                else:
                    warning_flag = False

                # retains only ki/kd and precise values, plus additional information to check for validity and refs
                df = df.append({'smiles': entry["canonical_smiles"], 'standard_type': entry['standard_type'],
                                'value': entry["standard_value"], 'units': entry["standard_units"],
                                'chembl_id': entry["molecule_chembl_id"], 'document_chembl_id': entry["document_chembl_id"],
                                'data_validity_comment': entry["data_validity_comment"],
                                'data_validity_description': entry["data_validity_description"],
                                'activity_comment': entry["activity_comment"], 'warning_flag': warning_flag},
                               ignore_index=True)
    except StopIteration:
        print(str(len(df.index)) + " molecules collected.")

    return df



