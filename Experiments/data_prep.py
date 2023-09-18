"""
Author: Derek van Tilborg -- TU/e -- 20-06-2022

Script for processing all raw data used for the benchmark

"""

import pandas as pd
import os
from tqdm import tqdm
from MoleculeACE.benchmark.data_prep import split_data
from MoleculeACE.benchmark.data_fetching import fetch_chembl

SIMILARITY = 0.9
POTENCY_FOLD = 10
TEST_SIZE = 0.2

def main():

    raw_files = [f for f in os.listdir('MoleculeACE/Data/benchmark_data/raw/') if f.endswith('.csv')]
    for file in tqdm(raw_files):

        df_raw = pd.read_csv(f"MoleculeACE/Data/benchmark_data/raw/{file}")
        df_raw = df_raw.dropna()

        se_idx = [i for i, smi in enumerate(df_raw.smiles) if '[Se]' in smi]
        df_raw = df_raw.drop(index=se_idx)

        te_idx = [i for i, smi in enumerate(df_raw.smiles) if '[te]' in smi]
        df_raw = df_raw.drop(index=te_idx)

        raw_smiles = df_raw['smiles'].tolist()
        raw_bioactivity = df_raw['exp_mean [nM]'].tolist()

        df = split_data(raw_smiles, raw_bioactivity, n_clusters=5, test_size=TEST_SIZE, similarity=SIMILARITY,
                        potency_fold=POTENCY_FOLD, remove_stereo=True)

        df.to_csv(f"MoleculeACE/Data/benchmark_data/{file}", index=False)


if __name__ == "__main__":
    main()
