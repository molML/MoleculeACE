from tqdm import tqdm

from MoleculeACE.benchmark import load_data
from MoleculeACE.benchmark.utils.const import Descriptors, DATA_PATH


def main():
    datasets = ['CHEMBL1871_Ki', 'CHEMBL4203_Ki', 'CHEMBL2971_Ki']
    descriptors = [d for d in Descriptors]
    print("Test load_data")
    for dataset in datasets:
        print(f"Dataset {dataset}")
        for descriptor in tqdm(descriptors):
            print(f"Dataset {descriptor}")
            data = load_data(dataset, descriptor=descriptor, smiles_colname='smiles',
                             y_colname='exp_mean [nM]', chembl_id_colname='chembl_id',
                             data_root=DATA_PATH, tolog10=True, fold_threshold=12,
                             similarity_threshold=0.80, scale=True, augment_smiles=0)
            print(f"Encoded data {data.x_train[0]}")
            print(f"Train size {len(data.smiles_train)}")
            print(f"Tanimoto Similarity Cliffs  {data.cliffs.tanimoto_sim.shape}")
            print()


if __name__ == "__main__":
    main()
