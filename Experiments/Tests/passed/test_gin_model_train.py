import os

from tqdm import tqdm

from MoleculeACE.benchmark import load_data, models, evaluation
from MoleculeACE.benchmark.utils.const import Descriptors, DATA_PATH, Algorithms, CONFIG_PATH

def main():
    dataset = 'CHEMBL4203_Ki'
    descriptors_algorithms = {
        Descriptors.ATTENTIVE_GRAPH: [Algorithms.GIN_INFOMAX]}
    print("Test load configs for gin models")
    for descriptor, algorithms in tqdm(descriptors_algorithms.items()):
        data = load_data(dataset, descriptor=descriptor, smiles_colname='smiles',
                         y_colname='exp_mean [nM]', chembl_id_colname='chembl_id',
                         data_root=DATA_PATH, tolog10=True, fold_threshold=12,
                         similarity_threshold=0.80, scale=True, augment_smiles=0)
        print(f"Descriptor {descriptor.value}")
        for algorithm in tqdm(algorithms):
            print(f"Algorithm {algorithm.value}")
            config_file = os.path.join(CONFIG_PATH, dataset, f"{algorithm.value}_{descriptor.value}.yml")
            model = models.train_model(data, algorithm=algorithm, config_file=config_file)
            predictions = model.test_predict()
            results = evaluation.evaluate(data=data, predictions=predictions)
            print(f"Results {results}")
            print()


if __name__ == "__main__":
    main()
