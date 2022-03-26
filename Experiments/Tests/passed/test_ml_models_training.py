from tqdm import tqdm

from MoleculeACE.benchmark import load_data, models, evaluation
from MoleculeACE.benchmark.utils.const import Descriptors, DATA_PATH, Algorithms


def main():
    dataset = 'CHEMBL4203_Ki'
    descriptor = Descriptors.ECFP
    algorithms = [Algorithms.RF, Algorithms.GBM, Algorithms.SVM, Algorithms.KNN, Algorithms.MLP]
    data = load_data(dataset, descriptor=descriptor, smiles_colname='smiles',
                     y_colname='exp_mean [nM]', chembl_id_colname='chembl_id',
                     data_root=DATA_PATH, tolog10=True, fold_threshold=12,
                     similarity_threshold=0.80, scale=True, augment_smiles=0)
    print("Test train ml models")
    for algorithm in tqdm(algorithms):
        print(f"Algorithm {algorithm.value}")

        model = models.train_model(data, algorithm=algorithm)
        predictions = model.test_predict()
        results = evaluation.evaluate(data=data, predictions=predictions)
        print(f"Results {results}")
        print()



if __name__ == "__main__":
    main()
