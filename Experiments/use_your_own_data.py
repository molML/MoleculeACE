"""
This is an example script showing how to process your own data. You need to provide SMILES and their bioactivity.
"""

from MoleculeACE.models import RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN, LSTM, Transformer
from MoleculeACE.benchmark.const import Descriptors
from MoleculeACE.benchmark.data_prep import process_data
from MoleculeACE.benchmark.utils import Data, calc_rmse, calc_cliff_rmse
import os

# Setup some variables
dataset = os.path.join(utils.WORKING_DIR, 'benchmark_data', 'raw', 'CHEMBL234_Ki.csv')
descriptor = Descriptors.ECFP
algorithm = GBM


def main():

    your_own_data = pd.read_csv(dataset)

    # Process your data data. You'll have to provide both the SMILES and the corresponding label
    df = process_data(your_own_data['smiles'], your_own_data['exp_mean [nM]'], in_log10=False, n_clusters=5,
                      test_size=0.2, similarity=0.9, potency_fold=10, remove_stereo=True)

    # Load processed data into Data class, which manages train/test splits, molecule featurization, and cliff stuff
    data = Data(df)

    # Featurize SMILES strings with a specific method
    data(descriptor)

    # Train a model. You can also implement your own model here.
    model = algorithm()
    model.train(data.x_train, data.y_train)
    y_hat = model.predict(data.x_test)

    # Evaluate your model on activity cliff compounds
    rmse = calc_rmse(data.y_test, y_hat)
    # If you want, you can provide your own cliff molecules -- according to your definition -- as cliff_mols_test
    rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=data.y_test, cliff_mols_test=data.cliff_mols_test)

    print(f"rmse: {rmse}\nrmse_cliff: {rmse_cliff}")


if __name__ == '__main__':
    main()
