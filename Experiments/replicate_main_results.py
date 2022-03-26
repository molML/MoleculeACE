""" Run every model for the MoleculeACE benchmark. Based on the outcome of this script, we based the results of our
paper. In total 720 machine learning models will be trained, so might take a while"""

import os
from MoleculeACE.benchmark import load_data, models, evaluation
from MoleculeACE.benchmark.utils.const import Descriptors, DATA_PATH, Algorithms, datasets, CONFIG_PATH, WORKING_DIR


def make_combinations(datasets, classical=True, graph=True, lstm=True, cnn=True, mlp=True, augment_smiles=10):
    """ Make a list of tuples for every combination we need to run """
    combinations = []
    for dataset in datasets:
        if classical:
            # Start with the classical machine learning methods
            for descr in [Descriptors.ECFP, Descriptors.MACCS, Descriptors.PHYSCHEM, Descriptors.WHIM]:
                for algo in [Algorithms.RF, Algorithms.GBM, Algorithms.SVM, Algorithms.KNN]:
                    combinations.append((dataset, descr, algo, 0))

        # Add the vanilla neural network
        if mlp:
            combinations.append((dataset, Descriptors.ECFP, Algorithms.MLP, 0))

        # Add the graph neural networks
        if graph:
            combinations.append((dataset, Descriptors.CANONICAL_GRAPH, Algorithms.GCN, 0))
            combinations.append((dataset, Descriptors.CANONICAL_GRAPH, Algorithms.MPNN, 0))
            combinations.append((dataset, Descriptors.ATTENTIVE_GRAPH, Algorithms.AFP, 0))
            combinations.append((dataset, Descriptors.CANONICAL_GRAPH, Algorithms.GAT, 0))

        # Add SMILES-based methods (with 10x augmentation)
        if cnn:
            combinations.append((dataset, Descriptors.SMILES, Algorithms.CNN, augment_smiles))
        if lstm:
            combinations.append((dataset, Descriptors.SMILES, Algorithms.LSTM, augment_smiles))

    return combinations


def already_done(output_file, combi):
    """ Check if this model already has been run based on the output file. In case this script crashes (e.g. running out
    of memory, this allows us to continue where we left off"""
    import pandas as pd
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)[['dataset', 'algorithm', 'descriptor', 'augmentation']]
        done = list(df.to_records(index=False))
        done = [''.join([f"{i}" for i in j]) for j in done]
        combi_vals = (combi[0], combi[2].value, combi[1].value, combi[3])
        combi_vals = ''.join([f"{i}" for i in combi_vals])
        return combi_vals in done
    else:
        return False


def find_config(dataset, algo, descr):
    """ Create the name of the config file for this combination """
    return os.path.join(WORKING_DIR, 'configures', 'benchmark', dataset, f"{algo.value}_{descr.value}.yml")


def run_all_combinations(combinations, output_file: str, similarity_threshold: float = 0.9, fold_threshold: int = 10,
                         skip_done: bool = True, save_models: bool = True, save_dir: str = WORKING_DIR):
    """ Train all models

    Args:
        combinations: (tup) (dataset (str), Descriptor, Algorithm, augment (int))
        output_file: (str) path to results .csv
        similarity_threshold: (flt) value between 0 and 1 to determine Activity cliffs
        fold_threshold: (int) potency difference threshold for Activity cliffs. 10 means a 10x difference
        skip_done: (bool) skip combinations that you already did based on the output file
        save_models: (bool) Save models after training them

    Returns:

    """

    failed_combis = []
    prev_descr, prev_augment, prev_dataset = '', '', ''
    for idx, combi in enumerate(combinations):

        # Loop through every combination
        dataset, descr, algo, augment = combi

        do = True
        if skip_done:
            if already_done(output_file, combi):
                do = False

        if do:  # Check if you already ran this combi (helps if something crashes while running 690 models)
            print(f" -- ({idx + 1}/{len(combinations)})  {dataset}, {descr}, {algo}, Augment: {augment}")
            try:
                # If there is a change in descriptor or augmentation, redo the data
                if descr != prev_descr or dataset != prev_dataset:
                    data = load_data(dataset, descriptor=descr, data_root=DATA_PATH, tolog10=True,
                                     fold_threshold=fold_threshold, similarity_threshold=similarity_threshold,
                                     scale=True, augment_smiles=augment)


                # Update some variables for this combination
                prev_descr, prev_augment, prev_dataset = descr, augment, dataset

                # Determine the file extension of the model save file (we use .h5 for tensorflow stuff, it is faster)
                file_extension = 'h5' if algo in [Algorithms.CNN, Algorithms.MLP, Algorithms.LSTM] else 'pkl'

                os.makedirs(os.path.join(save_dir, 'pretrained_models', dataset), exist_ok=True)
                modelfile = os.path.join(save_dir, 'pretrained_models', dataset,
                                         f"{algo.value}_{descr.value}_{augment}.{file_extension}")

                # Find the config file for this combinations. If we can't find it, we will optimize hyperparameters
                # during training and save the best config settings
                config_file = find_config(dataset, algo, descr)

                # If you already trained this model, try to load it. Else train it.
                if os.path.exists(modelfile):
                    try:
                        model = models.load_model(data, algo, modelfile)
                    except:
                        model = models.train_model(data, algorithm=algo, config_file=config_file)
                else:
                    # Train the model
                    model = models.train_model(data, algorithm=algo, config_file=config_file)

                # Save models
                if save_models:
                    try:
                        model.save_model(modelfile)
                    except:
                        print(f" ---- Could not save {dataset}, {descr}, {algo}, {augment}")

                # Predict and evaluate test data, then write results to output file
                predictions = model.test_predict()
                results = evaluation.evaluate(data=data, predictions=predictions)

                results.to_csv(filename=output_file, algorithm=algo)

            except:
                print(f'something went wrong with {dataset}, {descr}, {algo}, {augment}')
                failed_combis.append(combi)

    return failed_combis


# Run all models
if __name__ == '__main__':

    # Settings for running everything
    output_file = os.path.join(WORKING_DIR, 'results', 'Benchmark_results.csv')

    # Create all combinations we want to run
    combinations = make_combinations(datasets, classical=True, graph=True, cnn=True, lstm=True, mlp=True,
                                     augment_smiles=10)

    # Train all models. Change save_dir to something else if you want to change where models are being saved
    run_all_combinations(combinations, output_file, similarity_threshold=0.9, fold_threshold=10, skip_done=True,
                         save_models=True, save_dir=WORKING_DIR)
