

from skopt import gp_minimize
from skopt.space.space import Categorical
from skopt.utils import use_named_args
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from MoleculeACE.benchmark.utils import calc_rmse
from tqdm import tqdm
from typing import Dict, List, Union
from transformers.tokenization_utils_base import BatchEncoding
import pandas as pd
from MoleculeACE.benchmark.utils import write_config


class BayesianOptimization:
    def __init__(self, model):
        self.best_rmse = 100
        self.history = []
        self.model = model
        self.results = None

    def best_param(self):
        if len(self.history) is not None:
            return self.history[[i[0] for i in self.history].index(min([i[0] for i in self.history]))][1]


    def optimize(self, x, y, dimensions: Dict[str, List[Union[float, str, int]]], x_val=None, y_val=None,
                 n_folds: int = 5, early_stopping: int = 10, n_calls: int = 50, min_init_points: int = 10):

        # Prevent too mant calls if there aren't as many possible hyperparameter combi's as calls (10 in the min calls)
        dimensions = {k: [v] if type(v) is not list else v for k, v in dimensions.items()}
        combinations = count_hyperparam_combinations(dimensions)
        if len(combinations) < n_calls:
            n_calls = len(combinations)
        if len(combinations) < min_init_points:
            min_init_points = len(combinations)

        dimensions = dict_to_search_space(dimensions)

        # Objective function for Bayesian optimization
        @use_named_args(dimensions=dimensions)
        def objective(**hyperparameters):

            epochs = None
            try:
                print(f"Current hyperparameters: {hyperparameters}")
                if n_folds > 0:
                    rmse, epochs = cross_validate(self.model, x, y, n_folds=n_folds, early_stopping=early_stopping,
                                                  **hyperparameters)
                else:
                    f = self.model(**hyperparameters)
                    f.train(x, y, x_val, y_val)
                    epochs = f.epoch
                    pred = f.predict(x_val)
                    rmse = calc_rmse(pred, y_val)

                    del f
                    torch.cuda.empty_cache()

            except:  # If this combination of hyperparameters fails, we use a dummy rmse that is worse than the best
                print(">>  Failed")
                rmse = self.best_rmse + 1

            if rmse < self.best_rmse:
                self.best_rmse = rmse

            if epochs is not None:
                hyperparameters['epochs'] = epochs

            self.history.append((rmse, hyperparameters))

            return rmse

        # Perform Bayesian hyperparameter optimization with 5-fold cross-validation
        self.results = gp_minimize(func=objective,
                                   dimensions=dimensions,
                                   acq_func='EI',  # expected improvement
                                   n_initial_points=min_init_points,
                                   n_calls=n_calls,
                                   verbose=True)

    def plot_progress(self):
        import matplotlib.pyplot as plt
        live = [i[0] for i in self.history]
        minimal = sorted([i[0] for i in self.history], reverse=True)
        tries = list(range(len(self.history)))

        plt.figure()
        plt.plot(tries, live, label='history')
        plt.plot(tries, minimal, label='best')
        plt.xlabel("Optimization attempts")
        plt.ylabel("RMSE")
        plt.legend(loc="upper right")
        plt.show()

    def history_to_csv(self, filename: str):
        results = []
        for i in self.history:
            d = {'rmse': i[0]}
            d.update(i[1])
            results.append(d)

        results = {k: [dic[k] for dic in results] for k in results[0]}

        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

    def save_config(self, filename: str):
        write_config(filename, self.best_param())


def dict_to_search_space(hyperparams: Dict[str, List[Union[float, str, int]]]):
    return [Categorical(categories=list(v), name=k) for k, v in hyperparams.items()]


def count_hyperparam_combinations(hyperparameters: Dict[str, List[Union[float, str, int]]]):
    from itertools import product
    return list(product(*[v for k, v in hyperparameters.items()]))

# model = Transformer
# x = data.x_train
# y = data.y_train
# early_stopping = 10
# n_folds = 5
# hyperparameters = {'epochs': 2, 'lr': 0.0005}
def cross_validate(model, x, y, n_folds: int = 5, early_stopping: int = 10, **hyperparameters):
    ss = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    cutoff = np.median(y)
    labels = [0 if i < cutoff else 1 for i in y]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    rmse_scores = []
    epochs = []
    for split in tqdm(splits):

        # Convert numpy types to regular python type (this bug cost me ages)
        hyperparameters = {k: v.item() if isinstance(v, np.generic) else v for k, v in hyperparameters.items()}

        f = model(**hyperparameters)

        if type(x) is BatchEncoding:
            x_tr_fold = {'input_ids': x['input_ids'][split['train_idx']],
                         'attention_mask': x['attention_mask'][split['train_idx']]}
            x_val_fold = {'input_ids': x['input_ids'][split['val_idx']],
                         'attention_mask': x['attention_mask'][split['val_idx']]}
        else:
            x_tr_fold = [x[i] for i in split['train_idx']] if type(x) is list else x[split['train_idx']]
            x_val_fold = [x[i] for i in split['val_idx']] if type(x) is list else x[split['val_idx']]

        y_tr_fold = [y[i] for i in split['train_idx']] if type(y) is list else y[split['train_idx']]
        y_val_fold = [y[i] for i in split['val_idx']] if type(y) is list else y[split['val_idx']]

        f.train(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, early_stopping)
        pred = f.predict(x_val_fold)

        rmse_scores.append(calc_rmse(pred, y_val_fold))

        if f.epoch is not None:
            if 'epochs' in hyperparameters:
                if hyperparameters['epochs'] > f.epoch:
                    f.epoch = f.epoch - early_stopping
            epochs.append(f.epoch)

        del f
        torch.cuda.empty_cache()

    if len(epochs) > 0:
        epochs = int(sum(epochs)/len(epochs))
    else:
        epochs = None

    return sum(rmse_scores)/len(rmse_scores), epochs


#
# descriptor = Descriptors.ECFP
# algo = MLP
#
# data = Data(f"CHEMBL2047_EC50")
# data(descriptor)
#
# config = get_config(os.path.join(CONFIG_PATH, 'default', f"{algo.__name__}.yml"))
# # config['epochs'] = 2
#
# opt = BayesianOptimization(algo)
# opt.optimize(data.x_train, data.y_train, config, n_calls=20, min_init_points=5)
#
# # 0.6295
# # 0.8565
#
# # model = algo(**{'epochs': 2, 'lr': 0.0005, 'max_len_model': 200, 'freeze_core': True})
# # model.train(data.x_train, data.y_train)
# # model.predict(data.x_test)
#
#
# opt.history
# opt.plot_progress()
#
# hyperparameters = {'epochs': 10, 'node_hidden': 64, 'transformer_hidden': 256, 'n_conv_layers': 2, 'n_fc_layers': 3, 'fc_hidden': 128, 'dropout': 0.0, 'lr': 5e-06}
#
# model = algo(**hparm)
# model.train(data.x_train, data.y_train, data.x_train, data.y_train, 10)
# model.predict(data.x_test)
#
#
# cross_validate(algo, data.x_train, data.y_train, n_folds=5, early_stopping=10, **hyperparameters)
#
# model = algo()
# model.model()
#
# hyperparameters = {'epochs': 10, 'hidden_channels': 32, 'num_timesteps': 4, 'num_layers': 4, 'dropout': 0.1, 'lr': 0.0005}
#
# model = GAT
# x = data.x_train
# y = data.y_train
#
# cross_validate(AFP, data.x_train, data.y_train, **hyperparameters)
#
# n_folds, early_stopping = 5, 10
#
#
# modl.train(data.x_train, data.y_train, epochs=2)
#
# modl.predict(data.x_test)
#
#
# converted_value = getattr(value, "tolist", lambda: value)()
#
# opt.history
# opt.plot_progress()
# opt.history_to_csv('test_run_MLP_ECFP.csv')
# opt.save_config('test_config_MLP_ECFP.yml')
#
# best_config = get_config('test_config_MLP_ECFP.yml')
#
#
#
#
#
#
#
#
#
# opt.history
# opt.best_param()
# opt.plot_progress()
# opt.results
# results = opt.history
#
#
# dimensions = {'hidden': [64, 128, 256, 512, 1024],
#               'n_layers': [1, 2, 3, 4, 5],
#               'dropout': [0, 0.1, 0.2],
#               'lr': [0.005, 0.0005, 0.00005, 0.000005]}
#
# count_hyperparam_combinations({'lr': [0.005, 0.0005, 0.00005, 0.000005]})
#
#
# # dimensions = [Categorical(categories=[1, 2, 3, 4, 5], name='n_layers'),
# #              Categorical(categories=[64, 128, 256 ,512, 1024], name='hidden'),
# #              Categorical(categories=[0, 0.1, 0.2], name='dropout'),
# #              Categorical(categories=[0.005, 0.0005, 0.00005, 0.000005], name='lr')]
#
# rmse = cross_validate(MLP, x, y)
#
#
# calc_rmse(y_val_fold, torch.squeeze(pred))
# calc_rmse(pred, y_val_fold)
#
# torch.squeeze(pred)
#
#
#
# # All hyperparameters that will be explored
# dimensions = [Categorical(categories=[32, 64], name='node_hidden'),
#               Categorical(categories=[32, 64], name='edge_hidden'),
#               Integer(low=2, high=5, name='message_steps'),
#               Categorical(categories=[0, 0.125, 0.25, 0.5], name='dropout'),
#               Categorical(categories=[8], name='transformer_heads'),
#               Categorical(categories=[64, 128, 256], name='transformer_hidden'),
#               Real(low=5e-5, high=5e-2, prior='log-uniform', name='lr'),
#               Real(low=1e-4, high=1e-1, prior='log-uniform', name='l2_lambda')]
#
#
#
# import pandas as pd
# from MoleculeACE.benchmark.featurization import Featurizer
# from MoleculeACE.models.gcn import GCN
# from MoleculeACE.models.attentivefp import AFP
# from MoleculeACE.models.ml import RF
# from MoleculeACE.models.mlp import MLP
#
# df = pd.read_csv(f"MoleculeACE/Data/benchmark_data/CHEMBL233_Ki.csv")
# smiles = df['smiles'].tolist()
# y = df['y'].tolist()
# feat = Featurizer()
# x = feat.ecfp(smiles)
#
# xx = feat.maccs(smiles)
# from numpy.typing import ArrayLike
#
# type(xx) is np.ndarray
#
# model = GCN()


