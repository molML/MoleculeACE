### New release coming mid June! This will support Transformers, include an overhaul of all graph neural network models and, the overall structure of the tool will be much more modular and easier to use.

![MolDox logo](img/MoleculeACE.png?raw=true "Title")
***

![repo version](https://img.shields.io/badge/Version-v.%201.0-green)
![python version](https://img.shields.io/badge/python-v.3.8-blue)
![license](https://img.shields.io/badge/license-MIT-orange)

Molecule Activity Cliff Estimation (**MoleculeACE**) is a tool for evaluating the predictive performance on activity cliff compounds of machine learning models. 

MoleculeACE can be used to:
1) Analyze and compare the performance on activity cliffs of machine learning methods typically employed in 
QSAR.
2) Identify best practices to enhance a model’s predictivity in the presence of activity cliffs.
3) Design guidelines to consider when developing novel QSAR approaches. 


## Benchmark study
***

In a benchmark study we collected and curated bioactivity data on 30 macromolecular targets, which were used to evaluate 
the performance of many machine learning algorithms on activity cliffs. We used classical machine learning methods
combined with common molecular descriptors and neural networks based on unstructured molecular data like molecular 
graphs or SMILES strings.

**Activity cliffs are molecules with small differences in structure but large differences in potency.** Activity cliffs
play an important role in drug discovery, but the bioactivity of activity cliff compounds are notoriously difficult to 
predict. 

![Activity cliff example](img/cliff_example.png?raw=true "activity_cliff_example")
*Example of an activity cliff on the Dopamine D3 receptor, D3R*

## Tool
***

Any regression model can be evaluated on activity cliff performance using MoleculeACE on third party data or the 30
included molecular bioactivity data sets. All 23 machine learning strategies covered in our benchmark study can be used 
out of the box.

![MolDox logo](img/moleculeACE_example.png?raw=true "activity_cliff_example")


## Requirements
***
MoleculeACE currently supports Python 3.8
- [tensorflow](https://www.tensorflow.org/)
- [pytorch](https://pytorch.org/)
- [dgl](https://www.dgl.ai/) 
- [dgllife](https://lifesci.dgl.ai/)

## Installation
***
MoleculeACE can be installed as

```pip install MoleculeACE```

### Manual installation
```git clone https://github.com/derekvantilborg/MoleculeACE```

```
pip install rdkit-pypi pandas numpy pandas chembl_webresource_client scikit-learn matplotlib tqdm progress python-Levenshtein
```

### Getting started
***

#### Run an out-of-the-box model on one of the many included datasets

```python
from MoleculeACE.benchmark import load_data, models, evaluation, utils

# Define which dataaset, descriptor, and algorithm to use
dataset = 'CHEMBL287_Ki'
descriptor = utils.Descriptors.CANONICAL_GRAPH
algorithm = utils.Algorithms.MPNN

# Load data
data = load_data(dataset, descriptor=descriptor)

# Train and a model, if config_file = None, hyperparameter optimization is performed
model = models.train_model(data, algorithm=algorithm, config_file=None)
predictions = model.test_predict()

# Evaluate your model on activity cliff compounds
results = evaluation.evaluate(data=data, predictions=predictions)
```


#### Evaluate your own data or model

```python
from MoleculeACE.benchmark import load_data, models, evaluation, utils, process_data

# Setup some variables
dataset = 'path/to/your_own_data.csv'
descriptor = utils.Descriptors.ECFP
algorithm = utils.Algorithms.GBM

    # Process your data data
process_data(dataset, smiles_colname='smiles', y_colname='exp_mean [nM]', test_size=0.2,
             fold_threshold=10, similarity_threshold=0.9)

# Load data
data = load_data(dataset, descriptor=descriptor, tolog10=True)

# Train and optimize a model. You can also implement your own model here
model = models.train_model(data, algorithm=algorithm, config_file=None)
predictions = model.test_predict()

# Evaluate your model on activity cliff compounds
results = evaluation.evaluate(data=data, predictions=predictions)
```

## How to cite
***
van Tilborg et al. (2022). Exposing the limitations of molecular machine learning with activity cliffs. ChemRxiv.

## License
***
MoleculeACE is under MIT license. For use of specific models, please refer to the model licenses found in the original 
packages.
