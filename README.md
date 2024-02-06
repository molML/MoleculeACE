![MolDox logo](img/MoleculeACE.png?raw=true "Title")

![repo version](https://img.shields.io/badge/Version-v.%203.0.0-green)
![python version](https://img.shields.io/badge/python-v.3.8-blue)
![license](https://img.shields.io/badge/license-MIT-orange)

Molecule Activity Cliff Estimation (**MoleculeACE**) is a tool for evaluating the predictive performance on activity cliff compounds of machine learning models. 

MoleculeACE can be used to:
1) Analyze and compare the performance on activity cliffs of machine learning methods typically employed in 
QSAR.
2) Identify best practices to enhance a model’s predictivity in the presence of activity cliffs.
3) Design guidelines to consider when developing novel QSAR approaches. 


<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>


<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Benchmark study"> ➤ Benchmark study</a></li>
    <li><a href="#Tool"> ➤ Tool</a></li>
    <li><a href="#Prerequisites"> ➤ Prerequisites</a></li>
    <li>
          <a href="#Installation"> ➤ Installation</a>
          <ul>
            <li><a href="#Pip-installation">Pip installation</a></li>
            <li><a href="#Manual-installation">Manual installation</a></li>
          </ul>
    </li>
    <li>
          <a href="#Getting-started"> ➤ Getting started</a>
          <ul>
            <li><a href="#train-model">Train an out-of-the-box model</a></li>
            <li><a href="#eval-own-model">Evaluate your own model</a></li>
          </ul>
    </li>
    <li><a href="#How-to-cite"> ➤ How to cite</a></li>
    <li><a href="#License"> ➤ Licence</a></li>
  </ol>
</details>




<!-- Benchmark study-->
<h2 id="benchmark-study">Benchmark study</h2>


In a benchmark study we collected and curated bioactivity data on 30 macromolecular targets, which were used to evaluate 
the performance of many machine learning algorithms on activity cliffs. We used classical machine learning methods
combined with common molecular descriptors and neural networks based on unstructured molecular data like molecular 
graphs or SMILES strings.

**Activity cliffs are molecules with small differences in structure but large differences in potency.** Activity cliffs
play an important role in drug discovery, but the bioactivity of activity cliff compounds are notoriously difficult to 
predict. 

![Activity cliff example](img/cliff_example.png?raw=true "activity_cliff_example")
*Example of an activity cliff on the Dopamine D3 receptor, D3R*

<!-- Tool-->
<h2 id="Tool">Tool</h2>


Any regression model can be evaluated on activity cliff performance using MoleculeACE on third party data or the 30
included molecular bioactivity data sets. All 24 machine learning strategies covered in our benchmark study can be used 
out of the box.

![MolDox logo](img/moleculeACE_example.png?raw=true "activity_cliff_example")


<!-- Prerequisites-->
<h2 id="Prerequisites">Prerequisites</h2>

MoleculeACE currently supports Python 3.8. Some required deep learning packages are not included in the pip install. 
- [Tensorflow](https://www.tensorflow.org/) (2.9.0)
- [PyTorch](https://pytorch.org/) (1.11.0)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) (2.0.4)
- [Transformers](https://huggingface.co/docs/transformers/installation) (4.20.1)

<!-- Installation-->
<h2 id="Installation">Installation</h2>

<h3 id="Pip-installation"> Pip installation</h3>
MoleculeACE can be installed as

```pip install MoleculeACE```

<h3 id="Manual-installation"> Manual installation</h3>

```git clone https://github.com/molML/MoleculeACE.git```

```
pip install rdkit-pypi pandas numpy pandas chembl_webresource_client scikit-learn matplotlib tqdm python-Levenshtein
```

<!-- Getting started-->
<h2 id="Getting-started">Getting started</h2>

<h3 id="train-model"> Train an out-of-the-box model on one of the many included datasets</h3>

```python
from MoleculeACE import MPNN, Data, Descriptors, calc_rmse, calc_cliff_rmse, get_benchmark_config

dataset = 'CHEMBL2034_Ki'
descriptor = Descriptors.GRAPH
algorithm = MPNN

# Load data
data = Data(dataset)

# Get the already optimized hyperparameters
hyperparameters = get_benchmark_config(dataset, algorithm, descriptor)

# Featurize SMILES strings with a specific method
data(descriptor)

# Train and a model
model = algorithm(**hyperparameters)
model.train(data.x_train, data.y_train)
y_hat = model.predict(data.x_test)

# Evaluate your model on activity cliff compounds
rmse = calc_rmse(data.y_test, y_hat)
rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=data.y_test, cliff_mols_test=data.cliff_mols_test)

print(f"rmse: {rmse}")
print(f"rmse_cliff: {rmse_cliff}")
```

<h3 id="eval-own-model"> Evaluate the performance of your own model</h3>

```python
from MoleculeACE import calc_rmse, calc_cliff_rmse

# Train your own model
model = ...
y_hat = model.predict(...)

# Evaluate your model on activity cliff compounds
rmse = calc_rmse(y_test, y_hat)
# You need to provide both the predicted and true values of the test set + train labels + the train and test molecules
# Activity cliffs are calculated on the fly
rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=y_test, smiles_test=smiles_test, y_train=y_train, 
                             smiles_train=smiles_train, in_log10=True, similarity=0.9, potency_fold=10)

print(f"rmse: {rmse}")
print(f"rmse_cliff: {rmse_cliff}")
```

<!-- How to cite-->
<h2 id="How-to-cite">How to cite</h2>

Exposing the Limitations of Molecular Machine Learning with Activity Cliffs. Derek van Tilborg, Alisa Alenicheva, and Francesca Grisoni.
Journal of Chemical Information and Modeling, 2022, 62 (23), 5938-5951.
DOI: 10.1021/acs.jcim.2c01073   


<!-- License-->
<h2 id="License">License</h2>

MoleculeACE is under MIT license. For use of specific models, please refer to the model licenses found in the original 
packages.
