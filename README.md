[![Python](https://img.shields.io/badge/Python-3.8.10-%233776AB?logo=Python)](https://www.python.org/)

# FairUP
The official implmentation of "FairUP: a Framework for Fairness Analysis of Graph Neural Network-Based User Profiling Models"

![fairup_architecture](https://user-images.githubusercontent.com/45569039/220563974-905756a9-eb1f-4140-9a17-73b8c3a52529.png)

The framework currently supports these GNN models:
- [FairGNN](https://arxiv.org/abs/2009.01454)
- [RHGN](https://arxiv.org/abs/2110.07181)
- [CatGCN](https://arxiv.org/abs/2009.05303)
## Abstract
Modern user profiling approaches capture different forms of interactions with the data, from user-item to user-user relationships. Graph Neural Networks (GNNs) have become a natural way to model these behaviours and build efficient and effective user profiles. However, each GNN-based user profiling approach has its own way of processing information, thus creating heterogeneity that does not favour the benchmarking of these techniques. To overcome this issue, we present FairUP, a framework that standardises the input needed to run three state-of-the-art GNN-based models for user profiling tasks. Moreover, given the importance that algorithmic fairness is getting in the evaluation of machine learning systems, FairUP includes two additional components to (1) analyse pre-processing and post-processing fairness and (2) mitigate the potential presence of unfairness in the original datasets through three pre-processing debiasing techniques. The framework, while extensible in multiple directions, in its first version, allow the user to conduct experiments on four real-world datasets.

# Description
**FairUP** is a standardised framework that empowers researchers and practitioners to simultaneously analyse state-of-the-art Graph Neural Network-based models for user profiling task, in terms of classification performance and fairness metrics scores.

The framework, whose architecture is shown below, presents several components, which allow end-users to:
* compute the fairness of the input dataset by means of a pre-processing fairness metric, i.e. *disparate impact*;
* mitigate the unfairness of the dataset, if needed, by applying different debiasing methods, i.e. *sampling*, *reweighting* and *disparate impact remover*; 
* standardise the input (a graph in Neo4J or NetworkX format) for each of the included GNNs;
* train one or more GNN models, specifying the parameters for each of them;
* evaluate post-hoc fairness by exploiting four metrics, i.e. *statistical parity*, *equal opportunity*, *overall accuracy equality*, *treatment equality*.


##
## Requirements
The code has been executed under Python 3.8.1, with the dependencies listed below.

```
dgl==0.6.1
dgl_cu113==0.7.2
fasttext==0.9.2
fitlog==0.9.13
hickle==4.0.4
matplotlib==3.5.1
metis==0.2a5
networkx==2.6.3
numpy==1.22.0
pandas==1.3.5
scikit_learn==1.0.2
scipy==1.7.3
texttable==1.6.4
torch==1.10.1+cu113
torch_geometric==2.0.3
torch_scatter==2.0.9
tqdm==4.62.3
```
Notes:
* the file `requirements.txt` installs all dependencies for both models;
* the dependencies including `cu113` are meant to run on **CUDA 11.3** (install the correct package based on your version of CUDA).

## Web application
Available [here](https://mohamedabdelrazek9-fairup-homepage-gv365a.streamlit.app/)

## Later updates
- Integration of new GNN models.
- Integration of new datasets.

## Contact
- **M.Sc. Erasmo Purificato: erasmo.purificato@ovgu.com**
- **M.Sc. Mohamed Abdelrazek: mimo.1998@hotmail.com**
