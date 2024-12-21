# Gene-Gene Relationship Modeling Based on Genetic Evidence for Single-Cell RNA-Seq Data Imputation

<p align="center">
    <a href="https://pytorch.org/" alt="PyTorch">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <img src="https://img.shields.io/badge/-NeurIPS_2024-blue" />

The official source code for "Gene-Gene Relationship Modeling Based on Genetic Evidence for Single-Cell RNA-Seq Data Imputation", accepted at NeurIPS 2024.

## Requirements
- Python version : 3.9.16
- Pytorch version : 1.10.0
- scanpy : 1.9.3

## Download data

Create a directory to save the dataset:
```
mkdir dataset
```

You can download the preprocessed data [here](https://www.dropbox.com/sh/eaujyhthxjs0d5g/AADzvVv-h2yYWaoOfs1sybKea?dl=0)
This data was provided by the authors of "Single-cell RNA-seq data imputation using Feature Propagation", and we gratefully acknowledge their contribution.

## How to Run

You can easily reproduce the results using the following commands:
```
git clone https://github.com/daehoum/scCR.git
cd scCR
sh run.sh
```

## Acknowledgements

The implementation would not have been possible without referencing the following repositories:
- [https://github.com/Junseok0207/scFP](https://github.com/Junseok0207/scFP)
- [https://github.com/Junseok0207/scBFP](https://github.com/Junseok0207/scBFP)