# MCoMoE: Structure-Enhanced Bidirectional Collaborative Attention for Dynamically Interpretable Protein-RNA Interactions
A structurally enhanced and dynamically interpretable framework that enables bidirectional information interaction between sequence and structure for protein-RNA interaction  prediction.
## Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Data Availability](#data-availability)
- [License](#license)

- ## Overview
We propose MCoMoE, a structure-enhanced RNA model with bidirectional sequence-structure attention for multi-feature interaction. It leverages a multi-scale convolution module to capture in vivo structural cues, and uses adaptive gating and a MoE classifier to improve RBP binding prediction.
Evaluated on K562 and HepG2 cell line datasets against five mainstream baselines, MCoMoE achieves state-of-the-art performance in both static and dynamic RBP prediction. It reliably captures conserved binding motifs across diverse proteins and cell types.
In-depth interpretability analyses verify the model’s biological plausibility. Cross-protein validation highlights key regulatory sites in DDX3X. We further analyze functional SNVs on 3'UTR sequences and the leukemia-associated STAT3 transcript, revealing how genetic variants alter RBP binding and regulatory functions.
<img width="865" height="510" alt="image" src="https://github.com/user-attachments/assets/4043549e-9352-46fb-9106-6cce480e9453" />

## System Requirements
** MCoMoE mainly depends on the Python scientific stack
```bash
python = 3.8
numpy = 1.24.2
torch = 2.0.0+cu118
...
```
## Installation Guide

```bash
$ conda env create -f environment.yml 
```

## Usage

Download the RNA-FM files at https://github.com/ml4bio/RNA-FM.
Then, you can train a model with a certain RBP dataset using the following command:
```bash
python main.py --data_file  TIA1_HepG2 --data_path ./datasets_K562_HepG2 --train --RNAFM_model_path ./RNA-FM  --model_save_path  ./results
```
After training, you can validate the model by using :
```bash
python main.py --data_file  TIA1_HepG2 --data_path ./datasets_K562_HepG2 --dynamic_validate --RNAFM_model_path ./RNA-FM  --model_save_path  ./results
```

## Data Availability

We downloaded the K562 and HepG2 datasets from the ENCODE database. https://www.encodeproject.org/publication-data/ENCSR456FVU/

## License

Thank you for using MCoMoE! Any questions, suggestions or advice are welcome!
Contact: 24220854050017@hainanu.edu.cn, aoyun@hainanu.edu.cn
