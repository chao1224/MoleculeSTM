# Dataset Specifications for MoleculeSTM

We provide the raw dataset (after preprocessing) at [this Hugging Face link](https://huggingface.co/datasets/chao1224/MoleculeSTM). Or you can download them by running `python download_datasets.py`.

## 1. Pretraining Dataset: PubChemSTM

For PubChemSTM, please note that we can only release the chemical structure information. If you need the textual data, please follow our preprocessing scripts.

## 2. Downstream Datasets

Please refer to the following for three downstream tasks:
- `DrugBank_data` for zero-shot structure-text retrieval
- `ZINC250K_data` for space alignment (step 1 in editing)
- `Editing_data` for zero-shot text-guided (step 2 in editing)
    - `single_multi_property_SMILES.txt` for single-objective, multi-objective, binding-affinity-based, and drug relevance editing
    - `neighbor2drug` for neighborhood searching for patent drug molecules
    - `ChEMBL_data` for binding editing
- `MoleculeNet_data` for molecular property prediction

# Checkpoints Specifications for MoleculeSTM

We provide the optimal checkpoints at [this Hugging Face link](https://huggingface.co/chao1224/MoleculeSTM). Or you can download them by running `python download_checkpoints.py`.
