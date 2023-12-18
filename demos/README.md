Here we show demos on how to run MoleculeSTM pretraining and downstream tasks.

## Checkpoints for Demo

First, please check [this Hugging Face link](https://huggingface.co/chao1224/MoleculeSTM/tree/main/demo) for toy checkpoints.

Or you can run the following (also in `download.py`):
```
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
snapshot_download(repo_id="chao1224/MoleculeSTM", repo_type="model", local_dir='.', allow_patterns="*demo*")
```

Then move the folders under `demos` to this folder. The folder structure is the following:
```
.
├── demo_checkpoints_Graph
│   ├── foundation2generation_model.pth
│   ├── generation2foundation_model.pth
│   ├── mol2latent_model_final.pth
│   ├── mol2latent_model.pth
│   ├── molecule_model_final.pth
│   ├── molecule_model.pth
│   ├── text2latent_model_final.pth
│   ├── text2latent_model.pth
│   ├── text_model_final.pth
│   └── text_model.pth
├── demo_checkpoints_MegaMolBART
│   ├── foundation2generation_model.pth
│   ├── generation2foundation_model.pth
│   ├── mol2latent_model_final.pth
│   ├── mol2latent_model.pth
│   ├── molecule_model_final.pth
│   ├── molecule_model.pth
│   ├── text2latent_model_final.pth
│   ├── text2latent_model.pth
│   ├── text_model_final.pth
│   └── text_model.pth
├── demo_downstream_property_prediction_Graph.ipynb
├── demo_downstream_property_prediction_SMILES.ipynb
├── demo_downstream_retrieval_Graph.ipynb
├── demo_downstream_retrieval_SMILES.ipynb
├── demo_downstream_zero_shot_molecule_edit.ipynb
├── demo_pretrain_Graph.ipynb
├── demo_pretrain_SMILES.ipynb
├── download.py
└── README.md
```

## Pretraining

Please check `demo_pretrain_Graph.ipynb` and `demo_pretrain_SMILES.ipynb`.

## Downstream

Then we provide notebooks for three types of downstream tasks:
- For zero-shot structure-text retrieval: `demo_downstream_retrieval_SMILES.ipynb` and `demo_downstream_retrieval_Graph.ipynb`.
- For zero-shot text-based molecule editing: `demo_downstream_zero_shot_molecule_edit.ipynb`
    - Notice that at this step, we are only using the textual branch (SciBERT) and a pretrained molecule generative model (MegaMolBART). The MoleculeSTM chemical branch (MegaMolBART or GraphMVP) is only used at the module alignment phase, and we can change it in the `MoleculeSTM_model_dir` argument.
- For molecular property prediction: `demo_downstream_property_prediction_SMILES.ipynb` and `demo_downstream_property_prediction_Graph.ipynb`.
