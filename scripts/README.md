Most of the checkpoints perform quite stable performance, yet here we provide details on exactly which we use in the manuscript.

## 1 Downstream: Zero-shot Structure-text Retrieval

### 1.1 DrugBank-Description

| Type | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| SMILES | `pretrained_MoleculeSTM/SciBERT-MegaMolBART-3e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Figure 2 / Table 8 (ArXiv) |
| Graph | `pretrained_MoleculeSTM/SciBERT-Graph-3e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Figure 2 / Table 8 (ArXiv) |

### 1.2 DrugBank-Pharmacodynamics

| Type | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| SMILES | `pretrained_MoleculeSTM_Raw/SciBERT-MegaMolBART-1e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Figure 2 / Table 9 (ArXiv) |
| Graph | `pretrained_MoleculeSTM_Raw/SciBERT-Graph-1e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Figure 2 / Table 9 (ArXiv) |

### 1.3 DrugBank-ATC

| Type | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| SMILES | `pretrained_MoleculeSTM/SciBERT-MegaMolBART-1e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Figure 2 / Table 10 (ArXiv) |
| Graph | `pretrained_MoleculeSTM/SciBERT-Graph-1e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Figure 2 / Table 10 (ArXiv) |

## 2 Downstream: Zero-shot Text-based Molecule Editing

### 2.1 Single-objective molecule editing

| Type | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| SMILES | `pretrained_MoleculeSTM_Raw/SciBERT-MegaMolBART-3e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Figure 4 / Figure 5 / Sec D.2 |
| Graph | `pretrained_MoleculeSTM/SciBERT-Graph-3e-5-1-1e-4-1-EBM_NCE-0.1-32-32` | Figure 2 / Figure 5 / Sec D.2|

### 2.2 Multi-objective molecule editing

| Type | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| SMILES | `pretrained_MoleculeSTM/SciBERT-MegaMolBART-3e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Figure 4 / Figure 5 / Sec D.3 |
| Graph | `pretrained_MoleculeSTM/SciBERT-Graph-3e-5-1-1e-4-1-EBM_NCE-0.1-32-32` | Figure 2 / Figure 5 / Sec D.3 |

### 2.3 Binding-affinity-based molecule editing

| Type | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| SMILES | `pretrained_MoleculeSTM/SciBERT-MegaMolBART-1e-5-1-1e-4-1-EBM_NCE-0.1-32-32` | Figure 4 / Figure 5 / Sec D.4 |
| Graph | `pretrained_MoleculeSTM/SciBERT-Graph-3e-5-1-1e-4-1-EBM_NCE-0.1-32-32` | Figure 2 / Figure 5 / Sec D.4 |

### 2.4 Drug relevance editing

| Type | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| SMILES | `pretrained_MoleculeSTM/SciBERT-MegaMolBART-1e-5-1-1e-4-1-EBM_NCE-0.1-32-32` | Sec D.5 |
| Graph | `pretrained_MoleculeSTM/SciBERT-Graph-3e-5-1-1e-4-1-EBM_NCE-0.1-32-32` |  Sec D.5 |

### 2.5 Neighborhood searching

| Type | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| SMILES | `pretrained_MoleculeSTM_Raw/SciBERT-MegaMolBART-3e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Figure 5 / Sec D.6 |


## 3 Downstream: Molecular Property Prediction

| Type | Checkpoints | Tables/Figures in Manuscript|
| -- | -- | -- |
| SMILES | `pretrained_MoleculeSTM/SciBERT-MegaMolBART-1e-5-1-1e-4-1-EBM_NCE-0.1-32-32` | Table 1 |
| Graph | `pretrained_MoleculeSTM/SciBERT-Graph-3e-5-1-1e-4-1-InfoNCE-0.1-32-32` | Table 1 |
