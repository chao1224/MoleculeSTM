import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ChEMBL_Datasets_SMILES(Dataset):
    def __init__(self, root, train_mode, assay_ChEMBL_id):
        self.root = root

        # get assay
        assay_file = os.path.join(root, "raw", "assay.tsv")
        assay_df = pd.read_csv(assay_file, sep='\t')
        assay_ChEMBL_id_list = assay_df['assay_id'].tolist()
        matrix_id_list = assay_df['matrix_id'].tolist()
        assay_description_list = assay_df['assay_description'].tolist()
        
        self.matrix_id, self.description = None, None
        for ChEMBL_id, matrix_id, description in zip(assay_ChEMBL_id_list, matrix_id_list, assay_description_list):
            if ChEMBL_id == assay_ChEMBL_id:
                self.matrix_id = matrix_id
                self.description = description
                break
        print("*** target assay ChEMBL id: {}".format(assay_ChEMBL_id))
        print("*** target assay in the label matrix id: {}".format(self.matrix_id))
        print("*** target assay description: {}".format(self.description))

        smiles_file = os.path.join(root, "raw", "smiles_{}.csv".format(train_mode))
        smiles_df = pd.read_csv(smiles_file)
        smiles_list = smiles_df['smiles']
        
        label_file = os.path.join(root, "raw", "labels_{}.npz".format(train_mode))
        raw_label_data = np.load(label_file)['labels']
        raw_label_data = raw_label_data[:, self.matrix_id]
        assert len(smiles_list) == raw_label_data.shape[0]

        active_smiles_list, active_label_list = [], []
        for smiles, label in zip(smiles_list, raw_label_data):
            if label == 0:
                continue
            active_smiles_list.append(smiles)
            active_label_list.append(label)
        print("active smiles and label: {}".format(len(active_smiles_list)))

        self.SMILES_list = active_smiles_list
        self.label_list = active_label_list
        print()
        return

    def __getitem__(self, index):
        SMILES = self.SMILES_list[index]
        y = self.label_list[index]
        return self.description, SMILES, y

    def __len__(self):
        return len(self.SMILES_list)