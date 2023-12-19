import os
from itertools import repeat
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from MoleculeSTM.datasets.utils import mol_to_graph_data_obj_simple
from rdkit.Chem import AllChem


class ChEMBL_Datasets_Graph(InMemoryDataset):
    def __init__(
        self, root, train_mode, assay_ChEMBL_id,
        transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.train_mode = train_mode

        self.label_file = os.path.join(root, "raw", "labels_{}.npz".format(train_mode))
        self.smiles_file = os.path.join(root, "raw", "smiles_{}.csv".format(train_mode))

        super(ChEMBL_Datasets_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

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
        
        raw_label_data = np.load(self.label_file)['labels']
        raw_label_data = raw_label_data[:, self.matrix_id]

        active_idx_list = []
        for idx, label in enumerate(raw_label_data):
            if label == 0:
                continue
            active_idx_list.append(idx)
        self.active_idx_list =active_idx_list
        print("active smiles and label: {}".format(len(active_idx_list)))

        self.label_list = raw_label_data
        print()
        return

    def get_graph(self, index):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[index], slices[index + 1])
            data[key] = item[s]
        return data

    def __getitem__(self, index):
        active_idx = self.active_idx_list[index]
        graph = self.get_graph(active_idx)
        y = self.label_list[active_idx]
        y = torch.tensor(y)
        return self.description, graph, y

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_{}'.format(self.train_mode))

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        data_list = []
        smiles_df = pd.read_csv(self.smiles_file)
        SMILES_list = smiles_df['smiles']

        for SMILES in SMILES_list:
            rdkit_mol = AllChem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(rdkit_mol)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))
        print()
        return

    def __len__(self):
        return len(self.active_idx_list)