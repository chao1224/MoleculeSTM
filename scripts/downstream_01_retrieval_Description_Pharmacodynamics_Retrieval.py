import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader

from transformers import AutoModel, AutoTokenizer
from MoleculeSTM.datasets import PubChemSTM_Datasets_SMILES, PubChemSTM_Datasets_Graph, DrugBank_Datasets_SMILES_retrieval, DrugBank_Datasets_Graph_retrieval
from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
from MoleculeSTM.models import GNN, GNN_graphpred
from MoleculeSTM.utils import prepare_text_tokens, get_molecule_repr_MoleculeSTM
from torch.utils.data import Dataset, DataLoader


class RetrievalDataset(Dataset):
    def __init__(self, repr_array):
        self.repr_array = repr_array

    def __len__(self):
        return len(self.repr_array)

    def __getitem__(self, idx):
        return torch.Tensor(self.repr_array[idx])


def do_CL_eval(X, Y, neg_Y, args):
    X = F.normalize(X, dim=-1)
    X = X.unsqueeze(1) # B, 1, d

    Y = Y.unsqueeze(0)
    Y = torch.cat([Y, neg_Y], dim=0) # T, B, d
    Y = Y.transpose(0, 1)  # B, T, d
    Y = F.normalize(Y, dim=-1)

    logits = torch.bmm(X, Y.transpose(1, 2)).squeeze()  # B*T
    B = X.size()[0]
    labels = torch.zeros(B).long().to(logits.device)  # B*1

    criterion = nn.CrossEntropyLoss()

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    confidence = logits
    CL_conf = confidence.max(dim=1)[0]
    CL_conf = CL_conf.cpu().numpy()

    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B
    return CL_loss, CL_conf, CL_acc


def get_text_repr(text):
    text_tokens_ids, text_masks = prepare_text_tokens(
        device=device, description=text, tokenizer=text_tokenizer, max_seq_len=args.max_seq_len)
    text_output = text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
    text_repr = text_output["pooler_output"]
    return text_repr


@torch.no_grad()
def extract_retrieval_representation(retrieval_dataloader):
    if args.verbose:
        L = tqdm(retrieval_dataloader)
    else:
        L = retrieval_dataloader

    retrieval_molecule_repr_list, retrieval_description_representation_list = [], []
    for step, batch in enumerate(L):
        description = batch[0]
        molecule_data = batch[1]
        
        try:
            description_tokens_ids, description_masks = prepare_text_tokens(
                device=device, description=description, tokenizer=text_tokenizer, max_seq_len=args.max_seq_len)
            description_output = text_model(input_ids=description_tokens_ids, attention_mask=description_masks)
            description_repr = description_output["pooler_output"]

            if args.molecule_type == "SMILES":
                molecule_data = list(molecule_data) # for SMILES_list
                molecule_repr = get_molecule_repr_MoleculeSTM(
                    molecule_data, mol2latent=None,
                    molecule_type=args.molecule_type, MegaMolBART_wrapper=MegaMolBART_wrapper)
            else:
                molecule_data = molecule_data.to(device)
                molecule_repr = get_molecule_repr_MoleculeSTM(
                    molecule_data, mol2latent=None,
                    molecule_type=args.molecule_type, molecule_model=molecule_model)
        except:
            continue
        retrieval_description_representation_list.append(description_repr.detach().cpu().numpy())
        retrieval_molecule_repr_list.append(molecule_repr.detach().cpu().numpy())
            
    retrieval_description_representation_array = np.concatenate(retrieval_description_representation_list)
    retrieval_molecule_representation_array = np.concatenate(retrieval_molecule_repr_list)

    return retrieval_description_representation_array, retrieval_molecule_representation_array


def get_similarity_array(X, retrieval_loader):
    sim_list = []
    if args.verbose:
        L = tqdm(retrieval_loader)
    else:
        L = retrieval_loader
    for batch in L:
        batch = batch.to(device)
        sim = torch.matmul(X, batch.transpose(1, 0)).detach().cpu().numpy()
        sim_list.append(sim)
    sim_array = np.concatenate(sim_list, axis=1)
    return sim_array


@torch.no_grad()
def eval_epoch(dataloader):
    text_model.eval()
    molecule_model.eval()

    accum_acc_list = [0 for _ in args.T_list]
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    for batch in L:
        text = batch[0]
        molecule_data = batch[1]
        neg_text = batch[2]
        neg_molecule_data = batch[3]

        text_repr = get_text_repr(text)
        if args.molecule_type == "SMILES":
            molecule_data = list(molecule_data) # for SMILES_list
            molecule_repr = get_molecule_repr_MoleculeSTM(
                molecule_data, mol2latent=None,
                molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper)
        else:
            molecule_data = molecule_data.to(device)
            molecule_repr = get_molecule_repr_MoleculeSTM(
                molecule_data, mol2latent=None,
                molecule_type="Graph", molecule_model=molecule_model
            )
            
        if test_mode == "given_text":
            if args.molecule_type == "SMILES":
                neg_molecule_repr = [
                    get_molecule_repr_MoleculeSTM(
                        list(neg_molecule_data[idx]), mol2latent=None,
                        molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper) for idx in range(T_max)
                ]
                neg_molecule_repr = torch.stack(neg_molecule_repr)
            else:
                neg_molecule_repr = [
                    get_molecule_repr_MoleculeSTM(
                        neg_molecule_data[idx].to(device), mol2latent=None,
                        molecule_type="Graph", molecule_model=molecule_model) for idx in range(T_max)
                ]
                neg_molecule_repr = torch.stack(neg_molecule_repr)
            
            # Next we will do the retrieval:
            # text_repr -> retrieval_description_representation_array -> retrieval_molecule_representation_array
            similarity_array = get_similarity_array(text_repr, retrieval_description_representation_dataloader)
            batch_size = similarity_array.shape[0]
            retrieved_text_repr_list = []
            for batch_i in range(batch_size):
                temp_similarity_array = similarity_array[batch_i]
                sorted_index = np.argsort(temp_similarity_array)[::-1]
                optimal_index = sorted_index[0]
                retrieved_text_repr_list.append(retrieval_molecule_representation_array[optimal_index])
            retrieved_text_repr_list = np.array(retrieved_text_repr_list)
            retrieved_text_repr = torch.Tensor(retrieved_text_repr_list).to(device)

            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(retrieved_text_repr, molecule_repr, neg_molecule_repr[:T-1], args)
                accum_acc_list[T_idx] += acc
        
        elif test_mode == "given_molecule":
            neg_text_repr = [get_text_repr(neg_text[idx]) for idx in range(T_max)]
            neg_text_repr = torch.stack(neg_text_repr)

            # Next we will do the retrieval:
            # molecule_repr -> retrieval_molecule_representation_array -> retrieval_description_representation_array
            similarity_array = get_similarity_array(molecule_repr, retrieval_molecule_representation_dataloader)
            batch_size = similarity_array.shape[0]
            retrieved_mol_repr_list = []
            for batch_i in range(batch_size):
                temp_similarity_array = similarity_array[batch_i]
                sorted_index = np.argsort(temp_similarity_array)[::-1]
                optimal_index = sorted_index[0]
                retrieved_mol_repr_list.append(retrieval_description_representation_array[optimal_index])
            retrieved_mol_repr_list = np.array(retrieved_mol_repr_list)
            retrieved_mol_repr = torch.Tensor(retrieved_mol_repr_list).to(device)

            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(retrieved_mol_repr, text_repr, neg_text_repr[:T-1], args)
                accum_acc_list[T_idx] += acc
        else:
            raise Exception
    
    accum_acc_list = np.array(accum_acc_list)
    accum_acc_list /= len(dataloader)
    return accum_acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--text_type", type=str, default="SciBERT", choices=["SciBERT", "BioBERT"])
    parser.add_argument("--load_latent_projector", type=int, default=1)
    parser.add_argument("--model_loading_mode", type=str, default="load_from_latest", choices=["load_from_latest", "load_mode_0", "load_mode_1"])
    parser.add_argument("--training_mode", type=str, default="zero_shot", choices=["zero_shot"])
    parser.add_argument("--retrieval_folder", type=str, default="retrieval_similarity")

    ########## for dataset and split ##########
    parser.add_argument("--dataspace_path", type=str, default="../data")
    parser.add_argument("--task", type=str, default="molecule_description",
        choices=[
            "molecule_description", "molecule_description_Raw",
            "molecule_description_removed_PubChem", "molecule_description_removed_PubChem_Raw",
            "molecule_pharmacodynamics", "molecule_pharmacodynamics_Raw",
            "molecule_pharmacodynamics_removed_PubChem", "molecule_pharmacodynamics_removed_PubChem_Raw"])
    parser.add_argument("--test_mode", type=str, default="given_text", choices=["given_text", "given_molecule"])

    ########## for optimization ##########
    parser.add_argument("--T_list", type=int, nargs="+", default=[4, 10, 20])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--text_lr", type=float, default=1e-5)
    parser.add_argument("--mol_lr", type=float, default=1e-5)
    parser.add_argument("--text_lr_scale", type=float, default=0.1)
    parser.add_argument("--mol_lr_scale", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0)
    
    ########## for contrastive objective ##########
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE"])
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    ########## for BERT model ##########
    parser.add_argument("--max_seq_len", type=int, default=512)

    ########## for molecule model ##########
    parser.add_argument("--molecule_type", type=str, default="SMILES", choices=["SMILES", "Graph"])
    
    ########## for MegaMolBART ##########
    parser.add_argument("--vocab_path", type=str, default="../MoleculeSTM/bart_vocab.txt")

    ########## for 2D GNN ##########
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    ########## for saver ##########
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--input_model_dir", type=str, default=None)
    parser.add_argument("--input_model_path", type=str, default=None)
    parser.add_argument("--output_model_dir", type=str, default=None)

    args = parser.parse_args()
    print("arguments\t", args)
    torch.multiprocessing.set_sharing_strategy('file_system')

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    ##### prepare text model #####
    ##### by default, this is load_mode_1 #####
    if args.text_type == "SciBERT":
        pretrained_SciBERT_folder = os.path.join(args.dataspace_path, 'pretrained_SciBERT')
        text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
        text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(device)
        text_dim = 768
    else:
        raise Exception

    if args.model_loading_mode == "load_from_latest":
        input_model_path = os.path.join(args.input_model_dir, "text_model.pth")
        print("Loading from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        text_model.load_state_dict(state_dict)
    elif args.model_loading_mode == "load_mode_0":
        text_model.init_weights()
        print("Random init for BERT.")

    ##### prepare molecule model #####
    if args.molecule_type == "SMILES":
        if args.model_loading_mode == "load_from_latest":
            input_model_path = os.path.join(args.input_model_dir, "molecule_model.pth")
            print("Loading from {}...".format(input_model_path))
            MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=None, output_dir=None)
            molecule_model = MegaMolBART_wrapper.model
            state_dict = torch.load(input_model_path, map_location='cpu')
            molecule_model.load_state_dict(state_dict)
        elif args.model_loading_mode == "load_mode_0":
            MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=None, output_dir=None)
            molecule_model = MegaMolBART_wrapper.model
            print("Random init for MegaMolBART.")
        elif args.model_loading_mode == "load_mode_1":
            # This is loading from the pretarined_MegaMolBART
            MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir="../data/pretrained_MegaMolBART/checkpoints", output_dir=None)
            molecule_model = MegaMolBART_wrapper.model
            print("Loading from ../data/pretrained_MegaMolBART/checkpoints.")
        molecule_dim = 256

    else:
        molecule_node_model = GNN(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
            JK=args.JK, drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type)
        molecule_model = GNN_graphpred(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
            num_tasks=1, molecule_node_model=molecule_node_model) 
        molecule_dim = args.gnn_emb_dim
        if args.model_loading_mode == "load_from_latest":
            input_model_path = os.path.join(args.input_model_dir, "molecule_model.pth")
            print("Loading from {}...".format(input_model_path))
            state_dict = torch.load(input_model_path, map_location='cpu')
            molecule_model.load_state_dict(state_dict)
        elif args.model_loading_mode == "load_mode_0":
            print("Random init for GNN.")
        elif args.model_loading_mode == "load_mode_1":
            print("Loading from ../data/pretrained_GraphMVP/GraphMVP_G/model.pth")
            molecule_model.from_pretrained("../data/pretrained_GraphMVP/GraphMVP_G/model.pth")

    # Rewrite the seed by MegaMolBART
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    text_model = text_model.to(device)
    molecule_model = molecule_model.to(device)
    
    T_max = max(args.T_list) - 1

    initial_test_acc_list = []
    test_mode = args.test_mode
    dataset_folder = os.path.join(args.dataspace_path, "DrugBank_data")
    if args.molecule_type == "SMILES":
        dataset_class = DrugBank_Datasets_SMILES_retrieval
        dataloader_class = torch_DataLoader

        if args.task == "molecule_description":
            template = "SMILES_description_{}.txt"
        elif args.task == "molecule_description_removed_PubChem":
            template = "SMILES_description_removed_from_PubChem_{}.txt"
        elif args.task == "molecule_description_Raw":
            template = "SMILES_description_{}_Raw.txt"
        elif args.task == "molecule_description_removed_PubChem_Raw":
            template = "SMILES_description_removed_from_PubChem_{}_Raw.txt"
        elif args.task == "molecule_pharmacodynamics":
            template = "SMILES_pharmacodynamics_{}.txt"
        elif args.task == "molecule_pharmacodynamics_removed_PubChem":
            template = "SMILES_pharmacodynamics_removed_from_PubChem_{}.txt"
        elif args.task == "molecule_pharmacodynamics_Raw":
            template = "SMILES_pharmacodynamics_{}_Raw.txt"
        elif args.task == "molecule_pharmacodynamics_removed_PubChem_Raw":
            template = "SMILES_pharmacodynamics_removed_from_PubChem_{}_Raw.txt"
        full_dataset = dataset_class(dataset_folder, 'full', neg_sample_size=T_max, template=template)

        dataset_root = os.path.join(args.dataspace_path, "PubChemSTM_data")
        retrieval_dataset = PubChemSTM_Datasets_SMILES(dataset_root)

    else:
        dataset_class = DrugBank_Datasets_Graph_retrieval
        dataloader_class = pyg_DataLoader
        processed_dir_prefix = args.task

        if args.task == "molecule_description":
            template = "SMILES_description_{}.txt"
        elif args.task == "molecule_description_removed_PubChem":
            template = "SMILES_description_removed_from_PubChem_{}.txt"
        elif args.task == "molecule_description_Raw":
            template = "SMILES_description_{}_Raw.txt"
        elif args.task == "molecule_description_removed_PubChem_Raw":
            template = "SMILES_description_removed_from_PubChem_{}_Raw.txt"
        elif args.task == "molecule_pharmacodynamics":
            template = "SMILES_pharmacodynamics_{}.txt"
        elif args.task == "molecule_pharmacodynamics_removed_PubChem":
            template = "SMILES_pharmacodynamics_removed_from_PubChem_{}.txt"
        elif args.task == "molecule_pharmacodynamics_Raw":
            template = "SMILES_pharmacodynamics_{}_Raw.txt"
        elif args.task == "molecule_pharmacodynamics_removed_PubChem_Raw":
            template = "SMILES_pharmacodynamics_removed_from_PubChem_{}_Raw.txt"
        full_dataset = dataset_class(dataset_folder, 'full', neg_sample_size=T_max, processed_dir_prefix=processed_dir_prefix, template=template)

        dataset_root = os.path.join(args.dataspace_path, "PubChemSTM_data")
        retrieval_dataset = PubChemSTM_Datasets_Graph(dataset_root)

    full_dataloader = dataloader_class(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # The program will get blcoked with none-zero num_workers
    retrieval_dataloader = dataloader_class(retrieval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.retrieval_folder, exist_ok=True)
    retrieval_datapath = "{}/{}_{}".format(args.retrieval_folder, args.molecule_type, args.task)
    if os.path.exists(retrieval_datapath+".npz"):
        data = np.load(retrieval_datapath+".npz")
        retrieval_description_representation_array = data["retrieval_description_representation_array"]
        retrieval_molecule_representation_array = data["retrieval_molecule_representation_array"]
    else:
        retrieval_description_representation_array, retrieval_molecule_representation_array = extract_retrieval_representation(retrieval_dataloader)
        np.savez(retrieval_datapath, retrieval_description_representation_array=retrieval_description_representation_array, retrieval_molecule_representation_array=retrieval_molecule_representation_array)
    retrieval_description_representation_dataset = RetrievalDataset(retrieval_description_representation_array)
    retrieval_description_representation_dataloader = DataLoader(retrieval_description_representation_dataset, batch_size=512, shuffle=False, num_workers=args.num_workers)
    retrieval_molecule_representation_dataset = RetrievalDataset(retrieval_molecule_representation_array)
    retrieval_molecule_representation_dataloader = DataLoader(retrieval_molecule_representation_dataset, batch_size=512, shuffle=False, num_workers=args.num_workers)

    initial_test_acc_list = eval_epoch(full_dataloader)
    print('Initial', initial_test_acc_list)

    row = ", ".join(["{:.4f}".format(x * 100) for x in initial_test_acc_list])
    print("initial results,", row)
    