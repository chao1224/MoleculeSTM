from lib2to3.pgen2 import token
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

from MoleculeSTM.datasets import DrugBank_Datasets_SMILES_ATC
from MoleculeSTM.utils import prepare_text_tokens
from transformers import BertTokenizer, BertForPreTraining


class BigModel(nn.Module):
    def __init__(self, main_model):
        super(BigModel, self).__init__()
        self.main_model = main_model
        self.dropout = nn.Dropout(0.1)

    def forward(self, tok, att, cud=True):
        typ = torch.zeros(tok.shape).long()
        if cud:
            typ = typ.cuda()
        pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler_output']
        logits = self.dropout(pooled_output)
        return logits


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


def save_model(save_best, epoch=None):
    if args.output_model_dir is not None:
        if save_best:
            model_file = "model.pth"

        elif epoch is None:
            model_file = "model_final.pth"

        else:
            model_file = "model_{}.pth".format(epoch)

        saved_file_path = os.path.join(args.output_model_dir, model_file)
        torch.save(model.state_dict(), saved_file_path)
    return


def get_text_and_SMILES_repr(text):
    text_tokens_ids, text_masks = prepare_text_tokens(
        device=device, description=text, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    text_repr = model(text_tokens_ids, text_masks)
    return text_repr


@torch.no_grad()
def eval_epoch(dataloader):
    model.eval()

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

        text_repr = get_text_and_SMILES_repr(text)
        molecule_repr = get_text_and_SMILES_repr(molecule_data)

        if test_mode == "given_text":
            if args.molecule_type == "SMILES":
                neg_molecule_repr = [get_text_and_SMILES_repr(neg_molecule_data[idx]) for idx in range(T_max)]
                neg_molecule_repr = torch.stack(neg_molecule_repr)
            else:
                neg_molecule_repr = [get_text_and_SMILES_repr(neg_molecule_data[idx]) for idx in range(T_max)]
                neg_molecule_repr = torch.stack(neg_molecule_repr)
            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(text_repr, molecule_repr, neg_molecule_repr[:T-1], args)
                accum_acc_list[T_idx] += acc
        elif test_mode == "given_molecule":
            neg_text_repr = [get_text_and_SMILES_repr(neg_text[idx]) for idx in range(T_max)]
            neg_text_repr = torch.stack(neg_text_repr)
            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(molecule_repr, text_repr, neg_text_repr[:T-1], args)
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

    ########## for dataset and split ##########
    parser.add_argument("--dataspace_path", type=str, default="../data")
    parser.add_argument("--task", type=str, default="molecule_ATC",
                        choices=["molecule_ATC", "molecule_ATC_overlap_PubChem"])
    parser.add_argument("--test_mode", type=str, default="given_text",
                        choices=["given_text", "given_molecule"])
    parser.add_argument("--ATC_level", type=int, default=5, choices=[1, 3, 4, 5])

    ########## for optimization ##########
    parser.add_argument("--T_list", type=int, nargs="+", default=[4, 10, 20])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
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
    ########## for 2D GNN ##########
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    ########## for saver ##########
    parser.add_argument("--eval_interval", type=int, default=5)
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
    
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BigModel(bert_model0.bert)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('../data/pretrained_KV-PLM/ckpt_ret01.pt'))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load('../data/pretrained_KV-PLM/ckpt_ret01.pt', map_location=torch.device('cpu') ))
    model.eval()
    
    T_max = max(args.T_list) - 1

    prompt_template = "This molecule is for {}."

    initial_test_acc_list, optimal_test_acc_list = [], []
    test_mode = args.test_mode
    dataset_folder = os.path.join(args.dataspace_path, "DrugBank_data")

    if args.task == "molecule_ATC":
        full_file_name = "SMILES_ATC_{}_full.txt".format(args.ATC_level)
    dataset_class = DrugBank_Datasets_SMILES_ATC
    dataloader_class = torch_DataLoader

    full_dataset = dataset_class(dataset_folder, full_file_name, neg_sample_size=T_max, prompt_template=prompt_template)

    full_dataloader = dataloader_class(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    initial_test_acc_list = eval_epoch(full_dataloader)
    print('Initial', initial_test_acc_list)

    row = ", ".join(["{:.4f}".format(x * 100) for x in initial_test_acc_list])
    print("initial results,", row)
