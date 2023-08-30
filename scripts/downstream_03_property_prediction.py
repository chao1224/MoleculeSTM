import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader

from MoleculeSTM.datasets import MoleculeNetSMILESDataset, MoleculeNetGraphDataset
from MoleculeSTM.splitters import scaffold_split
from MoleculeSTM.utils import get_num_task_and_type, get_molecule_repr_MoleculeSTM
from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
from MoleculeSTM.models import GNN, GNN_graphpred


def train_classification(model, device, loader, optimizer):
    if args.training_mode == "fine_tuning":
        model.train()
    else:
        model.eval()
    linear_model.train()
    total_loss = 0

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        if args.molecule_type == "SMILES":
            SMILES_list, y = batch
            SMILES_list = list(SMILES_list)
            molecule_repr = get_molecule_repr_MoleculeSTM(
                SMILES_list, mol2latent=None,
                molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper)
            pred = linear_model(molecule_repr)
            pred = pred.float()
            y = y.to(device).float()
        else:
            batch = batch.to(device)
            molecule_repr = get_molecule_repr_MoleculeSTM(
                batch, mol2latent=None,
                molecule_type="Graph", molecule_model=model)
            pred = linear_model(molecule_repr)
            pred = pred.float()
            y = batch.y.view(pred.shape).to(device).float()

        is_valid = y ** 2 > 0
        loss_mat = criterion(pred, (y + 1) / 2)
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_classification(model, device, loader):
    model.eval()
    linear_model.eval()
    y_true, y_scores = [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        if args.molecule_type == "SMILES":
            SMILES_list, y = batch
            SMILES_list = list(SMILES_list)
            molecule_repr = get_molecule_repr_MoleculeSTM(
                SMILES_list, mol2latent=None,
                molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper)
            pred = linear_model(molecule_repr)
            pred = pred.float()
            y = y.to(device).float()
        else:
            batch = batch.to(device)
            molecule_repr = get_molecule_repr_MoleculeSTM(
                batch, mol2latent=None,
                molecule_type="Graph", molecule_model=model)
            pred = linear_model(molecule_repr)
            pred = pred.float()
            y = batch.y.view(pred.shape).to(device).float()

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print("{} is invalid".format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list), 0, y_true, y_scores


def train_regression(model, device, loader, optimizer):
    if args.training_mode == "fine_tuning":
        model.train()
    else:
        model.eval()
    linear_model.train()
    total_loss = 0

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        if args.molecule_type == "SMILES":
            SMILES_list, y = batch
            SMILES_list = list(SMILES_list)
            molecule_repr = get_molecule_repr_MoleculeSTM(
                SMILES_list, mol2latent=None,
                molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper)
            pred = linear_model(molecule_repr)
            pred = pred.float()
            y = y.to(device).float()
        else:
            batch = batch.to(device)
            molecule_repr = get_molecule_repr_MoleculeSTM(
                batch, mol2latent=None,
                molecule_type="Graph", molecule_model=model)
            pred = linear_model(molecule_repr)
            pred = pred.float()
            y = batch.y.view(pred.shape).to(device).float()

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_regression(model, device, loader):
    model.eval()
    y_true, y_pred = [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        if args.molecule_type == "SMILES":
            SMILES_list, y = batch
            SMILES_list = list(SMILES_list)
            molecule_repr = get_molecule_repr_MoleculeSTM(
                SMILES_list, mol2latent=None,
                molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper)
            pred = linear_model(molecule_repr)
            pred = pred.float()
            y = y.to(device).float()
        else:
            batch = batch.to(device)
            molecule_repr = get_molecule_repr_MoleculeSTM(
                batch, mol2latent=None,
                molecule_type="Graph", molecule_model=model)
            pred = linear_model(molecule_repr)
            pred = pred.float()
            y = batch.y.view(pred.shape).to(device).float()

        y_true.append(y)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--training_mode", type=str, default="fine_tuning", choices=["fine_tuning", "linear_probing"])
    parser.add_argument("--molecule_type", type=str, default="SMILES", choices=["SMILES", "Graph"])

    ########## for dataset and split ##########
    parser.add_argument("--dataspace_path", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="bace")
    parser.add_argument("--split", type=str, default="scaffold")

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--schedule", type=str, default="cycle")
    parser.add_argument("--warm_up_steps", type=int, default=10)

    ########## for MegaMolBART ##########
    parser.add_argument("--megamolbart_input_dir", type=str, default="../data/pretrained_MegaMolBART/checkpoints")
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

    parser.add_argument("--input_model_path", type=str, default=None)
    parser.add_argument("--output_model_dir", type=str, default=None)

    args = parser.parse_args()
    print("arguments\t", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    num_tasks, task_mode = get_num_task_and_type(args.dataset)
    dataset_folder = os.path.join(args.dataspace_path, "MoleculeNet_data", args.dataset)

    if args.molecule_type == "SMILES":
        dataset = MoleculeNetSMILESDataset(dataset_folder)
        dataloader_class = torch_DataLoader
        use_pyg_dataset = False
    else:
        dataset = MoleculeNetGraphDataset(dataset_folder, args.dataset)
        dataloader_class = pyg_DataLoader
        use_pyg_dataset = True

    assert args.split == "scaffold"
    print("split via scaffold")
    smiles_list = pd.read_csv(
        dataset_folder + "/processed/smiles.csv", header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset, smiles_list, null_value=0, frac_train=0.8,
        frac_valid=0.1, frac_test=0.1, pyg_dataset=use_pyg_dataset)

    train_loader = dataloader_class(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = dataloader_class(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = dataloader_class(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.molecule_type == "SMILES":
        if args.megamolbart_input_dir is not None:
            # This is loading from the pretarined_MegaMolBART
            # --megamolbart_input_dir=../../Datasets/pretrained_MegaMolBART/checkpoints
            MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.megamolbart_input_dir, output_dir=None)
            print("Start from pretrained MegaMolBART using MLM.")
        else:
            # This is starting from scratch
            MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=None, output_dir=None)
            print("Start from randomly initialized MegaMolBART.")
        model = MegaMolBART_wrapper.model
        if args.input_model_path is not None:
            print("Update MegaMolBART with pretrained MoleculeSTM. Loading from {}...".format(args.input_model_path))
            state_dict = torch.load(args.input_model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        molecule_dim = 256
    else:
        molecule_node_model = GNN(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
            JK=args.JK, drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type)
        model = GNN_graphpred(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
            num_tasks=1, molecule_node_model=molecule_node_model) 
        molecule_dim = args.gnn_emb_dim
        if args.input_model_path is not None:
            if "GraphMVP" in args.input_model_path:
                print("Start from pretrained model (GraphMVP) in {}.".format(args.input_model_path))
                model.from_pretrained(args.input_model_path)
            else:
                print("Start from pretrained model (MoleculeSTM) in {}.".format(args.input_model_path))
                state_dict = torch.load(args.input_model_path, map_location='cpu')
                model.load_state_dict(state_dict)
        else:
            print("Start from randomly initialized GNN.")

    # Rewrite the seed by MegaMolBART
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = model.to(device)
    linear_model = nn.Linear(molecule_dim, num_tasks).to(device)

    # set up optimizer
    if args.training_mode == "fine_tuning":
        model_param_group = [
            {"params": model.parameters()},
            {"params": linear_model.parameters(), 'lr': args.lr * args.lr_scale}
        ]
    else:
        model_param_group = [
            {"params": linear_model.parameters(), 'lr': args.lr * args.lr_scale}
        ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.weight_decay)

    if task_mode == "classification":
        train_func = train_classification
        eval_func = eval_classification

        train_roc_list, val_roc_list, test_roc_list = [], [], []
        train_acc_list, val_acc_list, test_acc_list = [], [], []
        best_val_roc, best_val_idx = -1, 0
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        for epoch in range(1, args.epochs + 1):
            loss_acc = train_func(model, device, train_loader, optimizer)
            print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

            if args.eval_train:
                train_roc, train_acc, train_target, train_pred = eval_func(model, device, train_loader)
            else:
                train_roc = train_acc = 0
            val_roc, val_acc, val_target, val_pred = eval_func(model, device, val_loader)
            test_roc, test_acc, test_target, test_pred = eval_func(model, device, test_loader)

            train_roc_list.append(train_roc)
            train_acc_list.append(train_acc)
            val_roc_list.append(val_roc)
            val_acc_list.append(val_acc)
            test_roc_list.append(test_roc)
            test_acc_list.append(test_acc)
            print("train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_roc, val_roc, test_roc))
            print()

            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_val_idx = epoch - 1
                if args.output_model_dir is not None:
                    ##### save best model #####
                    output_model_path = os.path.join(args.output_model_dir, "{}_model_best.pth".format(args.dataset))
                    saved_model_dict = {
                        "model": model.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)

                    filename = os.path.join(args.output_model_dir, "{}_evaluation_best.pth".format(args.dataset))
                    np.savez(
                        filename, val_target=val_target, val_pred=val_pred,
                        test_target=test_target, test_pred=test_pred)

        print("best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))

    else:
        train_func = train_regression
        eval_func = eval_regression
        criterion = torch.nn.MSELoss()
        
        train_result_list, val_result_list, test_result_list = [], [], []
        metric_list = ['RMSE', 'MAE']
        best_val_rmse, best_val_idx = 1e10, 0

        for epoch in range(1, args.epochs + 1):
            loss_acc = train_func(model, device, train_loader, optimizer)
            print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

            if args.eval_train:
                train_result, train_target, train_pred = eval_func(model, device, train_loader)
            else:
                train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}
            val_result, val_target, val_pred = eval_func(model, device, val_loader)
            test_result, test_target, test_pred = eval_func(model, device, test_loader)

            train_result_list.append(train_result)
            val_result_list.append(val_result)
            test_result_list.append(test_result)

            for metric in metric_list:
                print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))
            print()

            if val_result['RMSE'] < best_val_rmse:
                best_val_rmse = val_result['RMSE']
                best_val_idx = epoch - 1
                if args.output_model_dir is not None:
                    ##### save best model #####
                    output_model_path = os.path.join(args.output_model_dir, "{}_model_best.pth".format(args.dataset))
                    saved_model_dict = {
                        'model': model.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)

                    filename = os.path.join(args.output_model_dir, "{}_evaluation_best.pth".format(args.dataset))
                    np.savez(
                        filename, val_target=val_target, val_pred=val_pred,
                        test_target=test_target, test_pred=test_pred)

        for metric in metric_list:
            print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
                metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))

    ##### save final model #####
    if args.output_model_dir is not None:
        output_model_path = os.path.join(args.output_model_dir, '{}_model_final.pth'.format(args.dataset))
        saved_model_dict = {
            'model': model.state_dict()
        }
        torch.save(saved_model_dict, output_model_path)
    