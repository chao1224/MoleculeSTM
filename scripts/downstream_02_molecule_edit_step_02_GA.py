import argparse
from curses import tparm
import numpy as np
import random
import os

import torch
from MoleculeSTM.downstream_molecule_edit_utils import get_SMILES_list, get_description_list, evaluate_SMILES_list
from rdkit import Chem
import MoleculeSTM.models.GA.mutate as mu


def check_edit(SMILES, text):
    first_and_second_SMILES_list = []

    first_and_second_SMILES_list.append(SMILES) 
    first_and_second_SMILES_list.append(SMILES) 

    alpha_list = [1]
    result_SMILES_list_one_pair, result_eval_list_one_pair = [], []
    mol = Chem.MolFromSmiles(SMILES)
    
    for alpha in alpha_list:
        print("alpha: {}".format(alpha))
        current_SMILES_list = [first_and_second_SMILES_list[0]] + [first_and_second_SMILES_list[1]]
        
        mutated_mol = mol
        for _ in range(args.mutation_step):
            try:
                while True:
                    mutated_mol = mu.mutate(mutated_mol, args.mutation_rate)
                    if mutated_mol is not None:
                        break
            except:
                mutated_mol = mol

        generated_SMILES = Chem.MolToSmiles(mutated_mol)
        current_SMILES_list.append(generated_SMILES)
        result_SMILES_list_one_pair.append([text] + current_SMILES_list + ['{}'.format(alpha)])

        current_result_list = evaluate_SMILES_list(current_SMILES_list, text)
        result_eval_list_one_pair.append(current_result_list)
        print()
    
    result_eval_list_one_pair = np.array(result_eval_list_one_pair)
    result_eval_list_one_pair = np.any(result_eval_list_one_pair, axis=0, keepdims=True)
    print("result_eval_list_one_pair\n", result_eval_list_one_pair)
    return result_SMILES_list_one_pair, result_eval_list_one_pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)

    ########## for editing ##########
    parser.add_argument("--input_description", type=str, default=None)
    parser.add_argument("--input_description_id", type=int, default=None)
    parser.add_argument("--input_SMILES", type=str, default=None)
    parser.add_argument("--input_SMILES_file", type=str, default="../data/Editing_data/single_multi_property_SMILES.txt")
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--variance", type=float, default=1)
    parser.add_argument("--mutation_rate", type=float, default=1)
    parser.add_argument("--mutation_step", type=int, default=1)

    args = parser.parse_args()

    print(args)
    
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    
    print("\n\n\nstart editing\n\n\n")

    source_SMILES_list = get_SMILES_list(args)
    description_list = get_description_list(args)

    for description in description_list:
        print("===== for description {} =====".format(description))
        result_SMILES_list, result_acc_list = [], []
        for SMILES in source_SMILES_list:
            print("===== for SMILES {} =====".format(SMILES))
            result_SMILES_list_, result_acc_list_ = check_edit(SMILES, description)
            result_SMILES_list.extend(result_SMILES_list_)
            result_acc_list.append(result_acc_list_)
            print("\n\n\n")
        
        result_acc_list = np.concatenate(result_acc_list, axis=0)
        result_acc_list = np.sum(result_acc_list, axis=0)
        result_acc_list = 100. * result_acc_list / len(source_SMILES_list)
        result_acc_row = '\t'.join(['{}'.format(x) for x in result_acc_list])
        print("===== Accuracy =====\t{}".format(result_acc_row))

        if args.output_model_dir is not None:
            saver_file = os.path.join(args.output_model_dir, "edited_SMILES.tsv")
            with open(saver_file, 'a') as f:
                for row in result_SMILES_list:
                    row = "\t".join(row)
                    print(row, file=f)

            saver_file = os.path.join(args.output_model_dir, "accuracy")
            np.savez(saver_file, result_acc_list)
