import argparse
import math
import numpy as np
import os

import torch
import torch.nn.functional as F
from MoleculeSTM.downstream_molecule_edit_utils import get_SMILES_list, get_description_list, evaluate_SMILES_list
from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def mean_pooling(token_embeddings, attention_mask):
    attention_mask = ~attention_mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask


def check_edit(SMILES, text):
    first_and_second_SMILES_list = []

    SMILES_list = [SMILES]
    latent_code_init, pad_mask_init = MegaMolBART_wrapper.smileslist2embedding(SMILES_list)  # [pad, B, d], [pad, B]
    first_and_second_SMILES_list.append(SMILES) 

    generated_mols = MegaMolBART_wrapper.inverse_transform([latent_code_init], pad_mask_init.bool().cuda(), k=1, sanitize=True)
    first_and_second_SMILES_list.append(generated_mols[0])

    alpha_list = [
        1.0, 1.5, 2.0, 2.5, 3.0
    ]
    result_SMILES_list_one_pair, result_eval_list_one_pair = [], []
    
    print("Use random noise for init")
    random_noise = args.variance * torch.randn(latent_code_init.size()).to(device)
    random_noise = F.normalize(random_noise, dim=-1)
    
    for alpha in alpha_list:
        print("alpha: {}".format(alpha))
        current_SMILES_list = [first_and_second_SMILES_list[0]] + [first_and_second_SMILES_list[1]]

        latent = latent_code_init + alpha * random_noise / len(latent_code_init)
        pad_mask = pad_mask_init
        
        generated_mols = MegaMolBART_wrapper.inverse_transform([latent], pad_mask.bool().cuda(), k=1, sanitize=True)
        current_SMILES_list.append(generated_mols[0])
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

    ########## for generation ##########
    parser.add_argument("--generation_model_dir", type=str, default="../data/pretrained_MegaMolBART/checkpoints")
    args = parser.parse_args()

    print(args)
    MegaMolBART_wrapper = MegaMolBART(input_dir=args.generation_model_dir, output_dir=None)
    molecule_model_generation = MegaMolBART_wrapper.model
    print("Loading from pretrained MegaMolBART ({}).".format(args.generation_model_dir))
    molecule_dim_generation = 256
    
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

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
        result_acc_list = np.sum(result_acc_list, axis=0, keepdims=True)
        result_acc_list = 100. * result_acc_list / len(source_SMILES_list)
        print(description, result_acc_list)
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
