import os
import numpy as np
from collections import defaultdict
from foundation.datasets import ChEMBL_Datasets_SMILES
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json

sklearn_model_list = ["RF", "LR"]
sklearn_model_num = len(sklearn_model_list)


assay_dict = {
    401: ["CHEMBL1613777", "This molecule is tested positive in an assay that are inhibitors and substrates of an enzyme protein. It uses molecular oxygen inserting one oxygen atom into a substrate, and reducing the second into a water molecule."],
    402: ["CHEMBL1613797", "This molecule is tested positive in an assay for Anthrax Lethal, which acts as a protease that cleaves the N-terminal of most dual specificity mitogen-activated protein kinase kinases."],
    403: ["CHEMBL2114713", "This molecule is tested positive in an assay for Activators of ClpP, which cleaves peptides in various proteins in a process that requires ATP hydrolysis and has a limited peptidase activity in the absence of ATP-binding subunits."],
    404: ["CHEMBL1613838", "This molecule is tested positive in an assay for activators involved in the transport of proteins between the endosomes and the trans Golgi network."],
    405: ["CHEMBL1614236", "This molecule is an inhibitor of a protein that prevents the establishment of the cellular antiviral state by inhibiting ubiquitination that triggers antiviral transduction signal and inhibits post-transcriptional processing of cellular pre-mRNA."],
    406: ["CHEMBL1613903", "This molecule is tested positive in the high throughput screening assay to identify inhibitors of the SARS coronavirus 3C-like Protease, which cleaves the C-terminus of replicase polyprotein at 11 sites."],
}


def smiles_to_fps(smiles, ecfp_radius=2, ecfp_length=2048):
    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, ecfp_radius, nBits=ecfp_length)
    return ecfp


def extract_most_similar_pos_neg(pos_smiles_list, pos_ecfp_list, neg_smiles_list, neg_ecfp_list):
    optimal_pos_smiles, optimal_pos_ecfp, optimal_neg_smiles, optimal_neg_ecfp, optimal_sim_value = None, None, None, None, -1
    for pos_smiles, pos_ecfp in zip(pos_smiles_list, pos_ecfp_list):
        for neg_smiles, neg_ecfp in zip(neg_smiles_list, neg_ecfp_list):
            value = DataStructs.TanimotoSimilarity(pos_ecfp, neg_ecfp)
            if  value > optimal_sim_value and len(pos_smiles) <= 20:
                optimal_sim_value = value
                optimal_pos_smiles = pos_smiles
                optimal_pos_ecfp = pos_ecfp
                optimal_neg_smiles = neg_smiles
                optimal_neg_ecfp = neg_ecfp

    return optimal_pos_smiles, optimal_pos_ecfp, optimal_neg_smiles, optimal_neg_ecfp


def extract_ChEMBL_assay(assay_dict):
    sklearn_model_record = defaultdict(dict)
    for text_id, assay_record in assay_dict.items():
        print("text_id: {}".format(text_id))
        assay_ChEMBL_id = assay_record[0]
        print("assay_ChEMBL_id: {}".format(assay_ChEMBL_id))

        dataspace_path = "../../Datasets"
        dataset_folder = os.path.join(dataspace_path, "ChEMBL_data", "degree_threshold_0")

        train_dataset = ChEMBL_Datasets_SMILES(dataset_folder, 'train', assay_ChEMBL_id)
        test_dataset = ChEMBL_Datasets_SMILES(dataset_folder, 'test', assay_ChEMBL_id)

        def load_smiles(dataset):
            positive_smiles_list, negative_smiles_list = [], []
            L = len(dataset)
            for i in range(L):
                _, smiles, label = dataset[i]
                if label == 1:
                    positive_smiles_list.append(smiles)
                elif label == -1:
                    negative_smiles_list.append(smiles)
            return positive_smiles_list, negative_smiles_list
        
        positive_smiles_list, negative_smiles_list = [], []
        
        positive_smiles_list_, negative_smiles_list_ = load_smiles(train_dataset)
        positive_smiles_list.extend(positive_smiles_list_)
        negative_smiles_list.extend(negative_smiles_list_)

        positive_smiles_list_, negative_smiles_list_ = load_smiles(test_dataset)
        positive_smiles_list.extend(positive_smiles_list_)
        negative_smiles_list.extend(negative_smiles_list_)

        print("len of positive: {}".format(len(positive_smiles_list)))
        print("len of negative: {}".format(len(negative_smiles_list)))

        positive_fp_list = [smiles_to_fps(smiles) for smiles in positive_smiles_list]
        negative_fp_list = [smiles_to_fps(smiles) for smiles in negative_smiles_list]

        optimal_pos_smiles, optimal_pos_ecfp, optimal_neg_smiles, optimal_neg_ecfp = extract_most_similar_pos_neg(positive_smiles_list, positive_fp_list, negative_smiles_list, negative_fp_list)
        print("pos smiles:", optimal_pos_smiles)
        print("neg smiles:", optimal_neg_smiles)
        print("similarity in between: ", DataStructs.TanimotoSimilarity(optimal_pos_ecfp, optimal_neg_ecfp))

        optimal_fp_list = [optimal_pos_ecfp, optimal_neg_ecfp]

        positive_fp_list = [[float(x) for x in fp.ToBitString()] for fp in positive_fp_list]
        negative_fp_list = [[float(x) for x in fp.ToBitString()] for fp in negative_fp_list]
        optimal_fp_list = [[float(x) for x in fp.ToBitString()] for fp in optimal_fp_list]
        positive_label = [1 for _ in positive_smiles_list]
        negative_label = [0 for _ in negative_smiles_list]
        x = positive_fp_list + negative_fp_list
        x = np.array(x)
        y = positive_label + negative_label
        y = np.array(y)
        optimal_x = np.array(optimal_fp_list)

        for sklearn_model in sklearn_model_list:
            print("sklearn model {}".format(sklearn_model))
            if sklearn_model == "RF":
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_features="log2",
                    min_samples_leaf=1,
                    n_jobs=8,
                    class_weight="balanced_subsample",
                    random_state=123,
                    oob_score=False,
                    verbose=0)
            elif sklearn_model == "LR":
                model = LogisticRegression()
            model.fit(x, y)
            
            pred_optimal_y = model.predict_proba(optimal_x)
            print("pred on pos: ", pred_optimal_y[:, 1])
            
        print("\n\n\n")
        exit()

    return


def extract_tanimoto_similarity_list(ecfp, ecfp_list):
    similarity_list = []
    for ecfp_ in ecfp_list:
        sim = DataStructs.TanimotoSimilarity(ecfp, ecfp_)
        similarity_list.append(sim)
    return similarity_list


def extract_similarity(file_path, sklearn_model_record_each_text):
    smiles_record = defaultdict(list)    
    
    f = open(file_path, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split("\t")
        original_smiles = line[1]
        edited_smiles = line[3]
        alpha = float(line[4])

        smiles_record[original_smiles].append([edited_smiles, alpha])

    saver_path = os.path.join(file_path.replace("edited_SMILES.tsv", "edited_ECFP.json"))
    if os.path.exists(saver_path):
        infile = open(saver_path, 'r')
        fp_record = json.loads(infile.read())
    else:
        fp_record = {}
        for source_smiles, edited_list in smiles_record.items():
            fp_list = []
            source_fp = smiles_to_fps(source_smiles)
            fp_record[source_smiles] = source_fp
            for edited_smiles, alpha in edited_list:    
                edited_mol = Chem.MolFromSmiles(edited_smiles)
                if edited_mol is None:
                    continue
                edited_fp = smiles_to_fps(edited_smiles)
                fp_record[edited_smiles] = edited_fp
        outfile = open(saver_path, 'w')
        json.dump(fp_record, outfile)
    
    smiles_list, fp_list = [], []
    for smiles, fp in fp_record.items():
        smiles_list.append(smiles)
        fp = [float(x) for x in fp]
        fp_list.append(fp)
    fp_list = np.array(fp_list)

    accuracy_list = []
    for model_name in sklearn_model_list:
        accuracy, total_count, invalid_count = 0, 0, 0
        sklearn_model = sklearn_model_record_each_text[model_name]
        
        label_list = sklearn_model.predict_proba(fp_list)[:, 1]
        label_record = {}
        for smiles, label in zip(smiles_list, label_list):
            label_record[smiles] = label

        for source_smiles, edited_list in smiles_record.items():
            source_label = label_record[source_smiles]
            smiles_set = set()
            smiles_set.add(source_smiles)

            edited_label_list = []
            for edited_smiles, alpha in edited_list:
                if edited_smiles not in label_record:
                    continue
                smiles_set.add(edited_smiles)
                edited_label_list.append(label_record[edited_smiles])

            total_count += 1
            if len(smiles_set) == 1:
                invalid_count += 1
                continue
            if np.max(edited_label_list) > source_label:
                accuracy += 1

        total_count -= invalid_count
        accuracy = 100. * accuracy / total_count

        accuracy_list.append(accuracy)
    return accuracy_list


if __name__ == "__main__":
    output_dir = '../../output'
    generation_model = 'MegaMolBART'
    dataset_list = ["ChEMBL"]
    text_id_list = assay_dict.keys()

    sklearn_model_record = extract_ChEMBL_assay(assay_dict)
