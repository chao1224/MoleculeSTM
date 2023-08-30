from tqdm import tqdm
import json
import gzip
import numpy as np

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import multiprocessing
from multiprocessing import Pool
import sys


if __name__ == "__main__":
    datasets_home_folder = "../../../Datasets"

    PubChemSTM_datasets_description_home_folder = "{}/step_01_PubChemSTM_description".format(datasets_home_folder)
    with open("{}/PubChemSTM_data/raw/CID2text.json".format(datasets_home_folder), "r") as f:
        CID2text = json.load(f)
    target_CID_list = set(CID2text.keys())

    PubChemSTM_datasets_input_folder = "{}/step_02_PubChemSTM_SDF".format(datasets_home_folder)
    PubChemSTM_datasets_output_folder = "{}/step_03_PubChemSTM_filtered".format(datasets_home_folder)
    block_size = 500000

    def extract_one_SDF_file(block_id):
        valid_mol_count = 0

        writer = Chem.SDWriter('{}/filtered_{}.sdf'.format(PubChemSTM_datasets_output_folder, block_id))
        start_id = block_id * block_size + 1
        end_id = (block_id + 1) * block_size

        compound_file_name = "Compound_{:09d}_{:09d}.sdf.gz".format(start_id, end_id)
        gzip_loader = gzip.open("{}/{}".format(PubChemSTM_datasets_input_folder, compound_file_name))
        suppl = Chem.ForwardSDMolSupplier(gzip_loader)

        for mol in tqdm(suppl):
            if mol is None:
                continue
            cid = mol.GetProp("PUBCHEM_COMPOUND_CID")

            if cid not in target_CID_list:
                continue

            writer.write(mol)
            valid_mol_count += 1

        print("block id: {}\nfound {}\n\n".format(block_id, valid_mol_count))
        sys.stdout.flush()
        return
    
    num_process = multiprocessing.cpu_count()
    print("{} CPUs".format(num_process))
    num_process = 8
    p = Pool(num_process)

    total_block_num = 325
    block_id_list = np.arange(total_block_num+1)
    with p:
        p.map(extract_one_SDF_file, block_id_list)
