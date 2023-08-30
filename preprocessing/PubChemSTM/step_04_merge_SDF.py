from tqdm import tqdm
import json

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


if __name__ == "__main__":
    datasets_home_folder = "../../../Datasets"

    PubChemSTM_datasets_description_home_folder = "{}/step_01_PubChemSTM_description".format(datasets_home_folder)
    with open("{}/PubChemSTM_data/raw/CID2text.json".format(datasets_home_folder), "r") as f:
        CID2text = json.load(f)
    target_CID_list = set(CID2text.keys())
    print('The length of target_CID_list: {}'.format(len(target_CID_list)))

    PubChemSTM_datasets_folder = "{}/step_03_PubChemSTM_filtered".format(datasets_home_folder)
    writer = Chem.SDWriter('{}/PubChemSTM_data/raw/molecules.sdf'.format(datasets_home_folder))
    
    total_block_num = 325
    found_CID_set = set()
    for block_id in range(total_block_num+1):
        compound_file_path = "{}/filtered_{}.sdf".format(PubChemSTM_datasets_folder, block_id)
        try:
            suppl = Chem.SDMolSupplier(compound_file_path)

            for mol in tqdm(suppl):
                writer.write(mol)
                cid = mol.GetProp("PUBCHEM_COMPOUND_CID")
                found_CID_set.add(cid)
        except:
            print("block id: {} with 0 valid SDF file".format(block_id))
            continue

    for CID in target_CID_list:
        if CID not in found_CID_set:
            print("CID: {} not found.".format(CID))
    
    print("In total: {} molecules".format(len(found_CID_set)))