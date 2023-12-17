import os
import json
import pandas as pd
from random import sample
from rdkit import Chem
from tqdm import tqdm


if __name__ == "__main__":
    root = "../../../Datasets/PubChem_data"

    CID2text_raw_file = os.path.join(root, "raw/CID2text_raw.json")
    CID2text_file = os.path.join(root, "raw/CID2text.json")
    CID2SMILES_file = os.path.join(root, "raw/CID2SMILES.csv")

    with open(CID2text_raw_file, "r") as f:
        CID2text_raw_data = json.load(f)

    with open(CID2text_file, "r") as f:
        CID2text_data = json.load(f)

    SDF_file = os.path.join(root, "raw/molecules.sdf")
    suppl = Chem.SDMolSupplier(SDF_file)
    CID_list, SMILES_list = [], []
    for mol in tqdm(suppl):
        CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
        CAN_SMILES = mol.GetProp("PUBCHEM_OPENEYE_CAN_SMILES")
        ISO_SMILES = mol.GetProp("PUBCHEM_OPENEYE_ISO_SMILES")
        
        RDKit_mol = Chem.MolFromSmiles(CAN_SMILES)
        if RDKit_mol is None:
            continue
        RDKit_CAN_SMILES = Chem.MolToSmiles(RDKit_mol)

        CID_list.append(CID)
        SMILES_list.append(RDKit_CAN_SMILES)
    df = pd.DataFrame({"CID": CID_list, "SMILES": SMILES_list})
    df.to_csv(CID2SMILES_file)
