import argparse
from PubChem_utils import download_and_extract_compound_file


parser = argparse.ArgumentParser()
parser.add_argument("--block_id", type=int, default=0)
args = parser.parse_args()


if __name__ == "__main__":
    datasets_home_folder = "../../../Datasets"

    PubChemSTM_datasets_home_folder = "{}/step_02_PubChemSTM_SDF".format(datasets_home_folder)
    block_id = args.block_id
    block_size = 500000
    start_id = block_id * block_size + 1
    end_id = (block_id + 1) * block_size

    compound_file_name = "Compound_{:09d}_{:09d}.sdf.gz".format(start_id, end_id)
    download_and_extract_compound_file(PubChemSTM_datasets_home_folder, compound_file_name)
