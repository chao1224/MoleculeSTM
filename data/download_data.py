from huggingface_hub import HfApi, snapshot_download

api = HfApi()

snapshot_download(repo_id="chao1224/MoleculeSTM", repo_type="dataset", local_dir='.')
