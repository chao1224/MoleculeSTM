from huggingface_hub import HfApi, snapshot_download

api = HfApi()

snapshot_download(repo_id="chao1224/MoleculeSTM", repo_type="model", local_dir='.', allow_patterns="*MoleculeSTM*")
snapshot_download(repo_id="chao1224/MoleculeSTM", repo_type="model", local_dir='.', allow_patterns="*GraphMVP*")
snapshot_download(repo_id="chao1224/MoleculeSTM", repo_type="model", local_dir='.', allow_patterns="*KV*")
