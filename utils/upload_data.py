from huggingface_hub import HfApi

api=HfApi()
api.upload_file(
    path_or_fileobj="data/vatex_val.zip",
    path_in_repo="vatex_val.zip",
    repo_id="lmms-lab/vatex",
    repo_type="dataset",
)
