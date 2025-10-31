from huggingface_hub import hf_hub_download

def from_huggingface(modelstr: str):
    repo_name, repo_fpath = modelstr.split(":")
    fpath = hf_hub_download(repo_id=repo_name, filename=repo_fpath)
    return fpath