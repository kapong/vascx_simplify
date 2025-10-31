from huggingface_hub import hf_hub_download

def from_huggingface(modelstr: str):
    """Download a model from HuggingFace Hub.
    
    Args:
        modelstr: String in format "repo_name:repo_fpath" where repo_name is the 
                 HuggingFace repository name and repo_fpath is the path to the file
                 within the repository.
    
    Returns:
        str: Local path to the downloaded file.
    
    Example:
        >>> fpath = from_huggingface("username/model-name:model.pt")
    """
    repo_name, repo_fpath = modelstr.split(":")
    fpath = hf_hub_download(repo_id=repo_name, filename=repo_fpath)
    return fpath
