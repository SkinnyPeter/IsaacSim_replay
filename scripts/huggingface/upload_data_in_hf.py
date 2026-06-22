from pathlib import Path
from huggingface_hub import login, HfApi

login()

api = HfApi()
repo_id = "Sim2RealB/Sim2RealB_dataset"
commit_message = "Upload data folder with h5 and trajectories"

who = api.whoami()
print("Logged in as:", who["name"])

info = api.repo_info(repo_id=repo_id, repo_type="dataset")
print("Repo found:", info.id)

# Project root -> data
data_path = Path(__file__).resolve().parents[2] / "data"

api.upload_folder(
    folder_path=str(data_path),
    path_in_repo="data",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message=commit_message,
    ignore_patterns=[
        "**/.git/**",
        "**/__pycache__/**",
        "*.py",
        "*.log",
    ],
)

print("Upload complete")