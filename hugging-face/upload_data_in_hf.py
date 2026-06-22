#!/usr/bin/env python3
"""
upload_data_in_hf.py

Upload a single manipulation's pipeline outputs to the Hugging Face dataset.

Dataset structure written to HF:
    {manipulation_id}/object_initial_pose.npy        ← output/{id}/object_initial_pose.npy
    {manipulation_id}/container_initial_pose.npy     ← output/{id}/container_initial_pose.npy
    {manipulation_id}/object_states.csv              ← output/{id}/object_states.csv
    {manipulation_id}/container_states.csv           ← output/{id}/container_states.csv
    {manipulation_id}/about.yaml                     ← output/{id}/about.yaml (auto-created if absent)

Usage (pipeline-integrated, via run_pipeline.py --steps ...,6):
    python upload_data_in_hf.py --config config/pipeline.yaml

Usage (standalone):
    python upload_data_in_hf.py <manipulation_id>
    python upload_data_in_hf.py <manipulation_id> --object yellow_rubber_duck --container purple_bowl
    python upload_data_in_hf.py <manipulation_id> --source-dir /path/to/output/<id>
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

PIPELINE_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(Path(__file__).resolve().parent / ".env")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

REPO_ID = "Sim2RealB/Sim2RealB_dataset"
REPO_TYPE = "dataset"


def get_api() -> HfApi:
    if not HF_TOKEN:
        print("Error: HUGGINGFACE_TOKEN not set in hugging-face/.env")
        sys.exit(1)
    return HfApi(token=HF_TOKEN)


def check_login(api: HfApi) -> str:
    try:
        user = api.whoami()
        name = user["name"]
        print(f"[OK] Logged in as: {name}")
        return name
    except Exception as exc:
        print(f"[ERROR] Authentication failed: {exc}")
        sys.exit(1)


def upload_file(api: HfApi, local_path: Path, path_in_repo: str, commit_message: str) -> None:
    print(f"  {local_path.name} → {path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=commit_message,
    )


def do_upload(manipulation_id: str, object_name: str, container_name: str, source_dir: Path) -> None:
    api = get_api()
    check_login(api)

    try:
        info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"[OK] Repo: {info.id}")
    except Exception as exc:
        print(f"[ERROR] Could not reach repo {REPO_ID}: {exc}")
        sys.exit(1)

    obj_pose_path  = source_dir / "object_initial_pose.npy"
    con_pose_path  = source_dir / "container_initial_pose.npy"
    obj_states_path = source_dir / "object_states.csv"
    con_states_path = source_dir / "container_states.csv"
    about_path     = source_dir / "about.yaml"
    obj_traj_path  = source_dir / "object_trajectory.npy"
    con_traj_path  = source_dir / "container_trajectory.npy"

    missing = [str(p) for p in [obj_pose_path, con_pose_path, obj_states_path, con_states_path]
               if not p.exists()]
    if missing:
        print("[ERROR] Required files not found:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    if not about_path.exists():
        about_path.write_text(
            f"manipulation_id: {manipulation_id}\n"
            f"object: {object_name}\n"
            f"container: {container_name}\n"
        )
        print(f"[INFO] Created {about_path.name}")

    commit_msg = f"Upload {manipulation_id} ({object_name}, {container_name})"
    print(f"\nUploading to {REPO_ID} under {manipulation_id}/")

    upload_file(api, obj_pose_path,   f"{manipulation_id}/object_initial_pose.npy",    commit_msg)
    upload_file(api, con_pose_path,   f"{manipulation_id}/container_initial_pose.npy", commit_msg)
    upload_file(api, obj_states_path, f"{manipulation_id}/object_states.csv",          commit_msg)
    upload_file(api, con_states_path, f"{manipulation_id}/container_states.csv",       commit_msg)
    upload_file(api, about_path,      f"{manipulation_id}/about.yaml",                 commit_msg)

    for traj_path, repo_name in [
        (obj_traj_path, f"{manipulation_id}/object_trajectory.npy"),
        (con_traj_path, f"{manipulation_id}/container_trajectory.npy"),
    ]:
        if traj_path.exists():
            upload_file(api, traj_path, repo_name, commit_msg)
        else:
            print(f"[WARN] {traj_path.name} not found — skipping (run step 7 to generate it)")

    print("\n[OK] Upload complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload pipeline2 outputs to the Hugging Face dataset."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to pipeline.yaml; reads manipulation_id, object, and container names from it.",
    )
    parser.add_argument(
        "manipulation_id", nargs="?", default=None,
        help="Manipulation ID for standalone use, e.g. 20250804_104715.",
    )
    parser.add_argument(
        "--object", default=None, dest="object_name",
        help="Object name (standalone mode). Defaults to value in config or 'yellow_rubber_duck'.",
    )
    parser.add_argument(
        "--container", default=None, dest="container_name",
        help="Container name (standalone mode). Defaults to value in config or 'purple_bowl'.",
    )
    parser.add_argument(
        "--source-dir", default=None,
        help="Override the output directory. Defaults to output/<manipulation_id>.",
    )
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        manipulation_id = str(cfg["manipulation_id"])
        object_name    = args.object_name    or cfg["object"]["name"]
        container_name = args.container_name or cfg["container"]["name"]
    elif args.manipulation_id:
        manipulation_id = args.manipulation_id
        object_name    = args.object_name    or "yellow_rubber_duck"
        container_name = args.container_name or "purple_bowl"
    else:
        parser.error("Provide either --config or a manipulation_id positional argument.")

    source_dir = (
        Path(args.source_dir) if args.source_dir
        else PIPELINE_ROOT / "output" / manipulation_id
    )
    if not source_dir.is_dir():
        print(f"[ERROR] Source directory not found: {source_dir}")
        sys.exit(1)

    do_upload(manipulation_id, object_name, container_name, source_dir)


if __name__ == "__main__":
    main()
