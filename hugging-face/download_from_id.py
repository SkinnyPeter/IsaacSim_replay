#!/usr/bin/env python3
"""
download_from_id.py

Check whether dataset files for a manipulation ID exist locally, and
optionally download missing ones from the Hugging Face dataset repo.

Expected ID format:
    YYYYMMDD_HHMMSS  (8 digits + "_" + 6 digits)
Example:
    20250804_104715

Remote dataset structure:
    {manipulation_id}/trajectory_yellow_rubber_duck.npy
    {manipulation_id}/trajectory_purple_bowl.npy
    {manipulation_id}/about.yaml

Local mirror (under <project_root>/data/):
    data/{manipulation_id}/trajectory_yellow_rubber_duck.npy
    data/{manipulation_id}/trajectory_purple_bowl.npy
    data/{manipulation_id}/about.yaml

Authentication:
    HUGGINGFACE_TOKEN is read from the .env file next to this script.
    If unset, the script falls back to the token saved by `hf auth login`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import HfApi, file_exists, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import EntryNotFoundError

# ============================================================================
# Configuration
# ============================================================================

load_dotenv(Path(__file__).resolve().parent / ".env")

DEBUG = False

BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_DATA_DIR = BASE_DIR / "data"

HF_REPO_ID = "Sim2RealB/Sim2RealB_dataset"
HF_REPO_TYPE = "dataset"

HF_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")

# YYYYMMDD_HHMMSS
ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")

TRAJECTORY_FILES = [
    "trajectory_yellow_rubber_duck.npy",
    "trajectory_purple_bowl.npy",
]
ABOUT_FILE = "about.yaml"


# ============================================================================
# Helpers
# ============================================================================

def validate_manipulation_id(manipulation_id: str) -> None:
    if not ID_PATTERN.fullmatch(manipulation_id):
        raise ValueError(
            f"Invalid ID format: '{manipulation_id}'. "
            "Expected format is 'YYYYMMDD_HHMMSS', e.g. '20250804_104715'."
        )


def build_paths(manipulation_id: str) -> dict[str, Path | str]:
    local_dir = LOCAL_DATA_DIR / manipulation_id
    return {
        fname: {
            "local": local_dir / fname,
            "remote": f"{manipulation_id}/{fname}",
        }
        for fname in TRAJECTORY_FILES + [ABOUT_FILE]
    }


def check_hf_auth(token: Optional[str]) -> HfApi:
    if not token:
        raise RuntimeError(
            "No Hugging Face token found.\n"
            "Either set HUGGINGFACE_TOKEN in "
            "3dv-pose-pipeline/hugging-face/.env, "
            "or log in with: hf auth login"
        )

    api = HfApi(token=token)
    try:
        user = api.whoami()
        print(f"Logged in as: {user.get('name', '<unknown>')}")
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication failed. "
            "Check your token or run `hf auth login`."
        ) from exc

    return api


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def remote_file_exists(remote_path: str, token: str) -> bool:
    try:
        return file_exists(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            repo_type=HF_REPO_TYPE,
            token=token,
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Could not check remote file '{remote_path}'. "
            "Verify the repo ID, token, and network access."
        ) from exc


def download_file(remote_path: str, token: str) -> Path:
    """Download remote_path into LOCAL_DATA_DIR preserving its subdirectory."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            repo_type=HF_REPO_TYPE,
            token=token,
            local_dir=str(LOCAL_DATA_DIR),
        )
        return Path(downloaded)
    except EntryNotFoundError as exc:
        raise RuntimeError(f"Remote file not found: {remote_path}") from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Failed to download '{remote_path}' from Hugging Face."
        ) from exc


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether manipulation files exist locally and optionally "
            "download missing ones from Hugging Face."
        )
    )
    parser.add_argument(
        "manipulation_id",
        help="Manipulation ID, e.g. 20250804_104715",
    )
    args = parser.parse_args()

    manipulation_id = args.manipulation_id

    try:
        validate_manipulation_id(manipulation_id)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    paths = build_paths(manipulation_id)

    if DEBUG:
        api = HfApi(token=HF_TOKEN)
        print("Files in HF repo:")
        for f in api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE):
            print(f"  {f}")

    # Check local presence
    missing_locally = {
        fname: info
        for fname, info in paths.items()
        if not info["local"].is_file()
    }

    if not missing_locally:
        print(
            f"All files for '{manipulation_id}' are already present locally:"
        )
        for info in paths.values():
            print(f"  {info['local']}")
        return

    print(f"Missing local files for '{manipulation_id}':")
    for fname in missing_locally:
        print(f"  {paths[fname]['local']}")

    token = HF_TOKEN

    try:
        check_hf_auth(token)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    assert token is not None

    # Check which missing files exist remotely
    downloadable: dict[str, str] = {}
    not_found: list[str] = []

    for fname, info in missing_locally.items():
        remote = info["remote"]
        assert isinstance(remote, str)
        if remote_file_exists(remote, token):
            print(f"  Remote found: {remote}")
            downloadable[fname] = remote
        else:
            print(f"  Remote not found: {remote}")
            not_found.append(fname)

    if not_found:
        print(f"Not available remotely: {', '.join(not_found)}")

    if not downloadable:
        print("Nothing to download.")
        return

    if not ask_yes_no(
        f"Download {len(downloadable)} file(s) for '{manipulation_id}'?"
    ):
        print("Download cancelled.")
        return

    for fname, remote in downloadable.items():
        local_path = download_file(remote, token)
        print(f"Downloaded: {local_path}")

    print("Done.")


if __name__ == "__main__":
    main()
