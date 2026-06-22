#!/usr/bin/env python3
"""
download_from_id.py

Check whether dataset files corresponding to a sample ID already exist locally,
and optionally download any missing files from a private Hugging Face dataset
repository.

Expected ID format:
    8digits_6digits
Example:
    10125604_895734

Behavior:
1. Validate the ID format.
2. Check whether the following local files exist:
   - BASE_DIR / "data" / "h5" / "<id>.h5"
   - BASE_DIR / "data" / "trajectories" / "<id>_trajectory.npy"
3. If both exist locally, print that the data is already available in the data folder.
4. If one or both files are missing locally, check whether the missing file(s) exist
   in the Hugging Face dataset repository.
5. If file(s) exist remotely, ask the user whether they want to download them.
6. Download only the requested and missing file(s).
7. If a file is not found remotely, print a clear message.

Authentication:
- If HF_TOKEN is set below, it is used directly for Hugging Face API calls.
- Otherwise, the script tries to use a locally saved Hugging Face login token.

"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


from huggingface_hub import HfApi, file_exists, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import EntryNotFoundError



# ============================================================================
# Global configuration
# ============================================================================

DEBUG = False
# Base directory of your project.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Hugging Face repository configuration.
HF_REPO_ID = "Sim2RealB/Sim2RealB_dataset"
HF_REPO_TYPE = "dataset"

# Optional token for private repo access.
# Leave as None in the .env to use the token from `hf auth login`.
from dotenv import load_dotenv
import os
load_dotenv() # Loads .env where the secret key is stored
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Local folder structure.
LOCAL_DATA_DIR = BASE_DIR / "data"
LOCAL_H5_DIR = LOCAL_DATA_DIR / "h5"
LOCAL_TRAJ_DIR = LOCAL_DATA_DIR / "trajectories"

# Remote folder structure inside the Hugging Face repo.
REMOTE_H5_DIR = "data/h5"
REMOTE_TRAJ_DIR = "data/trajectories"

# Expected sample ID format: 8 digits + "_" + 6 digits.
ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")


# ============================================================================
# Helper functions
# ============================================================================

def validate_sample_id(sample_id: str) -> None:
    """
    Validate the sample ID format.

    Parameters
    ----------
    sample_id : str
        The dataset sample ID.

    Raises
    ------
    ValueError
        If the ID does not match the expected format.
    """
    if not ID_PATTERN.fullmatch(sample_id):
        raise ValueError(
            f"Invalid ID format: '{sample_id}'. "
            "Expected format is '8digits_6digits', e.g. '10125604_895734'."
        )


def build_paths(sample_id: str) -> dict[str, Path | str]:
    """
    Build local and remote paths associated with the given sample ID.

    Parameters
    ----------
    sample_id : str
        The dataset sample ID.

    Returns
    -------
    dict[str, Path | str]
        Dictionary containing all relevant local and remote paths.
    """
    h5_name = f"{sample_id}.h5"
    traj_name = f"{sample_id}_trajectory.npy"

    return {
        "local_h5": LOCAL_H5_DIR / h5_name,
        "local_traj": LOCAL_TRAJ_DIR / traj_name,
        "remote_h5": f"{REMOTE_H5_DIR}/{h5_name}",
        "remote_traj": f"{REMOTE_TRAJ_DIR}/{traj_name}",
    }

def check_hf_auth(token: Optional[str]) -> HfApi:
    """
    Validate that Hugging Face authentication is available and usable.

    Parameters
    ----------
    token : Optional[str]
        Token to use. If None, authentication is considered unavailable.

    Returns
    -------
    HfApi
        An authenticated HfApi client.

    Raises
    ------
    RuntimeError
        If no token is available or the token is invalid.
    """
    if not token:
        raise RuntimeError(
            "No Hugging Face token found.\n"
            "Either:\n"
            "  - set HF_TOKEN in the script, or\n"
            "  - log in locally with: hf auth login\n"
            "This is required because the dataset repository is private."
        )

    api = HfApi(token=token)

    try:
        user = api.whoami()
        username = user.get("name", "<unknown>")
        print(f"Hugging Face authentication available. Logged in as: {username}")
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication failed. "
            "Please check your token or run `hf auth login`."
        ) from exc

    return api


def ask_yes_no(question: str) -> bool:
    """
    Ask a yes/no question on the command line.

    Parameters
    ----------
    question : str
        The question to ask the user.

    Returns
    -------
    bool
        True if the user answered yes, False otherwise.
    """
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def get_repo_file_set(token: str) -> set[str]:
    """
    List the repo files
    """
    api = HfApi(token=token)
    files = api.list_repo_files(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
    )
    return set(files)


def remote_file_exists(remote_path: str, token: str) -> bool:
    """
    Check whether a file exists in the remote Hugging Face repository.

    Parameters
    ----------
    remote_path : str
        Path to the file inside the Hugging Face repository.
    token : str
        Hugging Face access token.

    Returns
    -------
    bool
        True if the file exists remotely, False otherwise.

    Raises
    ------
    RuntimeError
        If the repository cannot be accessed due to authentication/network issues.
    """
    try:
        return file_exists(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            repo_type=HF_REPO_TYPE,
            token=token,
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Could not check remote file existence for '{remote_path}'. "
            "Please verify the repository ID, token, and network access."
        ) from exc


def download_file(remote_path: str, local_dir: Path, token: str) -> Path:
    """
    Download a file from the Hugging Face repository into a local directory.

    Parameters
    ----------
    remote_path : str
        Path of the file inside the Hugging Face repository.
    local_dir : Path
        Local directory where the file should be stored.
    token : str
        Hugging Face access token.

    Returns
    -------
    Path
        Path to the downloaded local file.

    Raises
    ------
    RuntimeError
        If the download fails.
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            repo_type=HF_REPO_TYPE,
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        return Path(downloaded_path)
    except EntryNotFoundError as exc:
        raise RuntimeError(f"Remote file not found: {remote_path}") from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Failed to download '{remote_path}' from Hugging Face."
        ) from exc


# ============================================================================
# Main logic
# ============================================================================

def main() -> None:
    """
    Entry point of the script.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Check whether files associated with a sample ID exist locally and, "
            "if needed, offer to download missing files from Hugging Face."
        )
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="Sample ID in the format '8digits_6digits', e.g. 10125604_895734",
    )
    args = parser.parse_args()

    sample_id = args.sample_id

    try:
        validate_sample_id(sample_id)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    paths = build_paths(sample_id)

    local_h5_path = paths["local_h5"]
    local_traj_path = paths["local_traj"]
    remote_h5_path = paths["remote_h5"]
    remote_traj_path = paths["remote_traj"]

    assert isinstance(local_h5_path, Path)
    assert isinstance(local_traj_path, Path)
    assert isinstance(remote_h5_path, str)
    assert isinstance(remote_traj_path, str)

    local_h5_exists = local_h5_path.is_file()
    local_traj_exists = local_traj_path.is_file()

    # Case 1: both files already exist locally.
    if local_h5_exists and local_traj_exists:
        print(
            f"Data corresponding to ID '{sample_id}' is already present in the data folder."
        )
        print(f"  - {local_h5_path}")
        print(f"  - {local_traj_path}")
        return

    # Report missing local files.
    print(f"Data for ID '{sample_id}' is not fully present in the local data folder.")
    if not local_h5_exists:
        print(f"  - Missing local file: {local_h5_path}")
    if not local_traj_exists:
        print(f"  - Missing local file: {local_traj_path}")

    # We only need remote access if something is missing locally.
    token = HF_TOKEN

    try:
        _api = check_hf_auth(token)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    assert token is not None

    if DEBUG:
        print("Listing files from HF repo")
        # List all files inside the hg repo (debugging)
        repo_files = get_repo_file_set(token)
        for file in repo_files:
            print(f"  - {file}")

    # Check missing files on the remote repository.
    remote_h5_exists = local_h5_exists or remote_file_exists(remote_h5_path, token)
    remote_traj_exists = local_traj_exists or remote_file_exists(remote_traj_path, token)

    # Report remote availability.
    if not local_h5_exists:
        if remote_h5_exists:
            print(f"Remote file found: {remote_h5_path}")
        else:
            print(f"Remote file not found: {remote_h5_path}")

    if not local_traj_exists:
        if remote_traj_exists:
            print(f"Remote file found: {remote_traj_path}")
        else:
            print(f"Remote file not found: {remote_traj_path}")

    # Nothing can be downloaded.
    if (not local_h5_exists and not remote_h5_exists) and (
        not local_traj_exists and not remote_traj_exists
    ):
        print("Neither missing file was found in the Hugging Face repository.")
        return

    # Determine what is downloadable.
    can_download_h5 = (not local_h5_exists) and remote_h5_exists
    can_download_traj = (not local_traj_exists) and remote_traj_exists

    # Ask user what to do depending on the available files.
    if can_download_h5 and can_download_traj:
        should_download = ask_yes_no(
            "Both missing files are available in the Hugging Face repository. "
            "Would you like to download them?"
        )
        if not should_download:
            print("Download cancelled by user.")
            return
    elif can_download_h5 and not can_download_traj:
        should_download = ask_yes_no(
            "Only the H5 file is available in the Hugging Face repository. "
            "Would you like to download the h5 file anyway?"
        )
        if not should_download:
            print("Download cancelled by user.")
            return
    elif can_download_traj and not can_download_h5:
        should_download = ask_yes_no(
            "Only the trajectory file is available in the Hugging Face repository. "
            "Would you like to download the trajectory file anyway?"
        )
        if not should_download:
            print("Download cancelled by user.")
            return
    else:
        print("No missing local file is available for download from the repository.")
        return

    # Download selected files.
    if can_download_h5:
        downloaded_h5 = download_file(remote_h5_path, BASE_DIR, token)
        print(f"Downloaded H5 file to: {downloaded_h5}")

    if can_download_traj:
        downloaded_traj = download_file(remote_traj_path, BASE_DIR, token)
        print(f"Downloaded trajectory file to: {downloaded_traj}")

    print("Done.")


if __name__ == "__main__":
    main()
