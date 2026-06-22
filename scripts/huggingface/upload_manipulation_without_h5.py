#!/usr/bin/env python3
"""
Upload one manipulation folder from data/<manipulation_id> to Hugging Face,
excluding H5 files.

Example:
    python scripts/huggingface/upload_manipulation_without_h5.py 20250804_104715
    python scripts/huggingface/upload_manipulation_without_h5.py 20250804_104715 --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "Sim2RealB/Sim2RealB_dataset"
DEFAULT_REPO_TYPE = "dataset"
ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")

IGNORE_PATTERNS = [
    "*.h5",
    "**/*.h5",
    "*.hdf5",
    "**/*.hdf5",
    "**/.git/**",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
    "*.log",
    "**/*.log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload data/<manipulation_id> to Hugging Face without H5 files."
    )
    parser.add_argument("manipulation_id", help="Manipulation id, e.g. 20250804_104715")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=BASE_DIR / "data",
        help="Local data root containing <manipulation_id>/",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        help=f"Hugging Face repo type. Default: {DEFAULT_REPO_TYPE}",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Remote folder. Default: data/<manipulation_id>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be uploaded without calling Hugging Face.",
    )
    return parser.parse_args()


def validate_manipulation_id(manipulation_id: str) -> None:
    if not ID_PATTERN.fullmatch(manipulation_id):
        raise ValueError(
            f"Invalid manipulation id '{manipulation_id}'. Expected format: YYYYMMDD_HHMMSS"
        )


def should_skip(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5", ".pyc", ".log"}:
        return True
    return "__pycache__" in path.parts or ".git" in path.parts


def list_upload_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and not should_skip(path.relative_to(source_dir))
    )


def main() -> None:
    args = parse_args()
    validate_manipulation_id(args.manipulation_id)

    source_dir = args.data_root / args.manipulation_id
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Manipulation folder not found: {source_dir}")

    path_in_repo = args.path_in_repo or f"data/{args.manipulation_id}"
    upload_files = list_upload_files(source_dir)
    if not upload_files:
        raise RuntimeError(f"No non-H5 files found to upload in {source_dir}")

    print(f"Source: {source_dir}")
    print(f"Repo:   {args.repo_id} ({args.repo_type})")
    print(f"Remote: {path_in_repo}")
    print(f"Files to upload: {len(upload_files)}")
    for path in upload_files:
        print(f"  - {path.relative_to(source_dir)}")

    if args.dry_run:
        print("Dry run complete; no upload performed.")
        return

    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    api = HfApi(token=token)
    who = api.whoami()
    print("Logged in as:", who["name"])

    api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type)
    api.upload_folder(
        folder_path=str(source_dir),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=f"Upload {args.manipulation_id} data without H5",
        ignore_patterns=IGNORE_PATTERNS,
    )
    print("Upload complete")


if __name__ == "__main__":
    main()
