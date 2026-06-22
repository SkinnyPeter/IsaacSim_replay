#!/usr/bin/env python3
"""
Upload one data/<manipulation_id> folder to the Hugging Face dataset.

Examples:
    python scripts/huggingface/upload_manipulation_folder.py 20250804_105512_demo_0_468 --dry-run
    python scripts/huggingface/upload_manipulation_folder.py 20250804_105512_demo_0_468
    python scripts/huggingface/upload_manipulation_folder.py 20250804_105512_demo_0_468 --exclude-h5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "Sim2RealB/Sim2RealB_dataset"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_MANIPULATION_ID = "20250804_105512_demo_0_468"

COMMON_IGNORE_PATTERNS = [
    "**/.git/**",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
    "*.log",
    "**/*.log",
]

H5_IGNORE_PATTERNS = [
    "*.h5",
    "**/*.h5",
    "*.hdf5",
    "**/*.hdf5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a single data/<manipulation_id> folder to Hugging Face."
    )
    parser.add_argument(
        "manipulation_id",
        nargs="?",
        default=DEFAULT_MANIPULATION_ID,
        help=f"Folder under data/. Default: {DEFAULT_MANIPULATION_ID}",
    )
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
        "--exclude-h5",
        action="store_true",
        help="Skip *.h5 and *.hdf5 files if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be uploaded without calling Hugging Face.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message.",
    )
    return parser.parse_args()


def read_dotenv_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("\"'")
    return None


def load_token() -> str | None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    env_paths = [
        BASE_DIR / ".env",
        BASE_DIR / "hugging-face" / ".env",
    ]

    if load_dotenv is not None:
        for env_path in env_paths:
            load_dotenv(env_path)

    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        return token

    for env_path in env_paths:
        token = read_dotenv_value(env_path, "HUGGINGFACE_TOKEN") or read_dotenv_value(env_path, "HF_TOKEN")
        if token:
            return token

    return None


def should_skip(path: Path, exclude_h5: bool) -> bool:
    suffix = path.suffix.lower()
    if suffix in {".pyc", ".log"}:
        return True
    if exclude_h5 and suffix in {".h5", ".hdf5"}:
        return True
    return "__pycache__" in path.parts or ".git" in path.parts


def list_upload_files(source_dir: Path, exclude_h5: bool) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and not should_skip(path.relative_to(source_dir), exclude_h5)
    )


def main() -> None:
    args = parse_args()

    source_dir = args.data_root / args.manipulation_id
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Manipulation folder not found: {source_dir}")

    path_in_repo = args.path_in_repo or f"data/{args.manipulation_id}"
    upload_files = list_upload_files(source_dir, args.exclude_h5)
    if not upload_files:
        raise RuntimeError(f"No uploadable files found in {source_dir}")

    print(f"Source: {source_dir}")
    print(f"Repo:   {args.repo_id} ({args.repo_type})")
    print(f"Remote: {path_in_repo}")
    print(f"Files to upload: {len(upload_files)}")
    for path in upload_files:
        print(f"  - {path.relative_to(source_dir)}")

    if args.dry_run:
        print("Dry run complete; no upload performed.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install huggingface_hub python-dotenv") from exc

    token = load_token()
    api = HfApi(token=token)
    who = api.whoami()
    print("Logged in as:", who["name"])

    info = api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type)
    print("Repo found:", info.id)

    ignore_patterns = list(COMMON_IGNORE_PATTERNS)
    if args.exclude_h5:
        ignore_patterns.extend(H5_IGNORE_PATTERNS)

    commit_message = args.commit_message or f"Upload {args.manipulation_id} data folder"
    api.upload_folder(
        folder_path=str(source_dir),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )
    print("Upload complete")


if __name__ == "__main__":
    main()
