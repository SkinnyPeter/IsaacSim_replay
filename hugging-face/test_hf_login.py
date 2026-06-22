#!/usr/bin/env python3
"""
test_hf_login.py

Check that the Hugging Face token is valid and the dataset repo is reachable.

Usage:
    python test_hf_login.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(Path(__file__).resolve().parent / ".env")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

REPO_ID = "Sim2RealB/Sim2RealB_dataset"
REPO_TYPE = "dataset"


def main() -> None:
    # 1. Token present
    if not HF_TOKEN:
        print("[FAIL] HUGGINGFACE_TOKEN is not set.")
        print("       Create hugging-face/.env with:  HUGGINGFACE_TOKEN=hf_your_token_here")
        sys.exit(1)
    print(f"[OK]   Token found ({HF_TOKEN[:8]}...)")

    api = HfApi(token=HF_TOKEN)

    # 2. Token valid
    try:
        user = api.whoami()
        print(f"[OK]   Logged in as: {user['name']} ({user.get('email', 'no email')})")
    except Exception as exc:
        print(f"[FAIL] Authentication failed: {exc}")
        sys.exit(1)

    # 3. Repo reachable
    try:
        info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"[OK]   Repo reachable: {info.id}")
    except Exception as exc:
        print(f"[FAIL] Cannot reach repo {REPO_ID}: {exc}")
        sys.exit(1)

    print("\nAll checks passed. Ready to upload.")


if __name__ == "__main__":
    main()
