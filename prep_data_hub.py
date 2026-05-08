"""
Upload train/val/test parquets to a HF Hub dataset repo so any future
Colab/RunPod session can pull them in seconds instead of needing scp.

Run once from anywhere that has the parquets locally (Mac or pod):
    python prep_data_hub.py [--repo-id junho5400/svg-finetune-data] [--private]

After upload, fetch on Colab via:
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id='junho5400/svg-finetune-data', filename='train.parquet',
                    repo_type='dataset', local_dir='data/')
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_REPO = "junho5400/svg-finetune-data"
FILES = ["train.parquet", "val.parquet", "test.parquet"]


def main(repo_id: str, private: bool) -> None:
    api = HfApi()

    print(f"Creating dataset repo {repo_id} (private={private})...")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)

    for fname in FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  skip {fname}: not found at {path}")
            continue
        size_mb = path.stat().st_size / 1e6
        print(f"  uploading {fname} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"\nDone. View at https://huggingface.co/datasets/{repo_id}")
    print("Fetch on Colab:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  for f in {FILES}:")
    print(f"      hf_hub_download(repo_id='{repo_id}', filename=f,")
    print(f"                      repo_type='dataset', local_dir='data/')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--private", action="store_true",
                    help="Create as private dataset (default: public)")
    args = ap.parse_args()
    if "HF_TOKEN" not in os.environ:
        print("warning: HF_TOKEN not set; relying on cached credentials from `hf auth login`")
    main(args.repo_id, args.private)
