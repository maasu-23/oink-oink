"""
S3 helpers — the project's data/processed/ tree mirrored to a bucket.

Layout in the bucket mirrors the repo:
    s3://<bucket>/data/processed/circuit_nodes.csv
    s3://<bucket>/data/processed/circuit_adjacency.npz
    s3://<bucket>/data/processed/models/ppo_<variant>_seed<k>.zip
    s3://<bucket>/data/processed/logs/<run>/monitor.monitor.csv

The bucket name comes from PIGBRAIN_BUCKET (see .env.example). Everything
here is optional: training and the supervisor work from local files when the
variable is unset.

Usage:
    python -m oink.s3 push            # upload data/processed/ (skips unchanged)
    python -m oink.s3 pull            # download anything missing locally
"""
import argparse
import os
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data/processed"
BUCKET = os.environ.get("PIGBRAIN_BUCKET")
REGION = os.environ.get("AWS_REGION", "ap-south-1")

# Bulky artefacts that are outputs of the writeup, not inputs to anything.
SKIP_SUFFIXES = (".mp4", ".gif", ".png")


def _client():
    return boto3.client("s3", region_name=REGION)


def s3_key(local: Path) -> str:
    return local.resolve().relative_to(REPO_ROOT).as_posix()


def ensure_bucket(bucket: str):
    s3 = _client()
    existing = {b["Name"] for b in s3.list_buckets()["Buckets"]}
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": REGION})
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={k: True for k in ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")},
        )
        print(f"Created private bucket s3://{bucket} in {REGION}")


def push(bucket: str, root: Path = PROCESSED, include_media: bool = False) -> int:
    s3 = _client()
    remote = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=s3_key(root)):
        for obj in page.get("Contents", []):
            remote[obj["Key"]] = obj["Size"]
    n = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if not include_media and path.suffix in SKIP_SUFFIXES:
            continue
        key = s3_key(path)
        if remote.get(key) == path.stat().st_size:
            continue
        s3.upload_file(str(path), bucket, key)
        n += 1
        print(f"  up  {key}")
    return n


def pull(bucket: str, prefix: str = "data/processed/") -> int:
    s3 = _client()
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            local = REPO_ROOT / obj["Key"]
            if local.exists() and local.stat().st_size == obj["Size"]:
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(local))
            n += 1
            print(f"  down  {obj['Key']}")
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["push", "pull"])
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--include-media", action="store_true", help="Also upload .mp4/.gif/.png")
    args = p.parse_args()
    if not args.bucket:
        raise SystemExit("Set PIGBRAIN_BUCKET (or pass --bucket).")
    ensure_bucket(args.bucket)
    if args.action == "push":
        n = push(args.bucket, include_media=args.include_media)
        print(f"Uploaded {n} file(s) to s3://{args.bucket}/data/processed/")
    else:
        n = pull(args.bucket)
        print(f"Downloaded {n} file(s) from s3://{args.bucket}/data/processed/")


if __name__ == "__main__":
    main()
