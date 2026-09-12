#!/usr/bin/env python3
"""Synchronize a local compiler-cache directory with an object-storage bucket.

The cache transport is a generic S3-compatible endpoint: configuration
arrives entirely through environment variables so this script stays
agnostic of any particular provider. Transfers run on a small thread pool,
which matters here because the cache is a large collection of small
content-addressed objects.

Environment:
    CACHE_DIR              local cache directory (default: ~/.cache/sccache)
    S3_BUCKET              destination bucket (required)
    S3_ENDPOINT            S3-compatible endpoint (required)
    S3_REGION              signature region (default: us-east-1)
    SYNC_DIRECTION         push (local -> bucket) or pull (bucket -> local)
    DRY_RUN                enabled logs the plan without transferring
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   credentials
"""

import concurrent.futures
import os
import sys

try:
    import botocore.config
    import botocore.session
except ImportError:
    sys.exit("botocore is missing: pip install --user botocore")

CHUNK_ROOT = os.environ.get("CACHE_DIR") or os.path.join(
    os.path.expanduser("~"), ".cache", "sccache"
)
BUCKET = os.environ.get("S3_BUCKET") or sys.exit("S3_BUCKET must be set")
ENDPOINT = os.environ.get("S3_ENDPOINT") or sys.exit("S3_ENDPOINT must be set")
REGION = os.environ.get("S3_REGION", "us-east-1")
DIRECTION = os.environ.get("SYNC_DIRECTION", "push")
DRY_RUN = os.environ.get("DRY_RUN", "enabled") == "enabled"
WORKERS = 8


def client():
    return botocore.session.get_session().create_client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=botocore.config.Config(
            region_name=REGION,
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            max_pool_connections=WORKERS,
        ),
    )


def walk_local(root):
    found = {}
    for base, _, names in os.walk(root):
        for name in names:
            path = os.path.join(base, name)
            found[os.path.relpath(path, root)] = os.path.getsize(path)
    return found


def list_remote(s3, bucket):
    found = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            found[obj["Key"]] = obj["Size"]
    return found


def main():
    if DIRECTION not in ("push", "pull"):
        sys.exit(f"unknown SYNC_DIRECTION: {DIRECTION}")
    s3 = client()
    remote = list_remote(s3, BUCKET)
    local = walk_local(CHUNK_ROOT)
    if DIRECTION == "push":
        plan = [k for k, size in local.items() if remote.get(k) != size]
    else:
        plan = [k for k, size in remote.items() if local.get(k) != size]
    print(f"{DIRECTION}: {len(plan)} object(s) to transfer")

    if DRY_RUN:
        for key in plan[:20]:
            print(f"  {key}")
        if len(plan) > 20:
            print(f"  ... and {len(plan) - 20} more")
        return

    os.makedirs(CHUNK_ROOT, exist_ok=True)

    def transfer(key):
        if DIRECTION == "push":
            with open(os.path.join(CHUNK_ROOT, key), "rb") as fh:
                s3.put_object(Bucket=BUCKET, Key=key, Body=fh)
        else:
            path = os.path.join(CHUNK_ROOT, key)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            with open(path, "wb") as fh:
                fh.write(body)

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, _ in enumerate(
            concurrent.futures.as_completed(pool.submit(transfer, k) for k in plan), 1
        ):
            if done % 200 == 0 or done == len(plan):
                print(f"  {done}/{len(plan)}")


if __name__ == "__main__":
    main()
