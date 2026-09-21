"""
Phase 3 on SageMaker — launch the PPO comparison as SageMaker training jobs.

One job per (variant, seed). Each job runs oink/sagemaker_entry.py inside the
managed PyTorch CPU container, reads the circuit files from S3, and writes
model.tar.gz (policy + Monitor CSV) back to S3. `--wait` polls until every
job finishes and pulls the artefacts into data/processed/ so the existing
compare_results.py works unchanged.

Usage:
    python scripts/sagemaker_train.py launch --variant connectome baseline connectome_randinit --seed 0 1 2
    python scripts/sagemaker_train.py status
    python scripts/sagemaker_train.py fetch            # download finished jobs' outputs

Requires PIGBRAIN_BUCKET and an ml.m5.xlarge training quota > 0 in AWS_REGION.
"""
import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parents[1]
REGION = os.environ.get("AWS_REGION", "ap-south-1")
BUCKET = os.environ.get("PIGBRAIN_BUCKET")
ROLE_NAME = "PigBrainSageMakerRole"
JOB_PREFIX = "pigbrain"
STATE_FILE = REPO_ROOT / "data/processed/sagemaker_jobs.json"


# ----------------------------------------------------------------------------
# IAM role SageMaker assumes to pull the code/data and push results
# ----------------------------------------------------------------------------
def ensure_role(bucket: str) -> str:
    iam = boto3.client("iam")
    try:
        return iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    arn = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust), Description="PigBrain SageMaker training jobs")["Role"]["Arn"]
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess")
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket}"},
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"arn:aws:s3:::{bucket}/*"},
        ],
    }
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="PigBrainBucketAccess", PolicyDocument=json.dumps(bucket_policy))
    print(f"Created IAM role {ROLE_NAME}; waiting for it to propagate...")
    time.sleep(12)
    return arn


# ----------------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------------
def stage_source(bucket: str, stamp: str) -> str:
    """Tar up only the package (+ its requirements) -- never data/ -- and upload it.

    This is SageMaker "script mode": the PyTorch container downloads the
    tarball named by the sagemaker_submit_directory hyperparameter, pip
    installs its requirements.txt, and runs sagemaker_program.
    """
    tmp = Path(tempfile.mkdtemp(prefix="pigbrain_src_"))
    shutil.copytree(REPO_ROOT / "oink", tmp / "oink", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO_ROOT / "oink/requirements.txt", tmp / "requirements.txt")
    tar_path = tmp / "sourcedir.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(tmp / "oink", arcname="oink")
        tar.add(tmp / "requirements.txt", arcname="requirements.txt")
    key = f"sagemaker/code/{stamp}/sourcedir.tar.gz"
    boto3.client("s3", region_name=REGION).upload_file(str(tar_path), bucket, key)
    shutil.rmtree(tmp, ignore_errors=True)
    return f"s3://{bucket}/{key}"


def upload_circuit(bucket: str) -> str:
    s3 = boto3.client("s3", region_name=REGION)
    for name in ("circuit_nodes.csv", "circuit_adjacency.npz"):
        s3.upload_file(str(REPO_ROOT / "data/processed" / name), bucket, f"sagemaker/input/circuit/{name}")
    return f"s3://{bucket}/sagemaker/input/circuit/"


def training_image(framework_version: str) -> str:
    try:  # the SDK knows the per-region registry; fall back to the documented ECR path
        from sagemaker.core import image_uris

        return image_uris.retrieve("pytorch", REGION, version=framework_version, py_version="py312",
                                   instance_type="ml.m5.xlarge", image_scope="training")
    except Exception:
        return f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:{framework_version}-cpu-py312"


def launch(args):
    sm = boto3.client("sagemaker", region_name=REGION)
    role = ensure_role(args.bucket)
    circuit_uri = upload_circuit(args.bucket)
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M")
    code_uri = stage_source(args.bucket, stamp)
    image = training_image(args.framework_version)
    print(f"image: {image}\ncode:  {code_uri}")

    jobs = load_state()
    for variant in args.variant:
        for seed in args.seed:
            job_name = f"{JOB_PREFIX}-{variant.replace('_', '-')}-s{seed}-{stamp}"
            spec = dict(
                TrainingJobName=job_name,
                RoleArn=role,
                AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
                HyperParameters={  # script-mode HPs are JSON-encoded strings
                    "sagemaker_program": json.dumps("oink/sagemaker_entry.py"),
                    "sagemaker_submit_directory": json.dumps(code_uri),
                    "sagemaker_region": json.dumps(REGION),
                    "sagemaker_container_log_level": json.dumps(20),
                    "variant": json.dumps(variant),
                    "seed": json.dumps(seed),
                    "timesteps": json.dumps(args.timesteps),
                    **({"checkpoint-every": json.dumps(args.checkpoint_every)} if args.checkpoint_every else {}),
                },
                InputDataConfig=[{
                    "ChannelName": "circuit",
                    "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": circuit_uri, "S3DataDistributionType": "FullyReplicated"}},
                }],
                OutputDataConfig={"S3OutputPath": f"s3://{args.bucket}/sagemaker/output/"},
                ResourceConfig={"InstanceType": args.instance_type, "InstanceCount": 1, "VolumeSizeInGB": 10},
                StoppingCondition={"MaxRuntimeInSeconds": args.max_run_hours * 3600},
                Tags=[{"Key": "project", "Value": "pigbrain"}],
            )
            if args.spot:
                spec["EnableManagedSpotTraining"] = True
                spec["StoppingCondition"]["MaxWaitTimeInSeconds"] = args.max_run_hours * 3600
            sm.create_training_job(**spec)
            jobs[job_name] = {"variant": variant, "seed": seed, "timesteps": args.timesteps, "launched": stamp,
                              "checkpoint_every": args.checkpoint_every}
            print(f"launched {job_name}")
    save_state(jobs)
    print(f"{len(args.variant) * len(args.seed)} job(s) launched. Track with: python scripts/sagemaker_train.py status")


# ----------------------------------------------------------------------------
# Status / fetch
# ----------------------------------------------------------------------------
def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(jobs: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(jobs, indent=2))


def describe_all():
    sm = boto3.client("sagemaker", region_name=REGION)
    for name, meta in load_state().items():
        d = sm.describe_training_job(TrainingJobName=name)
        yield name, meta, d


def status(args):
    for name, meta, d in describe_all():
        secs = d.get("TrainingTimeInSeconds", 0)
        extra = f"  ({d.get('FailureReason', '')[:80]})" if d["TrainingJobStatus"] == "Failed" else ""
        print(f"{d['TrainingJobStatus']:<10} {d.get('SecondaryStatus', ''):<12} {secs / 3600:5.2f}h  {name}{extra}")


def fetch(args):
    s3 = boto3.client("s3", region_name=REGION)
    for name, meta, d in describe_all():
        if d["TrainingJobStatus"] != "Completed":
            continue
        if args.job and not any(name.startswith(j) for j in args.job):
            continue
        uri = d["ModelArtifacts"]["S3ModelArtifacts"]
        bucket, key = uri[5:].split("/", 1)
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "model.tar.gz"
            s3.download_file(bucket, key, str(tar_path))
            with tarfile.open(tar_path) as tar:
                tar.extractall(args.dest, filter="data")
        print(f"fetched {name} -> {args.dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", default=BUCKET)
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("launch")
    l.add_argument("--variant", nargs="+", default=["connectome", "baseline"])
    l.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    l.add_argument("--timesteps", type=int, default=150_000)
    l.add_argument("--instance-type", default="ml.m5.xlarge")
    l.add_argument("--framework-version", default="2.6")
    l.add_argument("--max-run-hours", type=int, default=6)
    l.add_argument("--spot", action="store_true", help="Use managed spot (cheaper, may be interrupted).")
    l.add_argument("--checkpoint-every", type=int, default=0, help="Save intermediate policies every N steps.")
    sub.add_parser("status")
    f = sub.add_parser("fetch")
    f.add_argument("--dest", type=Path, default=REPO_ROOT / "data/processed/sagemaker")
    f.add_argument("--job", nargs="+", default=None, help="Only fetch jobs whose name starts with one of these.")
    args = p.parse_args()
    if not args.bucket:
        raise SystemExit("Set PIGBRAIN_BUCKET (or pass --bucket).")
    {"launch": launch, "status": status, "fetch": fetch}[args.cmd](args)


if __name__ == "__main__":
    main()
