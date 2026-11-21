"""Sum the GPU time of this project's Slurm jobs (allocation time x GPUs) with sacct.

Usage: python scripts/gpu_hours.py --since 2026-11-21 [--out results/compute.json]
"""

import argparse
import json
import re
import subprocess

JOB_NAMES = "llm-jepa-sigreg,llm-jepa-sigreg-worker"


def seconds(elapsed: str) -> int:
    days, _, clock = elapsed.rpartition("-")
    h, m, s = (int(x) for x in clock.split(":"))
    return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--since", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            f"--name={JOB_NAMES}",
            f"--starttime={args.since}",
            "--format=JobID,JobName,Elapsed,State,AllocTRES",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    jobs = []
    for line in out.splitlines():
        job_id, name, elapsed, state, tres = line.split("|")
        gpus = re.search(r"gres/gpu=(\d+)", tres)
        n_gpu = int(gpus.group(1)) if gpus else 0
        jobs.append(
            {
                "job": job_id,
                "name": name,
                "state": state,
                "gpus": n_gpu,
                "gpu_hours": round(n_gpu * seconds(elapsed) / 3600, 3),
            }
        )
    report = {
        "since": args.since,
        "jobs": jobs,
        "total_gpu_hours": round(sum(j["gpu_hours"] for j in jobs), 2),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
