from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_OUT = ROOT / "data" / "raw_online" / "github_issue_candidates.jsonl"
CSV_OUT = ROOT / "data" / "processed" / "online_candidate_issue_pool.csv"
REPORT = ROOT / "reports" / "online_candidate_collection.md"

REPOS = [
    "triton-lang/triton",
    "pytorch/pytorch",
    "cupy/cupy",
    "jax-ml/jax",
    "numba/numba",
    "rapidsai/cudf",
    "rapidsai/cuml",
    "tensorflow/tensorflow",
    "apache/tvm",
    "NVIDIA/cutlass",
    "NVIDIA/cccl",
]

QUERY_TERMS = [
    "nan",
    "inf",
    "overflow",
    "underflow",
    "precision",
    "tolerance",
    "incorrect result",
    "wrong result",
    "dtype",
    "float16",
    "bfloat16",
    "fp16",
    "fp8",
    "cuda compile",
    "triton compile",
    "segfault cuda",
    "deterministic cuda",
    "allclose cuda",
    "silent correctness",
    "gpu numerical",
]

FAILURE_PATTERNS = {
    "nan_inf": re.compile(r"\b(nan|inf|infinite|nonfinite|non-finite)\b", re.I),
    "overflow_underflow": re.compile(r"\b(over[- ]?flow|under[- ]?flow|saturat|wraparound)\b", re.I),
    "precision_tolerance": re.compile(
        r"\b(precision|rounding|tolerance|allclose|rtol|atol|wrong result|incorrect result|incorrect output|mismatch|silent correctness|deterministic|nondeterministic)\b",
        re.I,
    ),
    "dtype_casting": re.compile(
        r"\b(dtype|data type|cast|casting|promotion|float8|float16|float32|float64|bfloat16|bf16|fp8|fp16|fp32|fp64|int64|int32|complex64|complex128)\b",
        re.I,
    ),
    "crash_compile": re.compile(
        r"\b(segfault|crash|assert|exception|compile error|compilation|fails to compile|internal error|llvm|ptxas|nvvm)\b",
        re.I,
    ),
    "performance_only": re.compile(r"\b(slow|slower|performance|tflops|throughput|benchmark|regression|latency)\b", re.I),
}

CAUSE_PATTERNS = {
    "memory_mask_bounds": re.compile(r"\b(mask|out[- ]of[- ]bounds|boundary|stride|descriptor|pointer|offset|broadcast|shape)\b", re.I),
    "compiler_codegen": re.compile(r"\b(compiler|codegen|lowering|ptx|llvm|inductor|xla|nightly|fusion|fused|kernel generation|nvvm)\b", re.I),
    "async_race_ordering": re.compile(r"\b(async|synchroni[sz]e|race|stream|barrier|order|nondeterministic|deterministic|flaky)\b", re.I),
    "hardware_backend": re.compile(r"\b(cuda|gpu|h100|h200|b200|a100|rtx|hopper|rocm|xla|mps|sm[0-9]|nccl)\b", re.I),
    "reduction_accumulation": re.compile(r"\b(reduction|sum|accumulat|softmax|attention|matmul|dot|exp|log|norm)\b", re.I),
    "api_semantics": re.compile(r"\b(api|expected behavior|semantics|unsupported|feature request)\b", re.I),
    "environment_configuration": re.compile(r"\b(version|driver|install|build|windows|linux|conda|pip|environment)\b", re.I),
}


def run_gh_search(repo: str, term: str, limit: int) -> list[dict[str, Any]]:
    args = [
        "gh",
        "search",
        "issues",
        term,
        "--repo",
        repo,
        "--limit",
        str(limit),
        "--json",
        "number,title,url,state,createdAt,updatedAt,labels,body",
    ]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout or "[]")


def clean(value: str, limit: int = 1200) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def labels_from_patterns(text: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def primary_failure_label(text: str) -> str:
    matches = set(labels_from_patterns(text, FAILURE_PATTERNS))
    for label in [
        "nan_inf",
        "overflow_underflow",
        "precision_tolerance",
        "dtype_casting",
        "crash_compile",
        "performance_only",
    ]:
        if label in matches:
            return label
    return "needs_review"


def priority_score(row: dict[str, str]) -> int:
    text = f"{row['title']} {row['body_excerpt']} {row['github_labels']}".lower()
    score = 0
    for token in ["cuda", "gpu", "triton", "kernel", "inductor", "xla", "nccl", "ptx", "nvvm"]:
        score += 2 if token in text else 0
    for token in ["nan", "inf", "wrong result", "incorrect", "mismatch", "dtype", "overflow", "precision"]:
        score += 3 if token in text else 0
    if row["candidate_primary_failure"] == "needs_review":
        score -= 3
    if row["candidate_primary_failure"] == "performance_only":
        score -= 1
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public GitHub issue candidates for GPU-NFBench expansion.")
    parser.add_argument("--per-query", type=int, default=25)
    parser.add_argument("--max-total", type=int, default=1400)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--repos", nargs="*", default=REPOS)
    parser.add_argument("--terms", nargs="*", default=QUERY_TERMS)
    parser.add_argument("--raw-out", type=Path, default=RAW_OUT)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    raw_out = args.raw_out if args.raw_out.is_absolute() else ROOT / args.raw_out
    csv_out = args.csv_out if args.csv_out.is_absolute() else ROOT / args.csv_out
    report = args.report if args.report.is_absolute() else ROOT / args.report

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    query_count = 0

    with raw_out.open("w", encoding="utf-8") as raw_fh:
        for repo in args.repos:
            for term in args.terms:
                query_count += 1
                try:
                    records = run_gh_search(repo, term, args.per_query)
                except RuntimeError as exc:
                    errors.append(f"{repo} :: {term} :: {exc}")
                    continue
                for item in records:
                    item["_source_repo"] = repo
                    item["_source_query"] = term
                    raw_fh.write(json.dumps(item, sort_keys=True) + "\n")

                    url = str(item.get("url") or "")
                    if not url:
                        continue
                    labels = [label.get("name", "") for label in item.get("labels", []) if isinstance(label, dict)]
                    title = str(item.get("title") or "")
                    body = str(item.get("body") or "")
                    text = f"{title}\n{' '.join(labels)}\n{body}"
                    failure_labels = labels_from_patterns(text, FAILURE_PATTERNS)
                    cause_labels = labels_from_patterns(text, CAUSE_PATTERNS)
                    candidate = {
                        "source": "github_search",
                        "source_query": term,
                        "repository": repo,
                        "issue_number": str(item.get("number") or ""),
                        "title": clean(title, 300),
                        "url": url,
                        "state": str(item.get("state") or ""),
                        "created_at": str(item.get("createdAt") or ""),
                        "updated_at": str(item.get("updatedAt") or ""),
                        "github_labels": "|".join(label for label in labels if label),
                        "candidate_failure_labels": "|".join(failure_labels) if failure_labels else "needs_review",
                        "candidate_primary_failure": primary_failure_label(text),
                        "candidate_cause_labels": "|".join(cause_labels) if cause_labels else "unknown",
                        "body_excerpt": clean(body, 1200),
                    }
                    old = seen.get(url)
                    if old is None or priority_score(candidate) > priority_score(old):
                        seen[url] = candidate
                print(f"{query_count}: {repo} {term} -> {len(records)}")
                if len(seen) >= args.max_total:
                    break
                if args.sleep:
                    time.sleep(args.sleep)
            if len(seen) >= args.max_total:
                break

    rows = sorted(seen.values(), key=lambda row: (-priority_score(row), row["repository"], row["url"]))[: args.max_total]
    for row in rows:
        row["priority_score"] = str(priority_score(row))

    fieldnames = [
        "source",
        "source_query",
        "repository",
        "issue_number",
        "title",
        "url",
        "state",
        "created_at",
        "updated_at",
        "github_labels",
        "candidate_failure_labels",
        "candidate_primary_failure",
        "candidate_cause_labels",
        "priority_score",
        "body_excerpt",
    ]
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    repo_counts = Counter(row["repository"] for row in rows)
    label_counts = Counter(row["candidate_primary_failure"] for row in rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Online Candidate Collection",
                "",
                f"Queries attempted: {query_count}",
                f"Unique candidates written: {len(rows)}",
                f"Raw JSONL: `{raw_out.relative_to(ROOT)}`",
                f"Candidate CSV: `{csv_out.relative_to(ROOT)}`",
                "",
                "These rows are candidate issues only. They are not gold labels until humans review `primary_failure_label`, `confidence`, and `evidence_quote` fields in a blind packet.",
                "",
                "## Candidate label counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(label_counts.items())],
                "",
                "## Repository counts",
                "",
                *[f"- {repo}: {count}" for repo, count in repo_counts.most_common()],
                "",
                "## Fetch errors",
                "",
                *([f"- {error}" for error in errors[:30]] if errors else ["- none"]),
            ]
        ),
        encoding="utf-8",
    )
    print(csv_out)
    print(report)


if __name__ == "__main__":
    main()
