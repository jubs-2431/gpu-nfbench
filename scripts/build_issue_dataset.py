from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = [ROOT / "data" / "raw", ROOT / "data" / "raw_more"]
OUT = ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv"


FAILURE_PATTERNS = {
    "nan_inf": re.compile(r"\b(nan|inf|infinite|nonfinite|non-finite)\b", re.I),
    "overflow_underflow": re.compile(r"\b(over[- ]?flow|under[- ]?flow)\b", re.I),
    "precision_tolerance": re.compile(r"\b(precision|rounding|tolerance|allclose|rtol|atol|wrong result|incorrect result|incorrect output|mismatch)\b", re.I),
    "dtype_casting": re.compile(r"\b(dtype|data type|cast|casting|float16|float32|float64|bfloat16|bf16|fp16|fp32|fp64|int64|int32|complex64|complex128)\b", re.I),
    "crash_compile": re.compile(r"\b(segfault|crash|assert|exception|compile error|compilation|fails to compile|error when compiling)\b", re.I),
    "performance_only": re.compile(r"\b(slow|slower|performance|tflops|benchmark|regression)\b", re.I),
}

CAUSE_PATTERNS = {
    "memory_mask_bounds": re.compile(r"\b(mask|out[- ]of[- ]bounds|boundary|stride|descriptor|pointer|offset|broadcast)\b", re.I),
    "compiler_codegen": re.compile(r"\b(compiler|codegen|lowering|ptx|llvm|inductor|xla|nightly|fusion|fused|kernel generation)\b", re.I),
    "async_race_ordering": re.compile(r"\b(async|synchroni[sz]e|race|stream|barrier|order|nondeterministic|deterministic)\b", re.I),
    "hardware_backend": re.compile(r"\b(cuda|gpu|h100|h200|b200|a100|rtx|hopper|rocm|xla|mps|sm[0-9])\b", re.I),
    "reduction_accumulation": re.compile(r"\b(reduction|sum|accumulat|softmax|attention|matmul|dot|exp|log)\b", re.I),
}


def labels_from_patterns(text: str, patterns: dict[str, re.Pattern[str]]) -> str:
    labels = [name for name, pattern in patterns.items() if pattern.search(text)]
    return "|".join(labels) if labels else "needs_review"


def primary_failure_label(text: str) -> str:
    labels = labels_from_patterns(text, FAILURE_PATTERNS).split("|")
    ordered = [
        "nan_inf",
        "overflow_underflow",
        "precision_tolerance",
        "dtype_casting",
        "crash_compile",
        "performance_only",
    ]
    for label in ordered:
        if label in labels:
            return label
    return "needs_review"


def main() -> None:
    rows: dict[str, dict[str, str]] = {}
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            continue
        for path in sorted(raw_dir.glob("*.json")):
            records = json.loads(path.read_text(encoding="utf-8") or "[]")
            for item in records:
                url = str(item.get("url", "")).strip()
                if not url:
                    continue
                labels = item.get("labels") or []
                label_names = []
                for label in labels:
                    if isinstance(label, dict):
                        label_names.append(str(label.get("name", "")))
                    else:
                        label_names.append(str(label))
                repo = item.get("repository") or {}
                body = str(item.get("body", "") or "")
                title = str(item.get("title", "") or "")
                text = f"{title}\n{body}"
                rows[url] = {
                    "source_file": path.name,
                    "repository": str(repo.get("nameWithOwner", "")),
                    "title": title.replace("\n", " ").strip(),
                    "url": url,
                    "state": str(item.get("state", "")),
                    "created_at": str(item.get("createdAt", "")),
                    "updated_at": str(item.get("updatedAt", "")),
                    "github_labels": "|".join(x for x in label_names if x),
                    "candidate_failure_labels": labels_from_patterns(text, FAILURE_PATTERNS),
                    "candidate_primary_failure": primary_failure_label(text),
                    "candidate_cause_labels": labels_from_patterns(text, CAUSE_PATTERNS),
                    "agent_review_label": "",
                    "agent_review_confidence": "",
                    "agent_review_notes": "",
                    "body_excerpt": body.replace("\n", " ").strip()[:800],
                }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "repository",
        "title",
        "url",
        "state",
        "created_at",
        "updated_at",
        "github_labels",
        "candidate_failure_labels",
        "candidate_primary_failure",
        "candidate_cause_labels",
        "agent_review_label",
        "agent_review_confidence",
        "agent_review_notes",
        "body_excerpt",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows.values())
    print(f"wrote {len(rows)} unique issues to {OUT}")


if __name__ == "__main__":
    main()
