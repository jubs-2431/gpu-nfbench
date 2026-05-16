from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data" / "processed" / "validation_subset.csv"
CONTEXT = ROOT / "data" / "validation_context"
OUT = ROOT / "data" / "processed" / "validation_adjudicated.csv"
AGREEMENT_TABLE = ROOT / "tables" / "validation_agreement.csv"
QUALITY_TABLE = ROOT / "tables" / "validation_quality.csv"
REPORT = ROOT / "reports" / "validation_adjudication.md"


DTYPE_SYMPTOM_PATTERNS = [
    re.compile(r"\b(dtype|data type|type promotion|promot(?:e|ion)|cast(?:ing)?|convert(?:ing|ed)?|signature mismatch|unsupported type)\b", re.I),
    re.compile(r"\b(incorrect type|wrong type|type mismatch|dtype mismatch|dtype bug)\b", re.I),
]
DTYPE_TOKEN_PATTERN = re.compile(
    r"\b(float16|float32|float64|bfloat16|bf16|fp16|fp32|fp64|int8|int16|int32|int64|uint32|uint64|complex64|complex128)\b",
    re.I,
)

LABEL_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "dtype_casting": [
        *DTYPE_SYMPTOM_PATTERNS,
        DTYPE_TOKEN_PATTERN,
    ],
    "nan_inf": [
        re.compile(r"\b(nan|inf|infinite|nonfinite|non-finite|invalid value)\b", re.I),
    ],
    "overflow_underflow": [
        re.compile(r"\b(over[- ]?flow|under[- ]?flow|saturat(?:e|ion)|exponent blow[- ]?up|out of range)\b", re.I),
    ],
    "precision_tolerance": [
        re.compile(r"\b(precision|rounding|tolerance|allclose|rtol|atol|wrong result|wrong output|incorrect result|incorrect output|mismatch|not close|accuracy|inconsistent|different from numpy)\b", re.I),
    ],
    "crash_compile": [
        re.compile(r"\b(segfault|segmentation fault|crash|assert(?:ion)?|exception|traceback|compile error|compilation error|fails? to compile|compiler error|runtime error|unexpected error|internal compiler error)\b", re.I),
    ],
    "performance_only": [
        re.compile(r"\b(slow|slower|performance|throughput|latency|tflops|benchmark|regression)\b", re.I),
    ],
}

CAUSE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "memory_mask_bounds": [
        re.compile(r"\b(mask|masked|out[- ]of[- ]bounds|boundary|stride|strided|descriptor|pointer|offset|broadcast|alignment)\b", re.I),
    ],
    "compiler_codegen": [
        re.compile(r"\b(compiler|codegen|lowering|ptx|llvm|mlir|inductor|xla|jit|fusion|fused|kernel generation|triton)\b", re.I),
    ],
    "async_race_ordering": [
        re.compile(r"\b(async|synchroni[sz]e|race|stream|barrier|order(?:ing)?|nondeterministic|non-deterministic|deterministic)\b", re.I),
    ],
    "hardware_backend": [
        re.compile(r"\b(cuda|gpu|h100|h200|b200|a100|v100|rtx|hopper|ampere|rocm|mps|sm[0-9]{2,3}|cu(?:blas|solver|dnn|rand))\b", re.I),
    ],
    "reduction_accumulation": [
        re.compile(r"\b(reduction|reduce|sum|accumulat(?:e|ion)|softmax|attention|matmul|matrix multiply|dot|exp|log|norm)\b", re.I),
    ],
}

REPRO_PATTERNS = [
    re.compile(r"```"),
    re.compile(r"\b(reproducer|repro|minimal example|to reproduce|steps to reproduce|example)\b", re.I),
    re.compile(r"\b(import numpy|import cupy|import torch|import jax|@triton|cuda\.jit|def test_)\b", re.I),
]
STACK_PATTERNS = [
    re.compile(r"\b(traceback|stack trace|segmentation fault|assertionerror|runtimeerror|typeerror|valueerror|cuda error)\b", re.I),
]
FIX_PATTERNS = [
    re.compile(r"\b(fix(?:ed|es)?|closed by|resolved by|merged|landed|regression fixed|workaround|patch)\b", re.I),
    re.compile(r"\b(PR|pull request)\s+#?\d+", re.I),
]


def safe_name(repo: str, url: str) -> str:
    number = url.rstrip("/").split("/")[-1]
    return repo.replace("/", "__") + f"__{number}"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8") or "null")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def count_matches(text: str, patterns: list[re.Pattern[str]]) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def contains_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def score_labels(title: str, body: str, comments: str) -> dict[str, int]:
    text = f"{title}\n{body}\n{comments}"
    scores: dict[str, int] = {}
    for label, patterns in LABEL_PATTERNS.items():
        score = 0
        if label == "dtype_casting":
            score += 6 * count_matches(title, DTYPE_SYMPTOM_PATTERNS)
            score += 3 * count_matches(body[:4000], DTYPE_SYMPTOM_PATTERNS)
            score += count_matches(comments[:8000], DTYPE_SYMPTOM_PATTERNS)
            score += 2 * count_matches(title, [DTYPE_TOKEN_PATTERN])
            # Plain dtype tokens inside repro code are common context, but they
            # should not overwhelm an issue whose reported symptom is a wrong
            # numerical result.
            score += min(2, count_matches(body[:4000], [DTYPE_TOKEN_PATTERN]))
            score += min(1, count_matches(comments[:8000], [DTYPE_TOKEN_PATTERN]))
            scores[label] = score
            continue
        score += 5 * count_matches(title, patterns)
        score += 2 * count_matches(body[:4000], patterns)
        score += count_matches(comments[:8000], patterns)
        scores[label] = score

    correctness_total = (
        scores["dtype_casting"]
        + scores["nan_inf"]
        + scores["overflow_underflow"]
        + scores["precision_tolerance"]
        + scores["crash_compile"]
    )
    if scores["performance_only"] > 0 and correctness_total == 0:
        scores["performance_only"] += 4
    elif correctness_total > 0:
        scores["performance_only"] = max(0, scores["performance_only"] - 3)
    return scores


def choose_primary(title: str, body: str, comments: str) -> tuple[str, str]:
    scores = score_labels(title, body, comments)
    title_body = f"{title}\n{body[:1200]}"

    # Strong explicit issue titles should dominate incidental later terms.
    strong_order = [
        "overflow_underflow",
        "dtype_casting",
        "nan_inf",
        "precision_tolerance",
        "crash_compile",
    ]
    for label in strong_order:
        patterns = DTYPE_SYMPTOM_PATTERNS if label == "dtype_casting" else LABEL_PATTERNS[label]
        if contains_any(title, patterns):
            confidence = "high" if scores[label] >= 5 else "medium"
            return label, confidence

    if contains_any(title, LABEL_PATTERNS["performance_only"]) and not contains_any(
        title,
        LABEL_PATTERNS["nan_inf"]
        + LABEL_PATTERNS["overflow_underflow"]
        + LABEL_PATTERNS["precision_tolerance"]
        + DTYPE_SYMPTOM_PATTERNS
        + LABEL_PATTERNS["crash_compile"],
    ):
        return "performance_only", "high" if scores["performance_only"] >= 5 else "medium"

    if re.search(r"\b(multi-gpu support|discussion|feature request)\b", title, re.I):
        return "performance_only", "medium"

    if contains_any(
        title,
        [
            re.compile(r"\b(unexpected output|wrong output|wrong result|incorrect result|incorrect output|floating point errors?|mismatch|inconsistent with)\b", re.I),
        ],
    ):
        return "precision_tolerance", "high" if scores["precision_tolerance"] >= 5 else "medium"

    # Overflow and dtype are often hidden by incidental "inf" or reference-code
    # tokens; give them priority when they appear in the early report context.
    for label in ["overflow_underflow", "nan_inf", "dtype_casting"]:
        patterns = DTYPE_SYMPTOM_PATTERNS if label == "dtype_casting" else LABEL_PATTERNS[label]
        if contains_any(title_body, patterns) and scores[label] >= 3:
            return label, "high" if scores[label] >= 6 else "medium"

    if contains_any(
        title_body,
        [
            re.compile(r"\b(should be|unexpected output|incorrect output|incorrect result|wrong output|wrong result|mismatch|not equal|not close|different from numpy|inconsistent with numpy)\b", re.I),
        ],
    ):
        return "precision_tolerance", "medium"

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    label, score = ranked[0]
    if score == 0:
        return "needs_review", "low"
    if label == "performance_only":
        return label, "medium" if score >= 4 else "low"
    if score >= 6:
        return label, "high"
    if score >= 2:
        return label, "medium"
    return "needs_review", "low"


def label_causes(text: str) -> str:
    labels = [
        label
        for label, patterns in CAUSE_PATTERNS.items()
        if contains_any(text, patterns)
    ]
    return "|".join(labels) if labels else "needs_review"


def evidence_for(label: str, sections: list[tuple[str, str]]) -> tuple[str, str]:
    patterns = LABEL_PATTERNS.get(label, [])
    if not patterns:
        return "", ""
    for source, text in sections:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                start = max(0, match.start() - 180)
                end = min(len(text), match.end() + 220)
                snippet = normalize_text(text[start:end])
                return source, snippet[:420]
    return "", ""


def md_table(rows: list[dict[str, object]], headers: list[str]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return out


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "0.0%"


def main() -> None:
    rows = list(csv.DictReader(SUBSET.open(newline="", encoding="utf-8")))
    output: list[dict[str, object]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    quality = Counter()

    for row in rows:
        repo = row["repository"]
        stem = safe_name(repo, row["url"])
        issue = load_json(CONTEXT / f"{stem}.issue.json")
        comments = load_json(CONTEXT / f"{stem}.comments.json")
        assert isinstance(issue, dict)
        assert isinstance(comments, list)

        title = str(issue.get("title") or row["title"])
        body = str(issue.get("body") or "")
        comment_bodies = [str(item.get("body") or "") for item in comments if isinstance(item, dict)]
        comments_text = "\n".join(comment_bodies)
        full_text = f"{title}\n{body}\n{comments_text}"

        context_label, confidence = choose_primary(title, body, comments_text)
        cause_labels = label_causes(full_text)
        evidence_source, evidence = evidence_for(
            context_label,
            [("title", title), ("issue_body", body), ("comments", comments_text)],
        )
        has_reproducer = contains_any(full_text, REPRO_PATTERNS)
        has_stack_trace = contains_any(full_text, STACK_PATTERNS)
        has_fix_signal = contains_any(full_text, FIX_PATTERNS)

        candidate = row["candidate_primary_failure"]
        confusion[(candidate, context_label)] += 1
        quality["has_reproducer"] += int(has_reproducer)
        quality["has_stack_trace"] += int(has_stack_trace)
        quality["has_fix_signal"] += int(has_fix_signal)
        quality["closed"] += int(str(issue.get("state", row["state"])).lower() == "closed")

        output.append(
            {
                "repository": repo,
                "number": issue.get("number", row["url"].rstrip("/").split("/")[-1]),
                "url": issue.get("html_url", row["url"]),
                "title": normalize_text(title),
                "state": issue.get("state", row["state"]),
                "candidate_primary_failure": candidate,
                "context_primary_failure": context_label,
                "context_confidence": confidence,
                "candidate_context_agree": str(candidate == context_label).lower(),
                "evidence_source": evidence_source,
                "evidence_snippet": evidence,
                "comment_count": len(comment_bodies),
                "has_reproducer": str(has_reproducer).lower(),
                "has_stack_trace": str(has_stack_trace).lower(),
                "has_linked_fix_signal": str(has_fix_signal).lower(),
                "cause_labels": cause_labels,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output[0].keys())
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)

    total = len(output)
    agree = sum(1 for row in output if row["candidate_context_agree"] == "true")
    context_counts = Counter(str(row["context_primary_failure"]) for row in output)
    candidate_counts = Counter(str(row["candidate_primary_failure"]) for row in output)
    confidence_counts = Counter(str(row["context_confidence"]) for row in output)

    agreement_rows = [
        {
            "candidate_primary_failure": candidate,
            "context_primary_failure": context,
            "issues": count,
        }
        for (candidate, context), count in sorted(confusion.items())
    ]
    AGREEMENT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with AGREEMENT_TABLE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "candidate_primary_failure",
                "context_primary_failure",
                "issues",
            ],
        )
        writer.writeheader()
        writer.writerows(agreement_rows)

    quality_rows = [
        {"metric": "validation_issues", "value": total, "share": "100.0%"},
        {"metric": "candidate_context_agreement", "value": agree, "share": pct(agree, total)},
        {"metric": "has_reproducer_or_code", "value": quality["has_reproducer"], "share": pct(quality["has_reproducer"], total)},
        {"metric": "has_stack_trace_or_error_log", "value": quality["has_stack_trace"], "share": pct(quality["has_stack_trace"], total)},
        {"metric": "has_fix_or_workaround_signal", "value": quality["has_fix_signal"], "share": pct(quality["has_fix_signal"], total)},
        {"metric": "closed_issues", "value": quality["closed"], "share": pct(quality["closed"], total)},
    ]
    with QUALITY_TABLE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value", "share"])
        writer.writeheader()
        writer.writerows(quality_rows)

    disagreements = [row for row in output if row["candidate_context_agree"] == "false"]
    representative = disagreements[:12]
    lines = [
        "# Validation Adjudication Report",
        "",
        f"Validation subset size: {total} public GitHub issues.",
        "",
        "This pass uses full issue bodies and public comments fetched through the GitHub API. "
        "The labels are context-adjudicated research labels, not official project labels and not a substitute for independent human annotation.",
        "",
        "## Quality and agreement",
        *md_table(quality_rows, ["metric", "value", "share"]),
        "",
        "## Context primary-label distribution",
        *md_table(
            [
                {
                    "context_primary_failure": label,
                    "issues": count,
                    "share": pct(count, total),
                }
                for label, count in context_counts.most_common()
            ],
            ["context_primary_failure", "issues", "share"],
        ),
        "",
        "## Candidate primary-label distribution in validation subset",
        *md_table(
            [
                {
                    "candidate_primary_failure": label,
                    "issues": count,
                    "share": pct(count, total),
                }
                for label, count in candidate_counts.most_common()
            ],
            ["candidate_primary_failure", "issues", "share"],
        ),
        "",
        "## Confidence distribution",
        *md_table(
            [
                {
                    "context_confidence": label,
                    "issues": count,
                    "share": pct(count, total),
                }
                for label, count in confidence_counts.most_common()
            ],
            ["context_confidence", "issues", "share"],
        ),
        "",
        "## Representative candidate/context disagreements",
        "",
    ]
    for row in representative:
        lines.append(
            f"- {row['repository']}#{row['number']}: candidate={row['candidate_primary_failure']}, "
            f"context={row['context_primary_failure']}, confidence={row['context_confidence']}; "
            f"{row['url']}"
        )
        if row["evidence_snippet"]:
            lines.append(f"  - evidence: {row['evidence_snippet']}")
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    print(REPORT)


if __name__ == "__main__":
    main()
