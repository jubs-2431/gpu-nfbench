from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from adjudicate_validation_subset import (
    LABEL_PATTERNS,
    choose_primary,
    contains_any,
    evidence_for,
    label_causes,
    normalize_text,
)


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "annotation"
CONTEXT = ROOT / "data" / "gold_context"
OUT_A = ANNOTATION / "ai_prelabel_pass_A_context_only.csv"
OUT_B = ANNOTATION / "ai_prelabel_pass_B_candidate_aware.csv"
OUT_DISAGREE = ANNOTATION / "ai_prelabel_disagreements.csv"
REPORT = ROOT / "reports" / "ai_preannotation_report.md"

API_PATTERNS = [
    re.compile(r"\b(api|signature|numpy compat|numpy-compatible|semantics|behavior|unsupported operation|not implemented|feature request)\b", re.I),
    re.compile(r"\b(inconsistent with numpy|different from numpy|matches numpy|compatibility)\b", re.I),
]
ENV_PATTERNS = [
    re.compile(r"\b(ld_library_path|install|installation|driver|cuda version|nvidia-smi|wheel|conda|pip|build|cmake|library path|libnvrtc|runtime configuration)\b", re.I),
]
FALSE_POSITIVE_PATTERNS = [
    re.compile(r"\b(documentation|docs|feature request|proposal|discussion|question|how to|support request)\b", re.I),
]
TRUE_FAILURE_LABELS = {
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(repo: str, number: str) -> str:
    return repo.replace("/", "__") + f"__{number}"


def load_context(row: dict[str, str]) -> tuple[str, str, str]:
    stem = safe_name(row["repository"], row["issue_number"])
    issue = json.loads((CONTEXT / f"{stem}.issue.json").read_text(encoding="utf-8"))
    comments = json.loads((CONTEXT / f"{stem}.comments.json").read_text(encoding="utf-8"))
    title = str(issue.get("title") or row["title"])
    body = str(issue.get("body") or "")
    comment_text = "\n".join(str(item.get("body") or "") for item in comments if isinstance(item, dict))
    return title, body, comment_text


def likely_not_numerical_failure(label: str, title: str, body: str, comments: str) -> bool:
    text = f"{title}\n{body[:1600]}\n{comments[:1600]}"
    if label != "needs_review":
        return False
    if contains_any(text, FALSE_POSITIVE_PATTERNS) and not any(
        contains_any(text, patterns)
        for key, patterns in LABEL_PATTERNS.items()
        if key not in {"performance_only"}
    ):
        return True
    return False


def is_true_numerical_failure(label: str) -> str:
    if label in {"performance_only", "not_numerical_failure"}:
        return "no"
    if label == "needs_review":
        return "unclear"
    if label in TRUE_FAILURE_LABELS:
        return "yes"
    return "unclear"


def enrich_causes(cause_labels: str, full_text: str) -> str:
    labels = [] if cause_labels == "needs_review" else [item for item in cause_labels.split("|") if item]
    if contains_any(full_text, API_PATTERNS):
        labels.append("api_semantics")
    if contains_any(full_text, ENV_PATTERNS):
        labels.append("environment_configuration")
    if not labels:
        labels = ["unknown"]
    return "|".join(sorted(dict.fromkeys(labels)))


def candidate_aware_label(
    strict_label: str,
    strict_confidence: str,
    candidate_label: str,
    title: str,
    body: str,
    comments: str,
) -> tuple[str, str]:
    if strict_confidence in {"high", "medium"} and strict_label not in {"needs_review"}:
        return strict_label, strict_confidence
    if candidate_label in LABEL_PATTERNS and contains_any(f"{title}\n{body[:3000]}", LABEL_PATTERNS[candidate_label]):
        return candidate_label, "medium"
    if candidate_label not in {"needs_review", ""}:
        return candidate_label, "low"
    return strict_label, strict_confidence


def make_row(
    base: dict[str, str],
    label: str,
    confidence: str,
    cause_labels: str,
    evidence: str,
    pass_name: str,
) -> dict[str, object]:
    return {
        **base,
        "primary_failure_label": label,
        "secondary_cause_labels_pipe_separated": cause_labels,
        "is_true_numerical_failure": is_true_numerical_failure(label),
        "evidence_quote": evidence,
        "confidence_high_medium_low": confidence,
        "annotator_notes": (
            f"AI prelabel pass {pass_name}; not a human annotation and not valid as gold without independent human review."
        ),
    }


def md_table(rows: list[dict[str, object]], headers: list[str]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return out


def main() -> None:
    blind_rows = read_csv(ANNOTATION / "annotator_A_blind.csv")
    candidate_rows = {
        row["blind_id"]: row
        for row in read_csv(ANNOTATION / "candidate_label_suggestions_hidden_from_annotators.csv")
    }

    pass_a: list[dict[str, object]] = []
    pass_b: list[dict[str, object]] = []
    disagreements: list[dict[str, object]] = []

    for row in blind_rows:
        title, body, comments = load_context(row)
        full_text = f"{title}\n{body}\n{comments}"
        strict_label, strict_confidence = choose_primary(title, body, comments)
        if likely_not_numerical_failure(strict_label, title, body, comments):
            strict_label, strict_confidence = "not_numerical_failure", "medium"
        causes = enrich_causes(label_causes(full_text), full_text)
        evidence_source, evidence = evidence_for(
            strict_label,
            [("title", title), ("issue_body", body), ("comments", comments)],
        )
        if not evidence:
            evidence = normalize_text(title)[:420]

        base = dict(row)
        a_row = make_row(base, strict_label, strict_confidence, causes, evidence, "A/context-only")
        pass_a.append(a_row)

        candidate_label = candidate_rows.get(row["blind_id"], {}).get("candidate_primary_failure", "")
        b_label, b_confidence = candidate_aware_label(
            strict_label,
            strict_confidence,
            candidate_label,
            title,
            body,
            comments,
        )
        b_evidence_source, b_evidence = evidence_for(
            b_label,
            [("title", title), ("issue_body", body), ("comments", comments)],
        )
        if not b_evidence:
            b_evidence = evidence
        b_row = make_row(base, b_label, b_confidence, causes, b_evidence, "B/candidate-aware")
        pass_b.append(b_row)

        if strict_label != b_label:
            disagreements.append(
                {
                    "blind_id": row["blind_id"],
                    "repository": row["repository"],
                    "issue_number": row["issue_number"],
                    "url": row["url"],
                    "title": row["title"],
                    "pass_A_label": strict_label,
                    "pass_A_confidence": strict_confidence,
                    "pass_B_label": b_label,
                    "pass_B_confidence": b_confidence,
                    "candidate_primary_failure": candidate_label,
                    "evidence_quote": b_evidence,
                }
            )

    fieldnames = list(pass_a[0].keys())
    write_csv(OUT_A, pass_a, fieldnames)
    write_csv(OUT_B, pass_b, fieldnames)
    write_csv(
        OUT_DISAGREE,
        disagreements,
        [
            "blind_id",
            "repository",
            "issue_number",
            "url",
            "title",
            "pass_A_label",
            "pass_A_confidence",
            "pass_B_label",
            "pass_B_confidence",
            "candidate_primary_failure",
            "evidence_quote",
        ],
    )

    a_counts = Counter(str(row["primary_failure_label"]) for row in pass_a)
    b_counts = Counter(str(row["primary_failure_label"]) for row in pass_b)
    lines = [
        "# AI Preannotation Report",
        "",
        "These files are AI-generated prelabels. They are not human annotations and must not be reported as an independent human-labeled gold benchmark.",
        "",
        f"Rows pre-labeled: {len(pass_a)}",
        f"Pass A/B disagreements: {len(disagreements)}",
        "",
        "## Pass A context-only labels",
        *md_table(
            [{"label": label, "issues": count} for label, count in a_counts.most_common()],
            ["label", "issues"],
        ),
        "",
        "## Pass B candidate-aware labels",
        *md_table(
            [{"label": label, "issues": count} for label, count in b_counts.most_common()],
            ["label", "issues"],
        ),
        "",
        "## Outputs",
        "",
        f"- `{OUT_A.relative_to(ROOT)}`",
        f"- `{OUT_B.relative_to(ROOT)}`",
        f"- `{OUT_DISAGREE.relative_to(ROOT)}`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_A)
    print(OUT_B)
    print(OUT_DISAGREE)
    print(REPORT)


if __name__ == "__main__":
    main()
