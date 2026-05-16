from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data" / "processed" / "gold_candidate_subset.csv"
CONTEXT = ROOT / "data" / "gold_context"
OUT_DIR = ROOT / "annotation"

PRIMARY_LABELS = [
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
    "needs_review",
]
CAUSE_LABELS = [
    "memory_mask_bounds",
    "compiler_codegen",
    "async_race_ordering",
    "hardware_backend",
    "reduction_accumulation",
    "api_semantics",
    "environment_configuration",
    "unknown",
]


def safe_name(repo: str, url: str) -> str:
    number = url.rstrip("/").split("/")[-1]
    return repo.replace("/", "__") + f"__{number}"


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def excerpt(value: str, limit: int) -> str:
    text = normalize(value)
    if len(text) <= limit:
        return text
    return text[: limit - 15].rstrip() + " ... [truncated]"


def stable_order_key(row: dict[str, str]) -> str:
    return hashlib.sha256(("gpu-nfbench-v1|" + row["url"]).encode("utf-8")).hexdigest()


def blind_id(row: dict[str, str], index: int) -> str:
    digest = hashlib.sha1(row["url"].encode("utf-8")).hexdigest()[:8]
    return f"GNF-{index:04d}-{digest}"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8") or "null")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = sorted(
        list(csv.DictReader(SUBSET.open(newline="", encoding="utf-8"))),
        key=stable_order_key,
    )

    packet_rows: list[dict[str, object]] = []
    suggestion_rows: list[dict[str, object]] = []
    adjudication_rows: list[dict[str, object]] = []

    for index, row in enumerate(rows, start=1):
        repo = row["repository"]
        stem = safe_name(repo, row["url"])
        issue_path = CONTEXT / f"{stem}.issue.json"
        comments_path = CONTEXT / f"{stem}.comments.json"
        issue = load_json(issue_path)
        comments = load_json(comments_path)
        assert isinstance(issue, dict)
        assert isinstance(comments, list)

        number = str(issue.get("number") or row["url"].rstrip("/").split("/")[-1])
        title = str(issue.get("title") or row["title"])
        body = str(issue.get("body") or "")
        comment_bodies = [str(item.get("body") or "") for item in comments if isinstance(item, dict)]
        comment_excerpt = "\n\n".join(comment_bodies[:8])
        bid = blind_id(row, index)

        packet_rows.append(
            {
                "blind_id": bid,
                "repository": repo,
                "issue_number": number,
                "url": issue.get("html_url", row["url"]),
                "title": normalize(title),
                "github_state": issue.get("state", row["state"]),
                "github_labels": row["github_labels"],
                "issue_body_excerpt": excerpt(body, 5000),
                "comments_excerpt": excerpt(comment_excerpt, 3500),
                "primary_failure_label": "",
                "secondary_cause_labels_pipe_separated": "",
                "is_true_numerical_failure": "",
                "evidence_quote": "",
                "confidence_high_medium_low": "",
                "annotator_notes": "",
            }
        )
        suggestion_rows.append(
            {
                "blind_id": bid,
                "repository": repo,
                "issue_number": number,
                "url": issue.get("html_url", row["url"]),
                "candidate_primary_failure": row["candidate_primary_failure"],
                "candidate_failure_labels": row["candidate_failure_labels"],
                "candidate_cause_labels": row["candidate_cause_labels"],
                "source_file": row["source_file"],
            }
        )
        adjudication_rows.append(
            {
                "blind_id": bid,
                "repository": repo,
                "issue_number": number,
                "url": issue.get("html_url", row["url"]),
                "annotator_a_primary": "",
                "annotator_b_primary": "",
                "annotator_a_causes": "",
                "annotator_b_causes": "",
                "gold_primary_failure": "",
                "gold_secondary_cause_labels_pipe_separated": "",
                "gold_is_true_numerical_failure": "",
                "gold_evidence_quote": "",
                "adjudicator_id": "",
                "adjudication_notes": "",
            }
        )

    packet_fields = list(packet_rows[0].keys())
    write_csv(OUT_DIR / "annotator_A_blind.csv", packet_rows, packet_fields)
    write_csv(OUT_DIR / "annotator_B_blind.csv", packet_rows, packet_fields)
    write_csv(OUT_DIR / "candidate_label_suggestions_hidden_from_annotators.csv", suggestion_rows, list(suggestion_rows[0].keys()))
    write_csv(OUT_DIR / "adjudication_template.csv", adjudication_rows, list(adjudication_rows[0].keys()))

    guide = f"""# GPU-NFBench Annotation Guide

This folder is for creating an independent human-labeled gold benchmark.
Do not call the dataset gold until two independent human annotators have filled
`annotator_A_blind.csv` and `annotator_B_blind.csv`, disagreements have been
adjudicated in `adjudication_template.csv`, and agreement metrics have been
computed.

## Allowed Primary Labels

{chr(10).join(f'- `{label}`' for label in PRIMARY_LABELS)}

## Allowed Secondary Cause Labels

Use pipe-separated labels when multiple causes apply.

{chr(10).join(f'- `{label}`' for label in CAUSE_LABELS)}

## Annotation Rules

1. Read the title, issue body excerpt, comments excerpt, and source URL if needed.
2. Assign exactly one `primary_failure_label`.
3. Set `is_true_numerical_failure` to `yes`, `no`, or `unclear`.
4. Add one or more secondary cause labels if the issue text supports them.
5. Provide a short evidence quote from the public issue or comment text.
6. Use `needs_review` when the issue is too ambiguous to classify from public evidence.
7. Use `not_numerical_failure` when the query matched numerical words but the issue is not actually a numerical correctness failure.
8. Do not look at `candidate_label_suggestions_hidden_from_annotators.csv` while annotating.

## Gold Release Rule

A row becomes gold only after the adjudication file has:

- `gold_primary_failure`
- `gold_secondary_cause_labels_pipe_separated`
- `gold_is_true_numerical_failure`
- `gold_evidence_quote`
- `adjudicator_id`

Run `python3 scripts/evaluate_gold_labels.py` from the project root after both
blind files and the adjudication file are complete.
"""
    (OUT_DIR / "ANNOTATION_GUIDE.md").write_text(guide, encoding="utf-8")
    print(f"wrote annotation packets for {len(packet_rows)} issues to {OUT_DIR}")


if __name__ == "__main__":
    main()
