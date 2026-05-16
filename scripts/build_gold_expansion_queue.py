from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
SEED = ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv"
ONLINE = ROOT / "data" / "processed" / "online_candidate_issue_pool.csv"
NEGATIVE = ROOT / "data" / "processed" / "negative_control_candidate_issue_pool.csv"
SUGGESTIONS = ROOT / "annotation" / "full_coverage_expansion_model_suggestions.csv"
OUT = ROOT / "annotation" / "gold_expansion_1000_queue.csv"
BLIND_OUT = ROOT / "annotation" / "gold_expansion_1000_blind.csv"
REPORT = ROOT / "reports" / "gold_expansion_1000_plan.md"

TARGET_LABELS = [
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
    "needs_review",
]

REVIEW_FIELDS = [
    "expansion_id",
    "source",
    "repository",
    "issue_number",
    "title",
    "url",
    "state",
    "created_at",
    "updated_at",
    "github_labels",
    "body_excerpt",
    "primary_failure_label",
    "secondary_cause_labels",
    "is_true_numerical_failure",
    "confidence",
    "evidence_quote",
    "notes",
    "candidate_primary_failure",
    "candidate_failure_labels",
    "candidate_cause_labels",
    "model_vote_prediction",
    "model_vote_confidence",
    "selection_reason",
    "priority_score",
]

BLIND_FIELDS = [
    "expansion_id",
    "repository",
    "issue_number",
    "title",
    "url",
    "state",
    "created_at",
    "updated_at",
    "github_labels",
    "body_excerpt",
    "primary_failure_label",
    "secondary_cause_labels",
    "is_true_numerical_failure",
    "confidence",
    "evidence_quote",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def issue_number_from_url(url: str) -> str:
    parts = url.rstrip("/").split("/")
    return parts[-1] if len(parts) >= 2 and parts[-2] == "issues" else ""


def expansion_id(url: str, index: int) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"EGNF1000-{index:04d}-{digest}"


def normalize_seed_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "source": row.get("source_file", "local_seed"),
        "repository": row.get("repository", ""),
        "issue_number": issue_number_from_url(row.get("url", "")),
        "title": row.get("title", ""),
        "url": row.get("url", ""),
        "state": row.get("state", ""),
        "created_at": row.get("created_at", ""),
        "updated_at": row.get("updated_at", ""),
        "github_labels": row.get("github_labels", ""),
        "body_excerpt": row.get("body_excerpt", ""),
        "candidate_primary_failure": row.get("candidate_primary_failure", "needs_review"),
        "candidate_failure_labels": row.get("candidate_failure_labels", "needs_review"),
        "candidate_cause_labels": row.get("candidate_cause_labels", "unknown"),
        "priority_score": "0",
    }


def normalize_online_row(row: dict[str, str], negative_control: bool = False) -> dict[str, str]:
    normalized = normalize_seed_row(row)
    normalized["source"] = row.get("source", "github_search")
    normalized["issue_number"] = row.get("issue_number", "") or issue_number_from_url(row.get("url", ""))
    normalized["priority_score"] = row.get("priority_score", "0")
    if negative_control and normalized["candidate_primary_failure"] in {"needs_review", "performance_only", "crash_compile"}:
        normalized["candidate_primary_failure"] = "not_numerical_failure"
        normalized["candidate_failure_labels"] = "not_numerical_failure_candidate"
        normalized["candidate_cause_labels"] = normalized["candidate_cause_labels"] or "unknown"
        normalized["source"] = "negative_control_github_search"
    return normalized


def choose_reason(row: dict[str, str], model_suggestions: dict[str, dict[str, str]], selected_counts: Counter[str]) -> tuple[int, str, str, str]:
    label = row.get("candidate_primary_failure") or "needs_review"
    model_vote = ""
    model_conf = ""
    url = row["url"]
    for suggestion in model_suggestions.values():
        if suggestion.get("url") == url:
            model_vote = suggestion.get("vote_prediction", "")
            model_conf = suggestion.get("vote_confidence", "")
            break

    reasons = []
    score = int(row.get("priority_score") or 0)
    if label in {"precision_tolerance", "crash_compile", "performance_only", "not_numerical_failure"}:
        score += 25
        reasons.append("rare_or_undercovered_candidate_label")
    if model_vote and model_vote != label:
        score += 20
        reasons.append("model_candidate_disagreement")
    if model_conf == "low":
        score += 10
        reasons.append("low_model_confidence")
    if selected_counts[label] < 125:
        score += 15
        reasons.append("label_balance_quota")
    if row.get("repository", "").startswith(("pytorch/", "tensorflow/", "jax-ml/", "rapidsai/")):
        score += 5
        reasons.append("cross_repository_generalization")
    return score, "|".join(reasons or ["high_priority_candidate"]), model_vote, model_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 1000-row human annotation queue for GPU-NFBench expansion.")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--per-label-soft-cap", type=int, default=160)
    args = parser.parse_args()

    gold_urls = {row["url"] for row in read_csv(GOLD)}
    model_suggestions = {row.get("expansion_id", ""): row for row in read_csv(SUGGESTIONS)}

    by_url: dict[str, dict[str, str]] = {}
    for row in read_csv(SEED):
        normalized = normalize_seed_row(row)
        if normalized["url"] and normalized["url"] not in gold_urls:
            by_url[normalized["url"]] = normalized
    for row in read_csv(ONLINE):
        normalized = normalize_online_row(row)
        if normalized["url"] and normalized["url"] not in gold_urls:
            old = by_url.get(normalized["url"])
            if old is None or int(normalized.get("priority_score") or 0) > int(old.get("priority_score") or 0):
                by_url[normalized["url"]] = normalized
    for row in read_csv(NEGATIVE):
        normalized = normalize_online_row(row, negative_control=True)
        if normalized["url"] and normalized["url"] not in gold_urls:
            old = by_url.get(normalized["url"])
            if old is None or int(normalized.get("priority_score") or 0) > int(old.get("priority_score") or 0):
                by_url[normalized["url"]] = normalized

    candidates = list(by_url.values())
    selected: list[dict[str, str]] = []
    selected_counts: Counter[str] = Counter()
    scored_rows = []
    for row in candidates:
        score, reason, model_vote, model_conf = choose_reason(row, model_suggestions, selected_counts)
        scored_rows.append((score, reason, model_vote, model_conf, row))
    scored_rows.sort(key=lambda item: (-item[0], item[4]["repository"], item[4]["url"]))

    deferred = []
    for score, reason, model_vote, model_conf, row in scored_rows:
        label = row.get("candidate_primary_failure") or "needs_review"
        if selected_counts[label] >= args.per_label_soft_cap:
            deferred.append((score, reason, model_vote, model_conf, row))
            continue
        selected.append({**row, "selection_reason": reason, "model_vote_prediction": model_vote, "model_vote_confidence": model_conf, "priority_score": str(score)})
        selected_counts[label] += 1
        if len(selected) >= args.target:
            break

    for score, reason, model_vote, model_conf, row in deferred:
        if len(selected) >= args.target:
            break
        selected.append({**row, "selection_reason": reason, "model_vote_prediction": model_vote, "model_vote_confidence": model_conf, "priority_score": str(score)})
        selected_counts[row.get("candidate_primary_failure") or "needs_review"] += 1

    review_rows: list[dict[str, str]] = []
    blind_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        base = {
            "expansion_id": expansion_id(row["url"], index),
            "source": row["source"],
            "repository": row["repository"],
            "issue_number": row["issue_number"],
            "title": row["title"],
            "url": row["url"],
            "state": row["state"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "github_labels": row["github_labels"],
            "body_excerpt": row["body_excerpt"],
            "primary_failure_label": "",
            "secondary_cause_labels": "",
            "is_true_numerical_failure": "",
            "confidence": "",
            "evidence_quote": "",
            "notes": "",
            "candidate_primary_failure": row["candidate_primary_failure"],
            "candidate_failure_labels": row["candidate_failure_labels"],
            "candidate_cause_labels": row["candidate_cause_labels"],
            "model_vote_prediction": row["model_vote_prediction"],
            "model_vote_confidence": row["model_vote_confidence"],
            "selection_reason": row["selection_reason"],
            "priority_score": row["priority_score"],
        }
        review_rows.append(base)
        blind_rows.append({field: base[field] for field in BLIND_FIELDS})

    write_csv(OUT, review_rows, REVIEW_FIELDS)
    write_csv(BLIND_OUT, blind_rows, BLIND_FIELDS)

    label_counts = Counter(row["candidate_primary_failure"] for row in review_rows)
    repo_counts = Counter(row["repository"] for row in review_rows)
    reason_counts: Counter[str] = Counter()
    for row in review_rows:
        reason_counts.update(row["selection_reason"].split("|"))
    REPORT.write_text(
        "\n".join(
            [
                "# Gold Expansion 1000 Plan",
                "",
                f"Candidate pool after removing existing gold URLs: {len(candidates)}",
                f"Rows selected for human labeling: {len(review_rows)}",
                f"Review queue: `{OUT.relative_to(ROOT)}`",
                f"Blind annotation queue: `{BLIND_OUT.relative_to(ROOT)}`",
                "",
                "The review queue includes model and weak-label suggestions for project planning. The blind queue hides these fields and is the file to send to annotators if preserving independent human labels.",
                "",
                "## Candidate label distribution",
                "",
                *[f"- {label}: {label_counts.get(label, 0)}" for label in TARGET_LABELS],
                "",
                "## Selection reasons",
                "",
                *[f"- {reason}: {count}" for reason, count in reason_counts.most_common()],
                "",
                "## Top repositories",
                "",
                *[f"- {repo}: {count}" for repo, count in repo_counts.most_common(20)],
            ]
        ),
        encoding="utf-8",
    )
    print(OUT)
    print(BLIND_OUT)
    print(REPORT)


if __name__ == "__main__":
    main()
