from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import gold_baseline_classifier as gbc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "annotation" / "gold_expansion_1000_queue.csv"
DEFAULT_OUT = ROOT / "annotation" / "gold_expansion_1000_model_suggestions.csv"
DEFAULT_REPORT = ROOT / "reports" / "gold_expansion_1000_model_suggestions.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_text(row: dict[str, str]) -> str:
    return " ".join(
        [
            row.get("repository", ""),
            row.get("title", ""),
            row.get("github_labels", ""),
            row.get("body_excerpt", ""),
            row.get("candidate_primary_failure", ""),
            row.get("candidate_failure_labels", ""),
            row.get("candidate_cause_labels", ""),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the 1000-row expansion queue with current deterministic gold-label models.")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    queue = args.queue if args.queue.is_absolute() else ROOT / args.queue
    out = args.out if args.out.is_absolute() else ROOT / args.out
    report = args.report if args.report.is_absolute() else ROOT / args.report

    train = gbc.build_rows()
    labels = sorted({row["gold_primary_failure"] for row in train})
    expansion = read_csv(queue)
    test = [
        {
            "blind_id": row["expansion_id"],
            "repository": row["repository"],
            "title": row["title"],
            "text": row_text(row),
            "gold_primary_failure": labels[0],
        }
        for row in expansion
    ]

    model_names = ["tfidf_linear_svm", "tfidf_logistic", "bigram_tfidf_logistic", "naive_bayes", "bm25_knn"]
    predictions = {model_name: gbc.predict_model(model_name, train, test, labels) for model_name in model_names}

    rows: list[dict[str, object]] = []
    for index, row in enumerate(expansion):
        votes = [predictions[model_name][index] for model_name in model_names]
        candidate = row.get("candidate_primary_failure", "")
        if candidate:
            votes.append(candidate)
        counts = Counter(votes)
        top_label, top_count = counts.most_common(1)[0]
        runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        confidence = "high" if top_count >= 5 else "medium" if top_count >= 3 else "low"
        rows.append(
            {
                "expansion_id": row["expansion_id"],
                "repository": row["repository"],
                "title": row["title"],
                "url": row["url"],
                "candidate_primary_failure": candidate,
                **{f"{model_name}_prediction": predictions[model_name][index] for model_name in model_names},
                "vote_prediction": top_label,
                "vote_count": top_count,
                "vote_margin": top_count - runner_up,
                "vote_confidence": confidence,
                "human_gold_status": "suggestion_only_not_gold",
            }
        )

    fieldnames = [
        "expansion_id",
        "repository",
        "title",
        "url",
        "candidate_primary_failure",
        *[f"{model_name}_prediction" for model_name in model_names],
        "vote_prediction",
        "vote_count",
        "vote_margin",
        "vote_confidence",
        "human_gold_status",
    ]
    write_csv(out, rows, fieldnames)

    vote_counts = Counter(str(row["vote_prediction"]) for row in rows)
    confidence_counts = Counter(str(row["vote_confidence"]) for row in rows)
    report.write_text(
        "\n".join(
            [
                "# Gold Expansion 1000 Model Suggestions",
                "",
                f"Queue scored: `{queue.relative_to(ROOT)}`",
                f"Suggestions written: `{out.relative_to(ROOT)}`",
                f"Rows scored: {len(rows)}",
                "",
                "These are pre-annotation suggestions only. They must not be treated as gold labels.",
                "",
                "## Vote prediction counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(vote_counts.items())],
                "",
                "## Vote confidence counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(confidence_counts.items())],
            ]
        ),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
