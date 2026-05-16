from __future__ import annotations

import csv
from pathlib import Path

import gold_baseline_classifier as gbc


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "annotation"
REPORTS = ROOT / "reports"
OUT = ANNOTATION / "full_coverage_expansion_model_suggestions.csv"
REPORT = REPORTS / "expansion_model_suggestions.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expansion_text(row: dict[str, str]) -> str:
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
    train = gbc.build_rows()
    labels = sorted({row["gold_primary_failure"] for row in train})
    expansion = read_csv(ANNOTATION / "full_coverage_expansion_review.csv")
    test = [
        {
            "blind_id": row["expansion_id"],
            "repository": row["repository"],
            "title": row["title"],
            "text": expansion_text(row),
            "gold_primary_failure": labels[0],
        }
        for row in expansion
    ]

    svm = gbc.predict_model("tfidf_linear_svm", train, test, labels)
    logistic = gbc.predict_model("tfidf_logistic", train, test, labels)
    bigram = gbc.predict_model("bigram_tfidf_logistic", train, test, labels)
    nb = gbc.predict_model("naive_bayes", train, test, labels)

    rows = []
    for row, p_svm, p_log, p_bigram, p_nb in zip(expansion, svm, logistic, bigram, nb):
        votes = [p_svm, p_log, p_bigram, p_nb, row.get("candidate_primary_failure", "")]
        vote_counts = {label: votes.count(label) for label in sorted(set(votes)) if label}
        top_label = max(vote_counts, key=vote_counts.get)
        confidence = "high" if vote_counts[top_label] >= 4 else "medium" if vote_counts[top_label] >= 3 else "low"
        rows.append(
            {
                "expansion_id": row["expansion_id"],
                "repository": row["repository"],
                "title": row["title"],
                "url": row["url"],
                "candidate_primary_failure": row.get("candidate_primary_failure", ""),
                "svm_prediction": p_svm,
                "logistic_prediction": p_log,
                "bigram_prediction": p_bigram,
                "naive_bayes_prediction": p_nb,
                "vote_prediction": top_label,
                "vote_confidence": confidence,
                "human_gold_status": "suggestion_only_not_gold",
            }
        )

    write_csv(
        OUT,
        rows,
        [
            "expansion_id",
            "repository",
            "title",
            "url",
            "candidate_primary_failure",
            "svm_prediction",
            "logistic_prediction",
            "bigram_prediction",
            "naive_bayes_prediction",
            "vote_prediction",
            "vote_confidence",
            "human_gold_status",
        ],
    )

    counts = {}
    for row in rows:
        counts[row["vote_prediction"]] = counts.get(row["vote_prediction"], 0) + 1
    lines = [
        "# Expansion Model Suggestions",
        "",
        "The current 191-row gold benchmark was used to train deterministic text models, then predict labels for the 215-row expansion packet.",
        "These are suggestions for faster human review, not gold labels.",
        "",
        f"- Suggestions written: `{OUT.relative_to(ROOT)}`",
        f"- Expansion rows scored: {len(rows)}",
        "",
        "Vote prediction counts:",
    ]
    for label, count in sorted(counts.items()):
        lines.append(f"- {label}: {count}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
