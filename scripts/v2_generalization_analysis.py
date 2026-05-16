from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import gold_baseline_classifier as gbc
import train_expanded_gold_models as egm


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "reports"
V2 = ROOT / "data" / "processed" / "gold_benchmark_expanded_v2_canonical.csv"
MODEL_NAMES = ["bm25_knn", "naive_bayes", "tfidf_logistic", "tfidf_linear_svm", "bigram_tfidf_logistic"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric(y: list[str], p: list[str]) -> tuple[float, float]:
    _, acc, macro = gbc.prf(y, p)
    return acc, macro


def vote(preds: dict[str, list[str]], tie_order: list[str]) -> list[str]:
    out = []
    for i in range(len(next(iter(preds.values())))):
        counts = Counter(values[i] for values in preds.values())
        top = counts.most_common(1)[0][1]
        winners = {label for label, count in counts.items() if count == top}
        out.append(next((preds[name][i] for name in tie_order if preds[name][i] in winners), counts.most_common(1)[0][0]))
    return out


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None) if value else datetime.min


def enriched_rows() -> list[dict[str, str]]:
    egm.EXPANDED_GOLD = V2
    rows = egm.build_rows()
    seed_by_url = {row["url"]: row for row in read_csv(ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv")}
    expansion_by_id = {row["expansion_id"]: row for row in read_csv(ROOT / "annotation" / "gold_expansion_1000_repaired.csv")}
    out = []
    for row in rows:
        created_at = ""
        if row["blind_id"].startswith("EGNF"):
            created_at = expansion_by_id.get(row["blind_id"], {}).get("created_at", "")
        else:
            created_at = seed_by_url.get(row["url"], {}).get("created_at", "")
        if created_at:
            out.append({**row, "created_at": created_at})
    return out


def loro(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    labels = sorted({row["gold_primary_failure"] for row in rows})
    pred_maps = {"candidate_weak_label": {row["blind_id"]: row["candidate_primary_failure"] for row in rows}}
    by_repo = []
    for name in MODEL_NAMES:
        pred_maps[name] = {}
    pred_maps["expanded_gold_vote_ensemble"] = {}
    for repo in sorted({row["repository"] for row in rows}):
        train = [row for row in rows if row["repository"] != repo]
        test = [row for row in rows if row["repository"] == repo]
        local = {}
        for name in MODEL_NAMES:
            p = gbc.predict_model(name, train, test, labels)
            local[name] = p
            for row, pred in zip(test, p):
                pred_maps[name][row["blind_id"]] = pred
            acc, macro = metric([row["gold_primary_failure"] for row in test], p)
            by_repo.append({"model_or_mode": name, "held_out_repository": repo, "test_issues": len(test), "accuracy": f"{acc:.3f}", "macro_f1": f"{macro:.3f}"})
        ens = vote(
            {
                "candidate_weak_label": [row["candidate_primary_failure"] for row in test],
                "tfidf_linear_svm": local["tfidf_linear_svm"],
                "tfidf_logistic": local["tfidf_logistic"],
                "bigram_tfidf_logistic": local["bigram_tfidf_logistic"],
                "naive_bayes": local["naive_bayes"],
            },
            ["tfidf_linear_svm", "tfidf_logistic", "candidate_weak_label", "bigram_tfidf_logistic", "naive_bayes"],
        )
        for row, pred in zip(test, ens):
            pred_maps["expanded_gold_vote_ensemble"][row["blind_id"]] = pred
        acc, macro = metric([row["gold_primary_failure"] for row in test], ens)
        by_repo.append({"model_or_mode": "expanded_gold_vote_ensemble", "held_out_repository": repo, "test_issues": len(test), "accuracy": f"{acc:.3f}", "macro_f1": f"{macro:.3f}"})
    summary = []
    y = [row["gold_primary_failure"] for row in rows]
    for name, pmap in pred_maps.items():
        p = [pmap[row["blind_id"]] for row in rows]
        acc, macro = metric(y, p)
        summary.append({"model_or_mode": name, "evaluation": "leave_one_repository_out", "answered_rows": len(rows), "coverage": "1.000", "accuracy": f"{acc:.3f}", "macro_f1": f"{macro:.3f}"})
    write_csv(TABLE_DIR / "v2_gold_loro_metrics.csv", summary, ["model_or_mode", "evaluation", "answered_rows", "coverage", "accuracy", "macro_f1"])
    write_csv(TABLE_DIR / "v2_gold_loro_by_repo.csv", by_repo, ["model_or_mode", "held_out_repository", "test_issues", "accuracy", "macro_f1"])
    return summary


def time_split(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: (parse_dt(row["created_at"]), row["blind_id"]))
    split = int(len(ordered) * 0.8)
    train, test = ordered[:split], ordered[split:]
    labels = sorted({row["gold_primary_failure"] for row in ordered})
    preds = {"candidate_weak_label": [row["candidate_primary_failure"] for row in test]}
    for name in MODEL_NAMES:
        preds[name] = gbc.predict_model(name, train, test, labels)
    preds["expanded_gold_vote_ensemble"] = vote(
        {
            "candidate_weak_label": preds["candidate_weak_label"],
            "tfidf_linear_svm": preds["tfidf_linear_svm"],
            "tfidf_logistic": preds["tfidf_logistic"],
            "bigram_tfidf_logistic": preds["bigram_tfidf_logistic"],
            "naive_bayes": preds["naive_bayes"],
        },
        ["tfidf_linear_svm", "tfidf_logistic", "candidate_weak_label", "bigram_tfidf_logistic", "naive_bayes"],
    )
    y = [row["gold_primary_failure"] for row in test]
    out = []
    for name, p in preds.items():
        acc, macro = metric(y, p)
        out.append({"model_or_mode": name, "evaluation": "chronological_80_20", "train_rows": len(train), "test_rows": len(test), "train_end_created_at": train[-1]["created_at"], "test_start_created_at": test[0]["created_at"], "accuracy": f"{acc:.3f}", "macro_f1": f"{macro:.3f}"})
    write_csv(TABLE_DIR / "v2_gold_time_split_metrics.csv", out, ["model_or_mode", "evaluation", "train_rows", "test_rows", "train_end_created_at", "test_start_created_at", "accuracy", "macro_f1"])
    return out


def main() -> None:
    rows = enriched_rows()
    loro_rows = loro(rows)
    time_rows = time_split(rows)
    REPORT_DIR.joinpath("v2_gold_generalization.md").write_text(
        "\n".join(
            [
                "# V2 Gold Generalization",
                "",
                "## Leave-one-repository-out",
                "",
                "| model/mode | accuracy | macro F1 |",
                "| --- | ---: | ---: |",
                *[f"| {row['model_or_mode']} | {row['accuracy']} | {row['macro_f1']} |" for row in loro_rows],
                "",
                "## Chronological 80/20",
                "",
                "| model/mode | accuracy | macro F1 |",
                "| --- | ---: | ---: |",
                *[f"| {row['model_or_mode']} | {row['accuracy']} | {row['macro_f1']} |" for row in time_rows],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(REPORT_DIR / "v2_gold_generalization.md")


if __name__ == "__main__":
    main()
