from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "gpu_numerical_issue_seed.csv"
TABLE_DIR = ROOT / "tables"
REPORT = ROOT / "reports" / "silver_label_classifier.md"
TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_+\-.]{1,}")


def read_rows() -> list[dict[str, str]]:
    with DATA.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text) if len(tok) <= 40]


def split_key(url: str) -> int:
    return int(sha256(url.encode("utf-8")).hexdigest()[:8], 16) % 10


def train_nb(rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, Counter[str]], Counter[str], set[str]]:
    label_counts: Counter[str] = Counter()
    word_counts: dict[str, Counter[str]] = defaultdict(Counter)
    vocab: set[str] = set()
    for row in rows:
        label = row["candidate_primary_failure"]
        text = f"{row['title']} {row['body_excerpt']}"
        tokens = tokenize(text)
        label_counts[label] += 1
        word_counts[label].update(tokens)
        vocab.update(tokens)
    total = sum(label_counts.values())
    priors = {label: math.log(count / total) for label, count in label_counts.items()}
    return priors, word_counts, label_counts, vocab


def predict(
    row: dict[str, str],
    priors: dict[str, float],
    word_counts: dict[str, Counter[str]],
    label_counts: Counter[str],
    vocab: set[str],
) -> str:
    tokens = tokenize(f"{row['title']} {row['body_excerpt']}")
    vocab_size = max(len(vocab), 1)
    best_label = ""
    best_score = -float("inf")
    for label, prior in priors.items():
        total_words = sum(word_counts[label].values())
        denom = total_words + vocab_size
        score = prior
        for tok in tokens:
            score += math.log((word_counts[label][tok] + 1) / denom)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    rows = [row for row in read_rows() if row["candidate_primary_failure"] != "needs_review"]
    train = [row for row in rows if split_key(row["url"]) < 8]
    test = [row for row in rows if split_key(row["url"]) >= 8]
    priors, word_counts, label_counts, vocab = train_nb(train)
    majority = label_counts.most_common(1)[0][0]

    labels = sorted(label_counts)
    confusion: dict[str, Counter[str]] = {label: Counter() for label in labels}
    correct = 0
    majority_correct = 0
    for row in test:
        gold = row["candidate_primary_failure"]
        pred = predict(row, priors, word_counts, label_counts, vocab)
        confusion[gold][pred] += 1
        correct += int(pred == gold)
        majority_correct += int(majority == gold)

    metrics = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        metrics.append(
            {
                "label": label,
                "support": sum(confusion[label].values()),
                "f1": round(f1(tp, fp, fn), 3),
            }
        )
    accuracy = correct / len(test) if test else 0.0
    majority_accuracy = majority_correct / len(test) if test else 0.0
    macro_f1 = sum(row["f1"] for row in metrics) / len(metrics) if metrics else 0.0

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with (TABLE_DIR / "silver_classifier_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["label", "support", "f1"])
        writer.writeheader()
        writer.writerows(metrics)
    with (TABLE_DIR / "silver_classifier_confusion.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gold\\pred", *labels])
        for label in labels:
            writer.writerow([label, *[confusion[label][pred] for pred in labels]])

    lines = [
        "# Silver-Label Classifier",
        "",
        "This experiment predicts the agent-assisted silver primary label from issue title and body excerpt. It is not a gold-standard evaluation.",
        "",
        f"- Train issues: {len(train)}",
        f"- Test issues: {len(test)}",
        f"- Majority baseline accuracy: {majority_accuracy:.3f}",
        f"- Naive Bayes silver-label accuracy: {accuracy:.3f}",
        f"- Naive Bayes macro F1: {macro_f1:.3f}",
        "",
        "| label | support | F1 |",
        "| --- | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(f"| {row['label']} | {row['support']} | {row['f1']:.3f} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

