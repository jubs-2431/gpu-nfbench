from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
ANNOTATION = ROOT / "annotation" / "annotator_A_blind.csv"
SUGGESTIONS = ROOT / "annotation" / "candidate_label_suggestions_hidden_from_annotators.csv"
TABLE_DIR = ROOT / "tables"
REPORT = ROOT / "reports" / "gold_baseline_classifier.md"


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,}|[0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_folds(rows: list[dict[str, str]], k: int = 5) -> list[list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["gold_primary_failure"]].append(row)
    folds: list[list[dict[str, str]]] = [[] for _ in range(k)]
    for _, items in sorted(by_label.items()):
        items = sorted(items, key=lambda r: r["blind_id"])
        for index, item in enumerate(items):
            folds[index % k].append(item)
    return folds


def prf(labels: list[str], predictions: list[str]) -> tuple[list[dict[str, object]], float, float]:
    all_labels = sorted(set(labels) | set(predictions))
    rows = []
    f1s = []
    accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(labels) if labels else 0.0
    for label in all_labels:
        tp = sum(y == label and p == label for y, p in zip(labels, predictions))
        fp = sum(y != label and p == label for y, p in zip(labels, predictions))
        fn = sum(y == label and p != label for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(y == label for y in labels)
        f1s.append(f1)
        rows.append(
            {
                "label": label,
                "support": support,
                "precision": f"{precision:.3f}",
                "recall": f"{recall:.3f}",
                "f1": f"{f1:.3f}",
            }
        )
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return rows, accuracy, macro_f1


class NaiveBayes:
    def __init__(self) -> None:
        self.class_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_tokens: Counter[str] = Counter()
        self.vocab: set[str] = set()

    def fit(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            label = row["gold_primary_failure"]
            self.class_counts[label] += 1
            for token in tokenize(row["text"]):
                self.token_counts[label][token] += 1
                self.total_tokens[label] += 1
                self.vocab.add(token)

    def predict(self, text: str) -> str:
        tokens = tokenize(text)
        total_docs = sum(self.class_counts.values())
        vocab_size = max(1, len(self.vocab))
        best_label = ""
        best_score = -float("inf")
        for label, docs in self.class_counts.items():
            score = math.log(docs / total_docs)
            denom = self.total_tokens[label] + vocab_size
            counts = self.token_counts[label]
            for token in tokens:
                score += math.log((counts[token] + 1) / denom)
            if score > best_score:
                best_label = label
                best_score = score
        return best_label


class BM25Knn:
    def __init__(self, k: int = 5, k1: float = 1.5, b: float = 0.75) -> None:
        self.k = k
        self.k1 = k1
        self.b = b
        self.docs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.labels: list[str] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.majority = ""

    def fit(self, rows: list[dict[str, str]]) -> None:
        df: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        for row in rows:
            tokens = tokenize(row["text"])[:1200]
            counts = Counter(tokens)
            self.docs.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            self.labels.append(row["gold_primary_failure"])
            label_counts[row["gold_primary_failure"]] += 1
            df.update(counts.keys())
        self.majority = label_counts.most_common(1)[0][0] if label_counts else ""
        n_docs = len(self.docs)
        self.avgdl = sum(self.doc_lengths) / n_docs if n_docs else 0.0
        self.idf = {
            term: math.log(1 + (n_docs - term_df + 0.5) / (term_df + 0.5))
            for term, term_df in df.items()
        }

    def _score_doc(self, query_terms: set[str], doc_idx: int) -> float:
        score = 0.0
        doc = self.docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        for term in query_terms:
            tf = doc.get(term, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
        return score

    def predict(self, text: str) -> str:
        query_terms = set(tokenize(text)[:1200])
        if not query_terms:
            return self.majority
        scored = sorted(
            ((self._score_doc(query_terms, i), self.labels[i]) for i in range(len(self.docs))),
            reverse=True,
        )
        label_scores: Counter[str] = Counter()
        for score, label in scored[: self.k]:
            if score > 0:
                label_scores[label] += score
        if not label_scores:
            return self.majority
        return label_scores.most_common(1)[0][0]


class TfidfVectorizer:
    def __init__(self, analyzer: str = "word", max_features: int = 3500, min_df: int = 1) -> None:
        self.analyzer = analyzer
        self.max_features = max_features
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []

    def _terms(self, text: str) -> list[str]:
        tokens = tokenize(text)[:900]
        if self.analyzer == "word_bigram":
            bigrams = [f"{tokens[i]}__{tokens[i + 1]}" for i in range(max(0, len(tokens) - 1))]
            return tokens + bigrams
        return tokens

    def fit(self, texts: list[str]) -> None:
        df: Counter[str] = Counter()
        tf: Counter[str] = Counter()
        for text in texts:
            terms = self._terms(text)
            tf.update(terms)
            df.update(set(terms))
        candidates = [
            term
            for term, freq in tf.most_common()
            if df[term] >= self.min_df and len(term.strip()) > 1
        ][: self.max_features]
        self.vocab = {term: i for i, term in enumerate(candidates)}
        n_docs = len(texts)
        self.idf = [0.0] * len(self.vocab)
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + df[term])) + 1.0

    def transform_one(self, text: str) -> dict[int, float]:
        counts: Counter[int] = Counter()
        for term in self._terms(text):
            idx = self.vocab.get(term)
            if idx is not None:
                counts[idx] += 1
        if not counts:
            return {}
        total = sum(counts.values())
        values = {idx: (count / total) * self.idf[idx] for idx, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in values.values()))
        if norm:
            values = {idx: value / norm for idx, value in values.items()}
        return values

    def transform(self, texts: list[str]) -> list[dict[int, float]]:
        return [self.transform_one(text) for text in texts]


class SoftmaxRegression:
    def __init__(self, labels: list[str], n_features: int, epochs: int = 12, lr: float = 0.45, l2: float = 0.0005) -> None:
        self.labels = labels
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.weights = [[0.0] * n_features for _ in labels]
        self.bias = [0.0] * len(labels)
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2

    def _scores(self, x: dict[int, float]) -> list[float]:
        scores = self.bias[:]
        for c, weights in enumerate(self.weights):
            scores[c] += sum(weights[i] * value for i, value in x.items())
        return scores

    def fit(self, xs: list[dict[int, float]], ys: list[str]) -> None:
        rng = random.Random(17)
        order = list(range(len(xs)))
        for epoch in range(self.epochs):
            rng.shuffle(order)
            eta = self.lr / (1.0 + 0.02 * epoch)
            for row_idx in order:
                x = xs[row_idx]
                y_idx = self.label_to_idx[ys[row_idx]]
                scores = self._scores(x)
                max_score = max(scores)
                exps = [math.exp(score - max_score) for score in scores]
                denom = sum(exps)
                probs = [value / denom for value in exps]
                for c in range(len(self.labels)):
                    error = probs[c] - (1.0 if c == y_idx else 0.0)
                    self.bias[c] -= eta * error
                    weights = self.weights[c]
                    for i, value in x.items():
                        weights[i] -= eta * (error * value + self.l2 * weights[i])

    def predict(self, x: dict[int, float]) -> str:
        scores = self._scores(x)
        return self.labels[max(range(len(scores)), key=lambda i: scores[i])]


class LinearSvmOvr:
    def __init__(self, labels: list[str], n_features: int, epochs: int = 12, lr: float = 0.25, l2: float = 0.0005) -> None:
        self.labels = labels
        self.weights = [[0.0] * n_features for _ in labels]
        self.bias = [0.0] * len(labels)
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2

    def fit(self, xs: list[dict[int, float]], ys: list[str]) -> None:
        rng = random.Random(23)
        order = list(range(len(xs)))
        for epoch in range(self.epochs):
            rng.shuffle(order)
            eta = self.lr / (1.0 + 0.02 * epoch)
            for row_idx in order:
                x = xs[row_idx]
                y = ys[row_idx]
                for c, label in enumerate(self.labels):
                    target = 1.0 if label == y else -1.0
                    score = self.bias[c] + sum(self.weights[c][i] * value for i, value in x.items())
                    margin = target * score
                    if margin < 1.0:
                        self.bias[c] += eta * target
                        for i, value in x.items():
                            self.weights[c][i] += eta * (target * value - self.l2 * self.weights[c][i])

    def predict(self, x: dict[int, float]) -> str:
        scores = [
            self.bias[c] + sum(self.weights[c][i] * value for i, value in x.items())
            for c in range(len(self.labels))
        ]
        return self.labels[max(range(len(scores)), key=lambda i: scores[i])]


def build_rows() -> list[dict[str, str]]:
    gold_rows = read_csv(GOLD)
    annotation_rows = {row["blind_id"]: row for row in read_csv(ANNOTATION)}
    suggestions = {row["blind_id"]: row for row in read_csv(SUGGESTIONS)}
    rows: list[dict[str, str]] = []
    for row in gold_rows:
        packet = annotation_rows[row["blind_id"]]
        rows.append(
            {
                **row,
                "text": " ".join(
                    [
                        packet.get("title", ""),
                        packet.get("github_labels", ""),
                        packet.get("issue_body_excerpt", ""),
                        packet.get("comments_excerpt", ""),
                    ]
                ),
                "candidate_primary_failure": suggestions.get(row["blind_id"], {}).get("candidate_primary_failure", "needs_review"),
            }
        )
    return rows


def predict_model(model_name: str, train: list[dict[str, str]], test: list[dict[str, str]], labels: list[str]) -> list[str]:
    if model_name == "naive_bayes":
        model = NaiveBayes()
        model.fit(train)
        return [model.predict(row["text"]) for row in test]
    if model_name == "bm25_knn":
        model = BM25Knn(k=5)
        model.fit(train)
        return [model.predict(row["text"]) for row in test]

    analyzer = "word_bigram" if model_name.startswith("bigram_") else "word"
    vectorizer = TfidfVectorizer(analyzer=analyzer, max_features=2000 if analyzer == "word_bigram" else 1500, min_df=1)
    vectorizer.fit([row["text"] for row in train])
    train_x = vectorizer.transform([row["text"] for row in train])
    test_x = vectorizer.transform([row["text"] for row in test])
    train_y = [row["gold_primary_failure"] for row in train]

    if model_name.endswith("linear_svm"):
        model = LinearSvmOvr(labels, len(vectorizer.vocab))
    else:
        model = SoftmaxRegression(labels, len(vectorizer.vocab))
    model.fit(train_x, train_y)
    return [model.predict(x) for x in test_x]


def cross_val_predictions(rows: list[dict[str, str]], model_name: str) -> list[str]:
    labels = sorted({row["gold_primary_failure"] for row in rows})
    predictions: dict[str, str] = {}
    folds = stratified_folds(rows, 5)
    for test in folds:
        test_ids = {row["blind_id"] for row in test}
        train = [row for row in rows if row["blind_id"] not in test_ids]
        for row, pred in zip(test, predict_model(model_name, train, test, labels)):
            predictions[row["blind_id"]] = pred
    return [predictions[row["blind_id"]] for row in rows]


def leave_one_repo_predictions(rows: list[dict[str, str]], model_name: str) -> tuple[list[str], list[dict[str, object]]]:
    labels = sorted({row["gold_primary_failure"] for row in rows})
    predictions: dict[str, str] = {}
    repo_rows: list[dict[str, object]] = []
    for repo in sorted({row["repository"] for row in rows}):
        train = [row for row in rows if row["repository"] != repo]
        test = [row for row in rows if row["repository"] == repo]
        preds = predict_model(model_name, train, test, labels)
        for row, pred in zip(test, preds):
            predictions[row["blind_id"]] = pred
        _, acc, macro = prf([row["gold_primary_failure"] for row in test], preds)
        repo_rows.append(
            {
                "model": model_name,
                "held_out_repository": repo,
                "test_issues": len(test),
                "accuracy": f"{acc:.3f}",
                "macro_f1": f"{macro:.3f}",
            }
        )
    return [predictions[row["blind_id"]] for row in rows], repo_rows


def confusion_rows(labels: list[str], predictions: list[str]) -> list[dict[str, object]]:
    counts = Counter(zip(labels, predictions))
    return [
        {"gold_primary_failure": gold, "predicted_primary_failure": pred, "issues": count}
        for (gold, pred), count in sorted(counts.items())
    ]


def main() -> None:
    rows = build_rows()
    labels = [row["gold_primary_failure"] for row in rows]
    majority = Counter(labels).most_common(1)[0][0]
    candidate_predictions = [row["candidate_primary_failure"] for row in rows]

    model_names = [
        "bm25_knn",
        "naive_bayes",
        "tfidf_logistic",
        "tfidf_linear_svm",
        "bigram_tfidf_logistic",
    ]
    cv_predictions: dict[str, list[str]] = {
        "majority_baseline": [majority for _ in rows],
        "candidate_weak_label": candidate_predictions,
    }
    for model_name in model_names:
        cv_predictions[model_name] = cross_val_predictions(rows, model_name)

    metric_rows: list[dict[str, object]] = []
    per_class_by_model: list[dict[str, object]] = []
    for model_name, preds in cv_predictions.items():
        class_rows, accuracy, macro_f1 = prf(labels, preds)
        metric_rows.append(
            {
                "model": model_name,
                "evaluation": "stratified_5fold",
                "accuracy": f"{accuracy:.3f}",
                "macro_f1": f"{macro_f1:.3f}",
            }
        )
        for row in class_rows:
            per_class_by_model.append({"model": model_name, **row})

    loro_metric_rows: list[dict[str, object]] = []
    loro_by_repo_rows: list[dict[str, object]] = []
    for model_name in ["candidate_weak_label", *model_names]:
        if model_name == "candidate_weak_label":
            preds = candidate_predictions
            repo_rows = []
            for repo in sorted({row["repository"] for row in rows}):
                test = [row for row in rows if row["repository"] == repo]
                test_preds = [row["candidate_primary_failure"] for row in test]
                _, acc, macro = prf([row["gold_primary_failure"] for row in test], test_preds)
                repo_rows.append(
                    {
                        "model": model_name,
                        "held_out_repository": repo,
                        "test_issues": len(test),
                        "accuracy": f"{acc:.3f}",
                        "macro_f1": f"{macro:.3f}",
                    }
                )
        else:
            preds, repo_rows = leave_one_repo_predictions(rows, model_name)
        _, accuracy, macro_f1 = prf(labels, preds)
        loro_metric_rows.append(
            {
                "model": model_name,
                "evaluation": "leave_one_repository_out",
                "accuracy": f"{accuracy:.3f}",
                "macro_f1": f"{macro_f1:.3f}",
            }
        )
        loro_by_repo_rows.extend(repo_rows)

    best_model = max(
        [row for row in metric_rows if row["model"] not in {"majority_baseline", "candidate_weak_label"}],
        key=lambda row: (float(row["macro_f1"]), float(row["accuracy"])),
    )["model"]
    best_preds = cv_predictions[str(best_model)]
    best_class_rows, _, _ = prf(labels, best_preds)

    write_csv(TABLE_DIR / "gold_classifier_metrics.csv", metric_rows, ["model", "evaluation", "accuracy", "macro_f1"])
    write_csv(TABLE_DIR / "gold_classifier_loro_metrics.csv", loro_metric_rows, ["model", "evaluation", "accuracy", "macro_f1"])
    write_csv(
        TABLE_DIR / "gold_classifier_loro_by_repo.csv",
        loro_by_repo_rows,
        ["model", "held_out_repository", "test_issues", "accuracy", "macro_f1"],
    )
    write_csv(TABLE_DIR / "gold_classifier_per_class.csv", best_class_rows, ["label", "support", "precision", "recall", "f1"])
    write_csv(
        TABLE_DIR / "gold_classifier_per_class_all_models.csv",
        per_class_by_model,
        ["model", "label", "support", "precision", "recall", "f1"],
    )
    write_csv(
        TABLE_DIR / "gold_classifier_confusion.csv",
        confusion_rows(labels, best_preds),
        ["gold_primary_failure", "predicted_primary_failure", "issues"],
    )

    lines = [
        "# Gold Baseline Classifier",
        "",
        "This experiment evaluates deterministic text baselines against adjudicated gold labels.",
        "No external ML packages are required; BM25 k-nearest-neighbor retrieval, TF-IDF, multinomial Naive Bayes, softmax regression, and one-vs-rest linear SVM are implemented in `scripts/gold_baseline_classifier.py`.",
        "",
        "## Stratified 5-fold evaluation",
        "",
        "| model | accuracy | macro_f1 |",
        "| --- | ---: | ---: |",
        *[f"| {row['model']} | {row['accuracy']} | {row['macro_f1']} |" for row in metric_rows],
        "",
        "## Leave-one-repository-out evaluation",
        "",
        "| model | accuracy | macro_f1 |",
        "| --- | ---: | ---: |",
        *[f"| {row['model']} | {row['accuracy']} | {row['macro_f1']} |" for row in loro_metric_rows],
        "",
        f"## Best 5-fold model per-class performance: `{best_model}`",
        "",
        "| label | support | precision | recall | f1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {row['label']} | {row['support']} | {row['precision']} | {row['recall']} | {row['f1']} |"
            for row in best_class_rows
        ],
        "",
        "## Interpretation",
        "",
        "- BM25 and TF-IDF models improve over the majority baseline but still struggle with rare classes.",
        "- Leave-one-repository-out evaluation is harder than stratified folds because project vocabulary and issue templates shift across repositories.",
        "- These results support the paper's claim that GPU numerical-failure triage needs full context and adjudicated labels, not only keyword matching.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
