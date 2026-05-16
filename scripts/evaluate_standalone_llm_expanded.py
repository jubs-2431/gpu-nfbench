from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "evaluation" / "expanded_gold_llm_training_packet.jsonl"
OUT = ROOT / "evaluation" / "standalone_llm_expanded_predictions.csv"
METRICS = ROOT / "tables" / "standalone_llm_expanded_metrics.csv"
REPORT = ROOT / "reports" / "standalone_llm_expanded_eval.md"

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,}|[0-9]+")
PRIMARY_LABELS = [
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
]
SECONDARY_LABELS = [
    "memory_mask_bounds",
    "compiler_codegen",
    "async_race_ordering",
    "hardware_backend",
    "reduction_accumulation",
    "api_semantics",
    "environment_configuration",
    "unknown",
]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def trim(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def read_packet(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def stratified_holdout(rows: list[dict[str, Any]], holdout_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["output"]["primary_failure_label"]].append(row)
    train = []
    test = []
    for label, items in sorted(by_label.items()):
        items = sorted(items, key=lambda row: row["id"])
        n_test = max(1, round(len(items) * holdout_frac))
        for index, item in enumerate(items):
            if index % max(1, round(len(items) / n_test)) == 0 and len([r for r in test if r["output"]["primary_failure_label"] == label]) < n_test:
                test.append(item)
            else:
                train.append(item)
    return train, sorted(test, key=lambda row: row["id"])


class BM25Index:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.docs: list[Counter[str]] = []
        self.lengths: list[int] = []
        df: Counter[str] = Counter()
        for row in rows:
            counts = Counter(tokenize(row["input"])[:1600])
            self.docs.append(counts)
            self.lengths.append(sum(counts.values()))
            df.update(counts.keys())
        n_docs = max(1, len(rows))
        self.avgdl = sum(self.lengths) / n_docs
        self.idf = {term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query_terms: set[str], index: int) -> float:
        k1 = 1.5
        b = 0.75
        doc = self.docs[index]
        length = self.lengths[index]
        score = 0.0
        for term in query_terms:
            tf = doc.get(term, 0)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * length / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * (tf * (k1 + 1)) / denom
        return score

    def top_examples(self, query: str, total: int, per_label_cap: int) -> list[dict[str, Any]]:
        terms = set(tokenize(query)[:1600])
        scored = sorted(((self.score(terms, idx), row) for idx, row in enumerate(self.rows)), key=lambda item: item[0], reverse=True)
        selected = []
        counts: Counter[str] = Counter()
        for score, row in scored:
            if score <= 0:
                continue
            label = row["output"]["primary_failure_label"]
            if counts[label] >= per_label_cap:
                continue
            selected.append(row)
            counts[label] += 1
            if len(selected) >= total:
                break
        return selected


def build_prompt(target: dict[str, Any], examples: list[dict[str, Any]]) -> str:
    lines = [
        "You are a standalone LLM classifier for GPU numerical failure issue reports.",
        "Classify the target issue into exactly one primary label.",
        "",
        "Primary labels:",
        "- nan_inf: NaN or Inf values are the main reported symptom.",
        "- overflow_underflow: overflow, underflow, saturation, wraparound, or range behavior is central.",
        "- precision_tolerance: wrong values, tolerance mismatch, nondeterminism, or accuracy loss is central.",
        "- dtype_casting: dtype conversion, promotion, casting, mixed precision, unsupported dtype, or type-specific behavior is central.",
        "- crash_compile: crash, compiler failure, internal error, segfault, build failure, or runtime exception is central.",
        "- performance_only: speed, latency, throughput, memory use, benchmark performance, or optimization is central without correctness failure.",
        "- not_numerical_failure: feature request, install/environment question, docs/API discussion, refactor/task, or general maintenance without concrete numerical/performance failure.",
        "",
        "Secondary labels:",
        ", ".join(SECONDARY_LABELS),
        "",
        "Decision rules:",
        "- First decide whether the issue reports a concrete failure. If it only asks for a feature/API/doc/install/help request, choose not_numerical_failure.",
        "- Next decide whether the concrete failure is correctness, crash/compile, or performance.",
        "- Pick the main reported symptom, not every keyword.",
        "- API/feature requests are not_numerical_failure unless they report a concrete bug.",
        "- Performance regressions or slowdowns are performance_only unless wrong values are central.",
        "- If codegen causes wrong values, use precision_tolerance and put compiler_codegen as secondary.",
        "- If the main problem is build/compile/runtime failure, use crash_compile.",
        "- Return JSON only. No markdown and no prose after JSON.",
        "",
        "Few-shot training examples from the training split:",
    ]
    for index, example in enumerate(examples, start=1):
        output = example["output"]
        lines.extend(
            [
                f"Example {index}:",
                trim(example["input"], 900),
                "Label JSON:",
                json.dumps(
                    {
                        "primary_failure_label": output["primary_failure_label"],
                        "secondary_cause_labels": output["secondary_cause_labels"],
                        "is_true_numerical_failure": output["is_true_numerical_failure"],
                        "evidence_quote": output["evidence_quote"],
                        "confidence": "high",
                    },
                    sort_keys=True,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "Target issue:",
            trim(target["input"], 2400),
            "",
            "Return this JSON shape:",
            json.dumps(
                {
                    "primary_failure_label": "one allowed primary label",
                    "secondary_cause_labels": ["one or more allowed secondary labels"],
                    "is_true_numerical_failure": "yes|no|unclear",
                    "evidence_quote": "short quote from target issue",
                    "confidence": "high|medium|low",
                },
                sort_keys=True,
            ),
        ]
    )
    return "\n".join(lines)


def ollama_generate(model: str, prompt: str, host: str, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 260},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("response", ""))


def parse_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def normalize(parsed: dict[str, Any]) -> tuple[str, str, str, str, str]:
    primary = str(parsed.get("primary_failure_label", "")).strip()
    if primary not in PRIMARY_LABELS:
        primary = "needs_review"
    secondary_raw = parsed.get("secondary_cause_labels", ["unknown"])
    if isinstance(secondary_raw, str):
        secondary = [part.strip() for part in secondary_raw.replace(",", "|").split("|") if part.strip()]
    elif isinstance(secondary_raw, list):
        secondary = [str(part).strip() for part in secondary_raw if str(part).strip()]
    else:
        secondary = []
    secondary = [label for label in secondary if label in SECONDARY_LABELS] or ["unknown"]
    true_failure = str(parsed.get("is_true_numerical_failure", "unclear")).strip().lower()
    if true_failure in {"true", "1"}:
        true_failure = "yes"
    if true_failure in {"false", "0"}:
        true_failure = "no"
    if true_failure not in {"yes", "no", "unclear"}:
        true_failure = "unclear"
    confidence = str(parsed.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    evidence = str(parsed.get("evidence_quote", ""))[:500]
    return primary, "|".join(dict.fromkeys(secondary)), true_failure, evidence, confidence


def prf(labels: list[str], preds: list[str]) -> tuple[float, float]:
    if not labels:
        return 0.0, 0.0
    accuracy = sum(a == b for a, b in zip(labels, preds)) / len(labels)
    all_labels = sorted(set(labels) | set(preds))
    f1s = []
    for label in all_labels:
        tp = sum(a == label and b == label for a, b in zip(labels, preds))
        fp = sum(a != label and b == label for a, b in zip(labels, preds))
        fn = sum(a == label and b != label for a, b in zip(labels, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return accuracy, sum(f1s) / len(f1s) if f1s else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a standalone local LLM on the expanded GPU-NFBench gold set.")
    parser.add_argument("--model", default="gpu-nfbench-triage")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--examples", type=int, default=7)
    parser.add_argument("--per-label-cap", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rows = read_packet(PACKET)
    train, test = stratified_holdout(rows, args.holdout_frac)
    if args.limit:
        test = test[: args.limit]
    index = BM25Index(train)

    existing: dict[str, dict[str, str]] = {}
    if args.resume and args.out.exists():
        with args.out.open(newline="", encoding="utf-8") as fh:
            existing = {row["id"]: row for row in csv.DictReader(fh)}

    predictions: list[dict[str, object]] = list(existing.values())
    done = set(existing)
    fieldnames = [
        "id",
        "repository",
        "gold_primary_failure",
        "predicted_primary_failure",
        "secondary_cause_labels",
        "is_true_numerical_failure",
        "confidence",
        "evidence_quote",
        "retrieved_example_ids",
        "elapsed_seconds",
        "error",
        "raw_response",
    ]
    for row_number, target in enumerate(test, start=1):
        if target["id"] in done:
            continue
        examples = index.top_examples(target["input"], args.examples, args.per_label_cap)
        prompt = build_prompt(target, examples)
        started = time.time()
        raw = ""
        error = ""
        try:
            raw = ollama_generate(args.model, prompt, args.host, args.timeout)
            parsed = parse_response(raw)
            primary, secondary, true_failure, evidence, confidence = normalize(parsed)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:1000]
            primary, secondary, true_failure, evidence, confidence = "needs_review", "unknown", "unclear", "", "low"
        elapsed = time.time() - started
        predictions.append(
            {
                "id": target["id"],
                "repository": target["repository"],
                "gold_primary_failure": target["output"]["primary_failure_label"],
                "predicted_primary_failure": primary,
                "secondary_cause_labels": secondary,
                "is_true_numerical_failure": true_failure,
                "confidence": confidence,
                "evidence_quote": evidence,
                "retrieved_example_ids": "|".join(example["id"] for example in examples),
                "elapsed_seconds": f"{elapsed:.3f}",
                "error": error,
                "raw_response": raw[:1200],
            }
        )
        write_csv(args.out, predictions, fieldnames)
        print(f"{row_number}/{len(test)} {target['id']} gold={target['output']['primary_failure_label']} pred={primary} err={bool(error)}", file=sys.stderr)

    eval_rows = [row for row in predictions if str(row.get("predicted_primary_failure", ""))]
    labels = [str(row["gold_primary_failure"]) for row in eval_rows]
    preds = [str(row["predicted_primary_failure"]) for row in eval_rows]
    accuracy, macro_f1 = prf(labels, preds)
    answered = [row for row in eval_rows if row["predicted_primary_failure"] != "needs_review"]
    ans_acc, ans_macro = prf([str(row["gold_primary_failure"]) for row in answered], [str(row["predicted_primary_failure"]) for row in answered])

    metric_rows = [
        {
            "mode": "standalone_llm_holdout_full_coverage",
            "train_rows": len(train),
            "test_rows": len(test),
            "answered_rows": len(eval_rows),
            "coverage": f"{len(eval_rows) / len(test):.3f}" if test else "0.000",
            "accuracy": f"{accuracy:.3f}",
            "macro_f1": f"{macro_f1:.3f}",
        },
        {
            "mode": "standalone_llm_holdout_excluding_needs_review",
            "train_rows": len(train),
            "test_rows": len(test),
            "answered_rows": len(answered),
            "coverage": f"{len(answered) / len(test):.3f}" if test else "0.000",
            "accuracy": f"{ans_acc:.3f}",
            "macro_f1": f"{ans_macro:.3f}",
        },
    ]
    write_csv(METRICS, metric_rows, ["mode", "train_rows", "test_rows", "answered_rows", "coverage", "accuracy", "macro_f1"])

    counts = Counter(preds)
    gold_counts = Counter(labels)
    REPORT.write_text(
        "\n".join(
            [
                "# Standalone LLM Expanded Evaluation",
                "",
                f"Model: `{args.model}`",
                f"Training/RAG rows: {len(train)}",
                f"Held-out test rows: {len(test)}",
                f"Predictions: `{display_path(args.out)}`",
                "",
                "| mode | answered | coverage | accuracy | macro F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
                *[
                    f"| {row['mode']} | {row['answered_rows']} | {row['coverage']} | {row['accuracy']} | {row['macro_f1']} |"
                    for row in metric_rows
                ],
                "",
                "## Gold label counts in evaluated rows",
                "",
                *[f"- {label}: {count}" for label, count in sorted(gold_counts.items())],
                "",
                "## Predicted label counts",
                "",
                *[f"- {label}: {count}" for label, count in sorted(counts.items())],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
