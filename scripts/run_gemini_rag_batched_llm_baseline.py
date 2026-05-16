from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "gold_benchmark.csv"
PACKET = ROOT / "annotation" / "annotator_A_blind.csv"
SUGGESTIONS = ROOT / "annotation" / "candidate_label_suggestions_hidden_from_annotators.csv"
OUT_DIR = ROOT / "evaluation"

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,}|[0-9]+")

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

LABEL_DEFINITIONS = {
    "nan_inf": "NaN or Inf values are the central reported symptom.",
    "overflow_underflow": "Overflow, underflow, saturation, wraparound, or numeric range behavior is central.",
    "precision_tolerance": "Wrong values, tolerance mismatch, nondeterminism, or accuracy loss is central, without explicit NaN/Inf or overflow as the main symptom.",
    "dtype_casting": "Dtype conversion, promotion, casting, mixed precision handling, unsupported dtype, or type-specific behavior is central.",
    "crash_compile": "The main report is a crash, compiler failure, internal error, segfault, build failure, or runtime exception.",
    "performance_only": "The main report is speed, latency, throughput, memory use, benchmark performance, or optimization, not correctness.",
    "not_numerical_failure": "The issue is unrelated to numerical failure, such as feature request, install/environment question, documentation/API discussion, or general maintenance.",
    "needs_review": "The supplied text is too ambiguous to classify confidently.",
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


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def trim(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    return value[: limit - 3] + "..." if len(value) > limit else value


def issue_text(row: dict[str, str]) -> str:
    return "\n".join(
        [
            row.get("repository", ""),
            row.get("title", ""),
            row.get("github_labels", ""),
            row.get("issue_body_excerpt", ""),
            row.get("comments_excerpt", ""),
        ]
    )


def stratified_folds(rows: list[dict[str, str]], k: int = 5) -> list[list[str]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["gold_primary_failure"]].append(row)
    folds: list[list[str]] = [[] for _ in range(k)]
    for _, items in sorted(by_label.items()):
        for index, item in enumerate(sorted(items, key=lambda r: r["blind_id"])):
            folds[index % k].append(item["blind_id"])
    return folds


class BM25Index:
    def __init__(self, rows: list[dict[str, str]], text_by_id: dict[str, str]) -> None:
        self.rows = rows
        self.docs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.idf: dict[str, float] = {}
        df: Counter[str] = Counter()
        for row in rows:
            counts = Counter(tokenize(text_by_id[row["blind_id"]])[:1200])
            self.docs.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            df.update(counts.keys())
        n_docs = max(1, len(self.docs))
        self.avgdl = sum(self.doc_lengths) / n_docs
        self.idf = {term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query_terms: set[str], index: int) -> float:
        k1 = 1.5
        b = 0.75
        doc = self.docs[index]
        doc_len = self.doc_lengths[index]
        score = 0.0
        for term in query_terms:
            tf = doc.get(term, 0)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * (tf * (k1 + 1)) / denom
        return score

    def top_examples(self, query: str, per_label_cap: int, total_cap: int) -> list[tuple[dict[str, str], float]]:
        terms = set(tokenize(query)[:1200])
        scored = sorted(((self.score(terms, i), row) for i, row in enumerate(self.rows)), key=lambda x: x[0], reverse=True)
        selected: list[tuple[dict[str, str], float]] = []
        label_counts: Counter[str] = Counter()
        for score, row in scored:
            if score <= 0:
                continue
            label = row["gold_primary_failure"]
            if label_counts[label] >= per_label_cap:
                continue
            selected.append((row, score))
            label_counts[label] += 1
            if len(selected) >= total_cap:
                break
        return selected


def retry_delay(error: str, attempt: int) -> float:
    if "retry_after_seconds" in error:
        try:
            after = error.split('"retry_after_seconds":', 1)[1].split(",", 1)[0].split("}", 1)[0]
            return max(float(after) + 2.0, 2.0)
        except (IndexError, ValueError):
            pass
    if "Please retry in" in error:
        try:
            after = error.split("Please retry in", 1)[1].split("s", 1)[0].strip()
            return max(float(after) + 2.0, 2.0)
        except (IndexError, ValueError):
            pass
    return min(90.0, 6.0 * attempt)


def post_gemini(api_key: str, model: str, prompt: str, max_tokens: int, timeout: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1800]}") from exc
    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Gemini response missing text: {json.dumps(body)[:1800]}") from exc


def parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
            return parsed.get("predictions", parsed)
        raise


def normalize(raw: dict[str, Any], blind_id: str, model: str, elapsed: float, error: str = "", raw_response: str = "") -> dict[str, str]:
    primary = str(raw.get("primary_failure_label", "needs_review")).strip()
    if primary not in PRIMARY_LABELS:
        primary = "needs_review"
    secondary_raw = raw.get("secondary_cause_labels", ["unknown"])
    if isinstance(secondary_raw, str):
        secondary = [part.strip() for part in secondary_raw.replace(",", "|").split("|") if part.strip()]
    else:
        secondary = [str(part).strip() for part in secondary_raw if str(part).strip()]
    secondary = [label for label in secondary if label in SECONDARY_LABELS] or ["unknown"]
    true_raw = raw.get("is_true_numerical_failure", "unclear")
    if isinstance(true_raw, bool):
        true_failure = "yes" if true_raw else "no"
    else:
        true_failure = str(true_raw).strip().lower()
    if true_failure not in {"yes", "no", "unclear"}:
        true_failure = "unclear"
    confidence_raw = raw.get("confidence", "low")
    if isinstance(confidence_raw, (int, float)):
        confidence = "high" if confidence_raw >= 0.75 else "medium" if confidence_raw >= 0.4 else "low"
    else:
        confidence = str(confidence_raw).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "blind_id": blind_id,
        "primary_failure_label": primary,
        "secondary_cause_labels": "|".join(dict.fromkeys(secondary)),
        "is_true_numerical_failure": true_failure,
        "evidence_quote": str(raw.get("evidence_quote", ""))[:500],
        "confidence": confidence,
        "model": model,
        "provider": "gemini_fold_safe_rag_batched",
        "elapsed_seconds": f"{elapsed:.3f}",
        "error": error,
        "raw_response": raw_response[:2000],
    }


def batch_prompt(
    batch: list[dict[str, str]],
    examples_by_id: dict[str, list[tuple[dict[str, str], float]]],
    packet_by_id: dict[str, dict[str, str]],
    suggestions: dict[str, dict[str, str]],
) -> str:
    lines = [
        "You classify public GPU/kernel issue reports for a research benchmark.",
        "Use only the target issue text, taxonomy definitions, and retrieved examples.",
        "The weak pre-classifier suggestion is noisy and can be wrong.",
        "Return JSON only: an array of objects, one per target row.",
        "Each object must include blind_id, primary_failure_label, secondary_cause_labels, is_true_numerical_failure, evidence_quote, confidence.",
        "",
        "Allowed primary labels:",
        ", ".join(PRIMARY_LABELS),
        "",
        "Primary label definitions:",
    ]
    for label in PRIMARY_LABELS:
        lines.append(f"- {label}: {LABEL_DEFINITIONS[label]}")
    lines.extend(
        [
            "",
            "Allowed secondary cause labels:",
            ", ".join(SECONDARY_LABELS),
            "",
            "Decision rules:",
            "- Label the main failure symptom, not every term mentioned.",
            "- If the report is mainly an API request, feature request, install problem, documentation question, or general maintenance, choose not_numerical_failure.",
            "- If the report is mainly a crash, compile failure, internal error, segfault, or runtime exception, choose crash_compile unless the text clearly centers on a numerical value mismatch.",
            "- If NaN/Inf is explicit and central, prefer nan_inf over precision_tolerance.",
            "- If dtype/promotion/casting/mixed precision is central, prefer dtype_casting over precision_tolerance.",
            "- Choose performance_only only when the main issue is runtime, throughput, memory use, or optimization without correctness failure.",
            "- Use needs_review only when the supplied text is genuinely insufficient.",
            "",
        ]
    )
    for row in batch:
        blind_id = row["blind_id"]
        suggestion = suggestions.get(blind_id, {})
        lines.extend(
            [
                f"### TARGET {blind_id}",
                "Weak suggestion:",
                f"primary: {suggestion.get('candidate_primary_failure', '')}",
                f"failure labels: {suggestion.get('candidate_failure_labels', '')}",
                f"cause labels: {suggestion.get('candidate_cause_labels', '')}",
                "",
                "Fold-safe retrieved labeled examples:",
            ]
        )
        for index, (example, score) in enumerate(examples_by_id[blind_id], start=1):
            packet = packet_by_id[example["blind_id"]]
            lines.extend(
                [
                    f"[example {index}] bm25_score={score:.3f}",
                    f"label: {example['gold_primary_failure']}",
                    f"repo: {example['repository']}",
                    f"title: {packet.get('title', '')}",
                    f"evidence: {trim(example.get('gold_evidence_quote', ''), 220)}",
                    f"issue excerpt: {trim(packet.get('issue_body_excerpt', ''), 360)}",
                    "",
                ]
            )
        lines.extend(
            [
                "Target issue:",
                f"blind_id: {blind_id}",
                f"repository: {row.get('repository', '')}",
                f"title: {row.get('title', '')}",
                f"github_labels: {row.get('github_labels', '')}",
                "issue_body_excerpt:",
                trim(row.get("issue_body_excerpt", ""), 1300),
                "comments_excerpt:",
                trim(row.get("comments_excerpt", ""), 900),
                "",
            ]
        )
    return "\n".join(lines)


def build_batches(rows: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fold-safe retrieval-augmented GPU-NFBench prompts against Gemini.")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--per-label-cap", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and write metadata without calling Gemini.")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "llm_rag_predictions_gemini31_flash_lite.csv")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not args.dry_run and not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")

    gold_rows = read_csv(GOLD)
    packet_rows = read_csv(PACKET)
    packet_by_id = {row["blind_id"]: row for row in packet_rows}
    suggestions = {row["blind_id"]: row for row in read_csv(SUGGESTIONS)}
    text_by_id = {row["blind_id"]: issue_text(row) for row in packet_rows}
    gold_by_id = {row["blind_id"]: row for row in gold_rows}
    folds = stratified_folds(gold_rows, args.folds)
    fold_by_id = {blind_id: idx for idx, fold in enumerate(folds) for blind_id in fold}

    test_ids = [row["blind_id"] for row in gold_rows]
    if args.start:
        test_ids = test_ids[args.start :]
    if args.limit:
        test_ids = test_ids[: args.limit]
    test_rows = [packet_by_id[blind_id] for blind_id in test_ids]

    examples_by_id: dict[str, list[tuple[dict[str, str], float]]] = {}
    retrieved_meta: dict[str, tuple[str, str]] = {}
    for blind_id in test_ids:
        fold_idx = fold_by_id[blind_id]
        train_rows = [gold_by_id[bid] for other_idx, fold in enumerate(folds) if other_idx != fold_idx for bid in fold]
        indexer = BM25Index(train_rows, text_by_id)
        examples = indexer.top_examples(text_by_id[blind_id], args.per_label_cap, args.examples)
        examples_by_id[blind_id] = examples
        retrieved_meta[blind_id] = (
            "|".join(example["blind_id"] for example, _ in examples),
            "|".join(example["gold_primary_failure"] for example, _ in examples),
        )

    fieldnames = [
        "blind_id",
        "primary_failure_label",
        "secondary_cause_labels",
        "is_true_numerical_failure",
        "evidence_quote",
        "confidence",
        "model",
        "provider",
        "elapsed_seconds",
        "retrieved_example_ids",
        "retrieved_example_labels",
        "weak_candidate_primary",
        "error",
        "raw_response",
    ]
    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(build_batches(test_rows, args.batch_size), start=1):
        prompt = batch_prompt(batch, examples_by_id, packet_by_id, suggestions)
        started = time.time()
        text = ""
        error = ""
        predictions: dict[str, dict[str, Any]] = {}
        if args.dry_run:
            error = "dry_run_no_api_call"
        else:
            for attempt in range(1, args.retries + 2):
                try:
                    text = post_gemini(str(api_key), args.model, prompt, args.max_tokens, args.timeout)
                    error = ""
                    break
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                    if attempt <= args.retries:
                        time.sleep(retry_delay(error, attempt))
            if not error:
                try:
                    parsed = parse_json(text)
                    if isinstance(parsed, dict):
                        parsed = parsed.get("predictions", [])
                    predictions = {
                        str(row.get("blind_id", "")): row
                        for row in parsed
                        if isinstance(row, dict) and row.get("blind_id")
                    }
                except Exception as exc:  # noqa: BLE001
                    error = f"parse_error: {exc}"
        elapsed = time.time() - started
        for item in batch:
            blind_id = item["blind_id"]
            row = normalize(
                predictions.get(blind_id, {}),
                blind_id,
                args.model,
                elapsed,
                error=error[:1000] if blind_id not in predictions else "",
                raw_response=text,
            )
            retrieved_ids, retrieved_labels = retrieved_meta[blind_id]
            row["retrieved_example_ids"] = retrieved_ids
            row["retrieved_example_labels"] = retrieved_labels
            row["weak_candidate_primary"] = suggestions.get(blind_id, {}).get("candidate_primary_failure", "")
            rows.append(row)
        print(f"batch {batch_index} rows={len(batch)} predictions={len(predictions)} error={error[:160]}", file=sys.stderr)
        if args.sleep and not args.dry_run:
            time.sleep(args.sleep)

    write_csv(args.out, rows, fieldnames)
    print(args.out)


if __name__ == "__main__":
    main()
