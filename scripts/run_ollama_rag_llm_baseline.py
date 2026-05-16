from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


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
    "nan_inf": "The core reported problem is NaN or Inf values produced by a GPU/kernel/framework operation.",
    "overflow_underflow": "The core reported problem is overflow, underflow, saturation, wraparound, or range-limit behavior.",
    "precision_tolerance": "The core reported problem is numerically wrong or different values, tolerance mismatch, nondeterministic precision, or approximate accuracy loss without explicit NaN/Inf or overflow as the main symptom.",
    "dtype_casting": "The core reported problem is dtype conversion, unsupported dtype, promotion/casting, mixed precision type handling, or type-specific compile/runtime behavior.",
    "crash_compile": "The report is mainly a crash, compiler failure, internal error, segfault, or build/runtime exception rather than a numerical-output mismatch.",
    "performance_only": "The report is mainly latency, throughput, memory use, benchmark speed, or performance regression rather than numerical correctness.",
    "not_numerical_failure": "The issue is a feature request, installation/environment question, documentation/API question, or unrelated report where numerical words are incidental.",
    "needs_review": "The evidence is too ambiguous to classify confidently from the supplied text.",
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
    return [token.lower() for token in TOKEN_RE.findall(text)]


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
        items = sorted(items, key=lambda r: r["blind_id"])
        for index, item in enumerate(items):
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
        score = 0.0
        doc = self.docs[index]
        doc_len = self.doc_lengths[index]
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


def system_prompt() -> str:
    return (
        "You are classifying GPU/kernel issue reports for a benchmark. "
        "Use taxonomy definitions, retrieved training examples, and the target issue text. "
        "The weak suggestion is noisy and may be wrong. Return JSON only."
    )


def build_prompt(
    row: dict[str, str],
    suggestion: dict[str, str],
    examples: list[tuple[dict[str, str], float]],
    packet_by_id: dict[str, dict[str, str]],
) -> str:
    lines = [
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
            "Allowed secondary labels:",
            ", ".join(SECONDARY_LABELS),
            "",
            "Decision rules:",
            "- Label the main failure symptom, not every term mentioned.",
            "- If an issue is primarily an API request, feature request, install problem, or documentation question, choose not_numerical_failure even if it mentions dtype, nan, precision, or overflow.",
            "- If the report is mainly a crash or compiler/runtime exception, choose crash_compile unless the text clearly centers on a numerical value mismatch.",
            "- If NaN/Inf is explicit and central, prefer nan_inf over generic precision_tolerance.",
            "- If dtype/promotion/casting is central, prefer dtype_casting over precision_tolerance.",
            "- Use needs_review only when the supplied text is genuinely insufficient.",
            "",
            "Noisy weak pre-classifier suggestion for target issue:",
            f"primary: {suggestion.get('candidate_primary_failure', '')}",
            f"candidate labels: {suggestion.get('candidate_failure_labels', '')}",
            f"candidate causes: {suggestion.get('candidate_cause_labels', '')}",
            "",
            "Retrieved labeled training examples. These examples come only from other cross-validation folds:",
        ]
    )
    for index, (example, score) in enumerate(examples, start=1):
        packet = packet_by_id[example["blind_id"]]
        lines.extend(
            [
                f"[example {index}] bm25_score={score:.3f}",
                f"label: {example['gold_primary_failure']}",
                f"repo: {example['repository']}",
                f"title: {packet.get('title', '')}",
                f"evidence: {trim(example.get('gold_evidence_quote', ''), 260)}",
                f"issue excerpt: {trim(packet.get('issue_body_excerpt', ''), 450)}",
                "",
            ]
        )
    lines.extend(
        [
            "Target issue to classify:",
            f"blind_id: {row['blind_id']}",
            f"repository: {row.get('repository', '')}",
            f"url: {row.get('url', '')}",
            f"title: {row.get('title', '')}",
            f"github_labels: {row.get('github_labels', '')}",
            "",
            "issue_body_excerpt:",
            trim(row.get("issue_body_excerpt", ""), 1600),
            "",
            "comments_excerpt:",
            trim(row.get("comments_excerpt", ""), 1200),
            "",
            "Return exactly this JSON shape:",
            '{"primary_failure_label":"...","secondary_cause_labels":["..."],"is_true_numerical_failure":"yes|no|unclear","evidence_quote":"short quote from target text","confidence":"high|medium|low"}',
        ]
    )
    return "\n".join(lines)


def api_chat(model: str, system: str, user: str, host: str, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 220,
        },
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("message", {}).get("content", ""))


def extract_json(text: str) -> dict[str, object]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def normalize_prediction(parsed: dict[str, object]) -> dict[str, str]:
    primary = str(parsed.get("primary_failure_label", "needs_review")).strip()
    if primary not in PRIMARY_LABELS:
        primary = "needs_review"
    raw_secondary = parsed.get("secondary_cause_labels", ["unknown"])
    if isinstance(raw_secondary, str):
        secondary_values = [part.strip() for part in raw_secondary.replace(",", "|").split("|")]
    elif isinstance(raw_secondary, list):
        secondary_values = [str(part).strip() for part in raw_secondary]
    else:
        secondary_values = []
    secondary = [value for value in secondary_values if value in SECONDARY_LABELS] or ["unknown"]
    true_failure = str(parsed.get("is_true_numerical_failure", "unclear")).strip().lower()
    if true_failure not in {"yes", "no", "unclear"}:
        true_failure = "unclear"
    confidence = str(parsed.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "primary_failure_label": primary,
        "secondary_cause_labels_pipe_separated": "|".join(dict.fromkeys(secondary)),
        "is_true_numerical_failure": true_failure,
        "confidence": confidence,
        "evidence_quote": str(parsed.get("evidence_quote", ""))[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fold-safe retrieval/few-shot Ollama LLM baseline.")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--per-label-cap", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    gold_rows = read_csv(GOLD)
    packet_rows = read_csv(PACKET)
    suggestions = {row["blind_id"]: row for row in read_csv(SUGGESTIONS)}
    packet_by_id = {row["blind_id"]: row for row in packet_rows}
    text_by_id = {row["blind_id"]: issue_text(row) for row in packet_rows}
    gold_by_id = {row["blind_id"]: row for row in gold_rows}
    folds = stratified_folds(gold_rows, args.folds)
    fold_by_id = {bid: idx for idx, fold in enumerate(folds) for bid in fold}

    test_ids = [row["blind_id"] for row in gold_rows]
    if args.limit:
        test_ids = test_ids[: args.limit]

    out_path = Path(args.out) if args.out else OUT_DIR / f"llm_rag_predictions_ollama_{args.model.replace(':', '_')}.csv"
    rows: list[dict[str, object]] = []

    for index, blind_id in enumerate(test_ids, start=1):
        row = packet_by_id[blind_id]
        fold_idx = fold_by_id[blind_id]
        train_rows = [gold_by_id[bid] for other_idx, fold in enumerate(folds) if other_idx != fold_idx for bid in fold]
        indexer = BM25Index(train_rows, text_by_id)
        examples = indexer.top_examples(text_by_id[blind_id], args.per_label_cap, args.examples)
        suggestion = suggestions.get(blind_id, {})
        raw_response = ""
        error = ""
        started = time.time()
        try:
            raw_response = api_chat(args.model, system_prompt(), build_prompt(row, suggestion, examples, packet_by_id), args.host, args.timeout)
            normalized = normalize_prediction(extract_json(raw_response))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            error = str(exc)
            normalized = {
                "primary_failure_label": "needs_review",
                "secondary_cause_labels_pipe_separated": "unknown",
                "is_true_numerical_failure": "unclear",
                "confidence": "low",
                "evidence_quote": "",
            }
        elapsed = time.time() - started
        rows.append(
            {
                "blind_id": blind_id,
                "model": args.model,
                "prompt_file": "fold_safe_rag_generated",
                "max_chars": "",
                "temperature": 0,
                "elapsed_seconds": f"{elapsed:.2f}",
                "retrieved_example_ids": "|".join(example["blind_id"] for example, _ in examples),
                "retrieved_example_labels": "|".join(example["gold_primary_failure"] for example, _ in examples),
                "weak_candidate_primary": suggestion.get("candidate_primary_failure", ""),
                **normalized,
                "raw_response": raw_response[:2000],
                "error": error,
            }
        )
        print(f"{index}/{len(test_ids)} {blind_id} {normalized['primary_failure_label']} {elapsed:.1f}s", flush=True)

    write_csv(
        out_path,
        rows,
        [
            "blind_id",
            "model",
            "prompt_file",
            "max_chars",
            "temperature",
            "elapsed_seconds",
            "retrieved_example_ids",
            "retrieved_example_labels",
            "weak_candidate_primary",
            "primary_failure_label",
            "secondary_cause_labels_pipe_separated",
            "is_true_numerical_failure",
            "confidence",
            "evidence_quote",
            "raw_response",
            "error",
        ],
    )
    print(out_path)


if __name__ == "__main__":
    main()
