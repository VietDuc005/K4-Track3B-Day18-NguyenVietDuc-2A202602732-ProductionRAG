from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _safe_float(val: object, default: float = 0.0) -> float:
    """Safely convert a value to float, handling None and NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        per_question = []
        for _, row in df.iterrows():
            raw_contexts = row.get("contexts", [])
            if hasattr(raw_contexts, "tolist"):
                ctx_list = raw_contexts.tolist()
            elif isinstance(raw_contexts, list):
                ctx_list = list(raw_contexts)
            else:
                ctx_list = [str(raw_contexts)]

            per_question.append(
                EvalResult(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    contexts=ctx_list,
                    ground_truth=str(row.get("ground_truth", "")),
                    faithfulness=_safe_float(row.get("faithfulness")),
                    answer_relevancy=_safe_float(row.get("answer_relevancy")),
                    context_precision=_safe_float(row.get("context_precision")),
                    context_recall=_safe_float(row.get("context_recall")),
                )
            )

        return {
            "faithfulness": _safe_float(result.get("faithfulness")),
            "answer_relevancy": _safe_float(result.get("answer_relevancy")),
            "context_precision": _safe_float(result.get("context_precision")),
            "context_recall": _safe_float(result.get("context_recall")),
            "per_question": per_question,
        }
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  RAGAS evaluation failed: {e}", flush=True)
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating",
            "Tighten prompt, lower temperature",
        ),
        "context_recall": (
            "Missing relevant chunks",
            "Improve chunking or add BM25",
        ),
        "context_precision": (
            "Too many irrelevant chunks",
            "Add reranking or metadata filter",
        ),
        "answer_relevancy": (
            "Answer doesn't match question",
            "Improve prompt template",
        ),
    }

    analyzed = []
    for r in eval_results:
        metric_scores = {
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "answer_relevancy": r.answer_relevancy,
        }
        avg_score = sum(metric_scores.values()) / len(metric_scores)
        worst_metric = min(metric_scores, key=metric_scores.get)
        worst_score = metric_scores[worst_metric]
        diagnosis, fix = diagnostic_tree.get(
            worst_metric, ("Unknown failure", "Investigate logs")
        )

        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": float(worst_score),
            "avg_score": float(avg_score),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    # Sort ascending by average score to surface the worst performing queries
    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
