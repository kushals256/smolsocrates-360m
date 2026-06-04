"""
Evaluation harness for SmolSocrates-360M.

Runs the model on eval test cases and scores responses using:
1. Rule-based heuristics (3 dimensions from rubric.py)
2. LLM-as-judge via Groq API (2 dimensions: Correct Direction, Progressive Hints)

Usage:
    python eval/evaluate.py --model HuggingFaceTB/SmolLM2-360M-Instruct --output eval/results/baseline_results.json
    python eval/evaluate.py --model your-username/SmolSocrates-360M --output eval/results/finetuned_results.json
    python eval/evaluate.py --compare eval/results/baseline_results.json eval/results/finetuned_results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.rubric import (
    EvalResult,
    RubricScore,
    score_rule_based,
)

load_dotenv()


# --- LLM-as-Judge (Groq) ---

LLM_JUDGE_PROMPT = """\
You are an expert evaluator of AI coding tutors. You are assessing whether a response \
follows the Socratic method — guiding students through questions rather than giving answers.

## Student's Question:
{prompt}

## Tutor's Response:
{response}

## Evaluate on these 2 criteria (score 0-3 each):

### 1. Correct Direction (0-3)
Does the tutor's question/hint point the student toward the right concept or fix?
- 3: Questions directly address the root issue and would lead to the correct solution
- 2: Questions are relevant but could be more targeted
- 1: Questions are somewhat related but might confuse the student
- 0: Questions are irrelevant or would lead the student astray

### 2. Progressive Hints (0-3)
Does the tutor scaffold learning — starting broad, getting more specific?
- 3: Clear progression from general to specific, breaks problem into steps
- 2: Some structure in the hints, moves toward specificity
- 1: Hints are present but not well-structured
- 0: No scaffolding — either too vague or jumps to the answer

## Response Format
Respond with ONLY a JSON object (no markdown, no explanation):
{{"correct_direction": {{"score": <0-3>, "explanation": "<brief reason>"}}, "progressive_hints": {{"score": <0-3>, "explanation": "<brief reason>"}}}}
"""


def llm_judge_score(prompt: str, response: str) -> list[RubricScore]:
    """
    Use Groq LLM (Llama 3) as a judge to score Correct Direction and Progressive Hints.
    Returns 2 RubricScore objects.
    """
    try:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("  ⚠️  GROQ_API_KEY not set, using default scores for LLM dimensions")
            return _default_llm_scores()

        client = Groq(api_key=api_key)

        judge_prompt = LLM_JUDGE_PROMPT.format(prompt=prompt, response=response)

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=300,
        )

        result_text = completion.choices[0].message.content.strip()

        # Parse JSON response
        # Strip markdown code fences if present
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1]
            result_text = result_text.rsplit("```", 1)[0]

        result = json.loads(result_text)

        return [
            RubricScore(
                "Correct Direction",
                result["correct_direction"]["score"],
                result["correct_direction"]["explanation"],
            ),
            RubricScore(
                "Progressive Hints",
                result["progressive_hints"]["score"],
                result["progressive_hints"]["explanation"],
            ),
        ]

    except Exception as e:
        print(f"  ⚠️  LLM judge error: {e}")
        return _default_llm_scores()


def _default_llm_scores() -> list[RubricScore]:
    """Fallback scores when LLM judge is unavailable."""
    return [
        RubricScore("Correct Direction", 1, "LLM judge unavailable — default score"),
        RubricScore("Progressive Hints", 1, "LLM judge unavailable — default score"),
    ]


# --- Model Inference ---

def generate_response(
    model_name: str,
    messages: list[dict],
    max_new_tokens: int = 512,
) -> str:
    """
    Generate a response from the model.
    Uses HuggingFace transformers pipeline.
    """
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    # Format as chat
    output = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
    )

    return output[0]["generated_text"].strip()


def generate_response_groq_baseline(
    messages: list[dict],
    model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Generate response using Groq API — used for quick baseline testing
    when you don't want to load a local model.
    """
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )

    return completion.choices[0].message.content.strip()


# --- Main Evaluation ---

def evaluate_model(
    model_name: str,
    eval_data_path: str = "data/eval_data.jsonl",
    output_path: str | None = None,
    use_llm_judge: bool = True,
    use_local_model: bool = True,
    max_examples: int | None = None,
) -> list[EvalResult]:
    """
    Run full evaluation pipeline on a model.

    Args:
        model_name: HuggingFace model name or path.
        eval_data_path: Path to evaluation JSONL file.
        output_path: Where to save results JSON.
        use_llm_judge: Whether to use Groq LLM for 2 additional scoring dimensions.
        use_local_model: If True, loads model locally. If False, uses Groq API.
        max_examples: Limit number of examples to evaluate (for testing).
    """
    # Load eval data
    eval_data = []
    with open(eval_data_path) as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    if max_examples:
        eval_data = eval_data[:max_examples]

    print(f"\n🧪 Evaluating: {model_name}")
    print(f"   Test cases: {len(eval_data)}")
    print(f"   LLM judge: {'enabled' if use_llm_judge else 'disabled'}")
    print(f"   Inference: {'local model' if use_local_model else 'Groq API'}")
    print("-" * 60)

    # Load local model once if needed
    pipe = None
    if use_local_model:
        print("   Loading model...")
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        print("   Model loaded ✓")

    results = []

    for i, example in enumerate(eval_data):
        test_id = example.get("id", f"test_{i:03d}")
        messages = example["messages"]

        # Extract the user prompt (last user message)
        user_prompt = ""
        eval_messages = []
        for msg in messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]
                eval_messages.append(msg)
            elif msg["role"] == "system":
                eval_messages.append(msg)
            # Skip assistant messages — we want the model to generate them

        # Only keep system + first user message for eval
        eval_msgs = [m for m in messages if m["role"] == "system"]
        first_user = next((m for m in messages if m["role"] == "user"), None)
        if first_user:
            eval_msgs.append(first_user)
            user_prompt = first_user["content"]

        print(f"\n   [{i+1}/{len(eval_data)}] {test_id}")
        print(f"   Prompt: {user_prompt[:80]}...")

        # Generate response
        try:
            if use_local_model and pipe:
                output = pipe(
                    eval_msgs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_full_text=False,
                )
                response = output[0]["generated_text"].strip()
            else:
                response = generate_response_groq_baseline(eval_msgs)
                time.sleep(0.5)  # Rate limiting for Groq free tier
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
            response = f"[ERROR: {e}]"

        print(f"   Response: {response[:80]}...")

        # Score — rule-based
        scores = score_rule_based(response)

        # Score — LLM judge
        if use_llm_judge and user_prompt and "[ERROR" not in response:
            llm_scores = llm_judge_score(user_prompt, response)
            scores.extend(llm_scores)
            time.sleep(0.3)  # Rate limiting

        result = EvalResult(
            test_id=test_id,
            prompt=user_prompt,
            response=response,
            model_name=model_name,
            scores=scores,
        )
        results.append(result)
        print(f"   Score: {result.total_score}/{result.max_score} ({result.percentage}%)")

    # Summary
    _print_summary(results, model_name)

    # Save results
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "num_examples": len(results),
                    "summary": _compute_summary(results),
                    "results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\n💾 Results saved to: {output_path}")

    return results


def _print_summary(results: list[EvalResult], model_name: str):
    """Print a summary table of evaluation results."""
    if not results:
        return

    summary = _compute_summary(results)

    print("\n" + "=" * 60)
    print(f"📊 EVALUATION SUMMARY: {model_name}")
    print("=" * 60)

    print(f"\n   Overall Score: {summary['avg_total']:.1f}/{summary['max_score']} "
          f"({summary['avg_percentage']:.1f}%)")

    print(f"\n   Per-Dimension Averages:")
    for dim_name, avg_score in summary["per_dimension"].items():
        bar = "█" * int(avg_score) + "░" * (3 - int(avg_score))
        print(f"      {dim_name:25s}  {avg_score:.1f}/3  {bar}")

    print(f"\n   No-Code Rate: {summary['no_code_rate']:.0f}% of responses contain zero code blocks")
    print("=" * 60)


def _compute_summary(results: list[EvalResult]) -> dict:
    """Compute aggregate statistics from results."""
    if not results:
        return {}

    total_scores = [r.total_score for r in results]
    max_score = results[0].max_score if results else 15

    # Per-dimension averages
    dim_scores: dict[str, list[int]] = {}
    for r in results:
        for s in r.scores:
            dim_scores.setdefault(s.name, []).append(s.score)

    per_dim = {name: sum(scores) / len(scores) for name, scores in dim_scores.items()}

    # No-code rate (% of responses with 0 code blocks)
    import re
    no_code_count = sum(
        1 for r in results
        if len(re.findall(r"```[\s\S]*?```", r.response)) == 0
    )

    return {
        "avg_total": sum(total_scores) / len(total_scores),
        "max_score": max_score,
        "avg_percentage": sum(r.percentage for r in results) / len(results),
        "per_dimension": per_dim,
        "no_code_rate": (no_code_count / len(results)) * 100,
        "min_score": min(total_scores),
        "max_score_achieved": max(total_scores),
    }


# --- Comparison ---

def compare_results(baseline_path: str, finetuned_path: str):
    """Print a side-by-side comparison of baseline vs fine-tuned results."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(finetuned_path) as f:
        finetuned = json.load(f)

    b_summary = baseline["summary"]
    f_summary = finetuned["summary"]

    print("\n" + "=" * 70)
    print("📊 BEFORE vs AFTER COMPARISON")
    print("=" * 70)
    print(f"\n   {'Metric':<30s} {'Baseline':>10s} {'Fine-tuned':>12s} {'Delta':>8s}")
    print(f"   {'-'*30} {'-'*10} {'-'*12} {'-'*8}")

    # Overall
    b_avg = b_summary["avg_total"]
    f_avg = f_summary["avg_total"]
    delta = f_avg - b_avg
    sign = "+" if delta > 0 else ""
    print(f"   {'Overall Score':<30s} {b_avg:>10.1f} {f_avg:>12.1f} {sign}{delta:>7.1f}")

    # Per dimension
    all_dims = set(b_summary.get("per_dimension", {}).keys()) | \
               set(f_summary.get("per_dimension", {}).keys())

    for dim in sorted(all_dims):
        b_val = b_summary.get("per_dimension", {}).get(dim, 0)
        f_val = f_summary.get("per_dimension", {}).get(dim, 0)
        delta = f_val - b_val
        sign = "+" if delta > 0 else ""
        print(f"   {dim:<30s} {b_val:>10.1f} {f_val:>12.1f} {sign}{delta:>7.1f}")

    # No-code rate
    b_ncr = b_summary.get("no_code_rate", 0)
    f_ncr = f_summary.get("no_code_rate", 0)
    delta = f_ncr - b_ncr
    sign = "+" if delta > 0 else ""
    print(f"   {'No-Code Rate (%)':<30s} {b_ncr:>10.0f} {f_ncr:>12.0f} {sign}{delta:>7.0f}")

    print("\n" + "=" * 70)

    # Sample comparisons
    print("\n📝 SAMPLE COMPARISON (first 3 examples):")
    for i in range(min(3, len(baseline["results"]), len(finetuned["results"]))):
        b_res = baseline["results"][i]
        f_res = finetuned["results"][i]
        print(f"\n   --- Test {b_res['test_id']} ---")
        print(f"   Prompt: {b_res['prompt'][:100]}...")
        print(f"\n   BASELINE ({b_res['total_score']}/{b_res['max_score']}):")
        print(f"   {b_res['response'][:200]}...")
        print(f"\n   FINE-TUNED ({f_res['total_score']}/{f_res['max_score']}):")
        print(f"   {f_res['response'][:200]}...")
    print()


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolSocrates Socratic tutoring quality")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation on a model")
    eval_parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    eval_parser.add_argument("--eval-data", default="data/eval_data.jsonl", help="Path to eval data")
    eval_parser.add_argument("--output", required=True, help="Path to save results JSON")
    eval_parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM-as-judge scoring")
    eval_parser.add_argument("--groq", action="store_true", help="Use Groq API instead of local model")
    eval_parser.add_argument("--max-examples", type=int, default=None, help="Limit eval examples")

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare baseline vs fine-tuned results")
    cmp_parser.add_argument("baseline", help="Path to baseline results JSON")
    cmp_parser.add_argument("finetuned", help="Path to fine-tuned results JSON")

    args = parser.parse_args()

    if args.command == "eval":
        evaluate_model(
            model_name=args.model,
            eval_data_path=args.eval_data,
            output_path=args.output,
            use_llm_judge=not args.no_llm_judge,
            use_local_model=not args.groq,
            max_examples=args.max_examples,
        )
    elif args.command == "compare":
        compare_results(args.baseline, args.finetuned)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
