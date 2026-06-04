"""
Dataset preparation for SmolSocrates-360M fine-tuning.

Downloads the PACT-Socratic-Coding-Tutor dataset from HuggingFace,
splits into train/eval, and saves in JSONL format ready for training.
"""

import json
import os
import random
from pathlib import Path

from datasets import load_dataset


# Socratic tutor system prompt — the core personality
SYSTEM_PROMPT = (
    "You are a Socratic coding tutor. Your role is to help students learn programming "
    "by asking guiding questions — never by giving direct answers or complete solutions.\n\n"
    "Rules:\n"
    "1. NEVER provide complete code solutions or direct fixes.\n"
    "2. Ask targeted questions that lead the student to discover the answer themselves.\n"
    "3. Break complex problems into smaller, manageable steps.\n"
    "4. If a student is stuck, offer a progressive hint — start broad, get specific.\n"
    "5. Celebrate progress and encourage self-discovery.\n"
    "6. Match the student's technical level in your language.\n"
    "7. When debugging, ask the student to trace through their code mentally."
)


def download_and_prepare(
    output_dir: str = "data",
    eval_size: int = 30,
    seed: int = 42,
):
    """
    Download PACT dataset, standardize system prompts, and split into train/eval.

    Args:
        output_dir: Directory to save processed data files.
        eval_size: Number of examples to hold out for evaluation.
        seed: Random seed for reproducible splitting.
    """
    print("📥 Downloading PACT-Socratic-Coding-Tutor dataset...")
    ds = load_dataset("AndreiSobo/PACT-Socratic-Coding-Tutor", split="train")
    print(f"   Loaded {len(ds)} examples")

    # Process all examples
    all_examples = []
    for i, example in enumerate(ds):
        messages = example["messages"]

        # Standardize: ensure consistent system prompt
        processed_messages = []
        has_system = False

        for msg in messages:
            if msg["role"] == "system":
                # Replace with our standardized system prompt
                processed_messages.append(
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    }
                )
                has_system = True
            else:
                processed_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        # Add system prompt if missing
        if not has_system:
            processed_messages.insert(
                0,
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
            )

        all_examples.append(
            {
                "id": f"pact_{i:04d}",
                "messages": processed_messages,
            }
        )

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_examples)

    eval_examples = all_examples[:eval_size]
    train_examples = all_examples[eval_size:]

    print(f"   Train: {len(train_examples)} examples")
    print(f"   Eval:  {len(eval_examples)} examples")

    # Save to JSONL
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train_data.jsonl"
    eval_path = out / "eval_data.jsonl"

    _save_jsonl(train_examples, train_path)
    _save_jsonl(eval_examples, eval_path)

    print(f"\n✅ Saved to:")
    print(f"   {train_path}")
    print(f"   {eval_path}")

    # Print a sample
    print("\n📝 Sample training example:")
    sample = train_examples[0]
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:120] + ("..." if len(msg["content"]) > 120 else "")
        print(f"   [{role}] {content}")

    return train_examples, eval_examples


def _save_jsonl(examples: list, path: Path):
    """Save a list of dicts as JSONL."""
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_eval_data(path: str = "data/eval_data.jsonl") -> list:
    """Load evaluation data from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


if __name__ == "__main__":
    download_and_prepare()
