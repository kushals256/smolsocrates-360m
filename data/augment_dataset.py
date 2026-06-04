"""
Generate synthetic Socratic coding tutoring examples using Groq API.

Creates additional training examples where a student asks a coding question
and the tutor responds with Socratic guidance (questions + hints, NO code).

Usage:
    .venv/bin/python data/augment_dataset.py
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

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

# Diverse coding scenarios to generate examples for
SCENARIOS = [
    # --- Debugging scenarios ---
    "Student has an off-by-one error in a for loop iterating over a list",
    "Student's recursive function has no base case and causes infinite recursion",
    "Student is getting a KeyError when accessing a dictionary",
    "Student's function returns None instead of the expected value (missing return statement)",
    "Student has a variable scope issue — using a local variable outside its function",
    "Student's list comprehension produces wrong results due to incorrect conditional",
    "Student gets TypeError: unsupported operand when adding a string and integer",
    "Student's while loop modifies a list while iterating over it causing skipped elements",
    "Student's class __init__ method doesn't assign self attributes correctly",
    "Student has an indentation error inside a nested if/else block",
    "Student's binary search implementation doesn't handle edge cases",
    "Student's sorting algorithm swaps elements incorrectly",
    "Student gets an AttributeError because they called a method on the wrong type",
    "Student's file reading code doesn't handle the file not existing",
    "Student's string comparison fails because of trailing whitespace",

    # --- Concept explanation scenarios ---
    "Student asks: What is the difference between a shallow copy and deep copy?",
    "Student asks: How do hash tables work internally?",
    "Student asks: What is Big O notation and why does it matter?",
    "Student asks: What are decorators in Python and when should I use them?",
    "Student asks: How does garbage collection work?",
    "Student asks: What is the difference between concurrency and parallelism?",
    "Student asks: Why would I use a linked list instead of an array?",
    "Student asks: What is memoization and how does it help?",
    "Student asks: How do generators work and when should I use them?",
    "Student asks: What is polymorphism in object-oriented programming?",
    "Student asks: What is the difference between stack and heap memory?",
    "Student asks: How does a binary tree differ from a binary search tree?",
    "Student asks: What are lambda functions and when are they useful?",
    "Student asks: What is the difference between a process and a thread?",
    "Student asks: How does recursion differ from iteration?",

    # --- Code improvement scenarios ---
    "Student has working code that uses nested for loops but could use a dictionary for O(1) lookup",
    "Student wrote a function that's 50 lines long and should be broken into smaller functions",
    "Student is using global variables everywhere instead of passing parameters",
    "Student has repetitive if/elif chains that could be replaced with a dictionary mapping",
    "Student is concatenating strings in a loop instead of using join()",
    "Student has magic numbers throughout their code with no named constants",
    "Student's code works but has no error handling for edge cases",
    "Student is reading an entire large file into memory instead of processing line by line",
    "Student has deeply nested code (4+ levels of indentation) that could be flattened",
    "Student's class has methods that don't use self — they should be standalone functions",
    "Student is using try/except with bare except clause catching all exceptions",
    "Student has duplicate code in multiple functions that could be extracted",
    "Student's variable names are single letters (x, y, n) making code unreadable",
    "Student is using a list where a set would be more efficient for membership checks",
    "Student's code passes but uses O(n²) approach where O(n) is straightforward",

    # --- Additional diverse scenarios ---
    "Student is confused about why their API request returns a 403 error",
    "Student doesn't understand why their regex pattern isn't matching",
    "Student asks how to handle multiple exceptions in Python",
    "Student's SQL query returns duplicate rows and they don't know why",
    "Student is confused about mutable default arguments in Python functions",
    "Student asks how to properly structure a Python project with multiple files",
    "Student doesn't understand the difference between == and is in Python",
    "Student's async code is running synchronously and they don't know why",
    "Student asks how to choose between a list, tuple, set, and dictionary",
    "Student is struggling with understanding closures and variable capture",
]

GENERATION_PROMPT = """\
Create a realistic coding tutoring conversation between a student and a Socratic tutor.

SCENARIO: {scenario}

Write exactly 2 messages:
1. STUDENT: A realistic question/problem description a beginner-intermediate student would write. Include buggy code if it's a debugging scenario (use Python). Make it feel natural, not overly formal.
2. TUTOR: A Socratic response that follows these STRICT rules:
   - NEVER include any code blocks (no ```) or complete solutions
   - Ask 2-4 guiding questions that lead the student toward the answer
   - Start with something encouraging or acknowledging their effort
   - Progress from broad conceptual questions to more specific hints
   - Keep it under 150 words

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{"student": "the student's message here", "tutor": "the tutor's response here"}}
"""


def generate_examples(
    num_examples: int = 120,
    output_path: str = "data/synthetic_data.jsonl",
    model: str = "llama-3.1-8b-instant",
):
    """Generate synthetic Socratic tutoring examples using Groq API."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print(f"🤖 Generating {num_examples} synthetic examples using {model}...")
    print(f"   Using {len(SCENARIOS)} unique scenarios")

    examples = []
    errors = 0

    # Cycle through scenarios, repeating with variation
    scenario_pool = SCENARIOS.copy()
    random.seed(42)
    random.shuffle(scenario_pool)

    for i in range(num_examples):
        scenario = scenario_pool[i % len(scenario_pool)]

        # Add variation for repeated scenarios
        variation = ""
        if i >= len(scenario_pool):
            variations = [
                " The student is a complete beginner.",
                " The student has some experience but is learning a new concept.",
                " The student is frustrated and needs extra encouragement.",
                " The student has tried several approaches already.",
            ]
            variation = random.choice(variations)

        try:
            prompt = GENERATION_PROMPT.format(scenario=scenario + variation)

            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # Higher temp for diversity
                max_tokens=500,
            )

            result_text = completion.choices[0].message.content.strip()

            # Strip markdown fences if present
            if result_text.startswith("```"):
                result_text = result_text.split("\n", 1)[1]
                result_text = result_text.rsplit("```", 1)[0].strip()

            parsed = json.loads(result_text)

            # Validate: tutor response must NOT contain code blocks
            if "```" in parsed["tutor"]:
                print(f"   ⚠️  [{i+1}] Skipping — tutor response contains code block")
                errors += 1
                continue

            # Validate: tutor response MUST contain at least one question
            if "?" not in parsed["tutor"]:
                print(f"   ⚠️  [{i+1}] Skipping — tutor response has no questions")
                errors += 1
                continue

            example = {
                "id": f"synth_{i:04d}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": parsed["student"]},
                    {"role": "assistant", "content": parsed["tutor"]},
                ],
            }
            examples.append(example)

            if (i + 1) % 10 == 0:
                print(f"   ✅ Generated {i+1}/{num_examples} ({errors} skipped)")

            # Rate limiting — Groq free tier is generous but let's be safe
            time.sleep(1.0)

        except json.JSONDecodeError as e:
            print(f"   ⚠️  [{i+1}] JSON parse error: {e}")
            errors += 1
            time.sleep(1.0)
        except Exception as e:
            print(f"   ❌ [{i+1}] Error: {e}")
            errors += 1
            time.sleep(2.0)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ Generated {len(examples)} examples ({errors} skipped)")
    print(f"   Saved to: {output_path}")

    return examples


def combine_datasets(
    original_path: str = "data/train_data.jsonl",
    synthetic_path: str = "data/synthetic_data.jsonl",
    output_path: str = "data/train_data_augmented.jsonl",
):
    """Combine original + synthetic training data."""
    original = []
    with open(original_path) as f:
        for line in f:
            if line.strip():
                original.append(json.loads(line))

    synthetic = []
    with open(synthetic_path) as f:
        for line in f:
            if line.strip():
                synthetic.append(json.loads(line))

    combined = original + synthetic
    random.seed(42)
    random.shuffle(combined)

    with open(output_path, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n📊 Combined dataset:")
    print(f"   Original:  {len(original)} examples")
    print(f"   Synthetic: {len(synthetic)} examples")
    print(f"   Combined:  {len(combined)} examples")
    print(f"   Saved to:  {output_path}")

    return combined


if __name__ == "__main__":
    # Step 1: Generate synthetic examples
    generate_examples(num_examples=120)

    # Step 2: Combine with original training data
    combine_datasets()
