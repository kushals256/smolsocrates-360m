"""
Simple inference script for SmolSocrates-360M.

Usage:
    python src/inference.py --model your-username/SmolSocrates-360M
    python src/inference.py --model your-username/SmolSocrates-360M --prompt "How do I sort a list?"
"""

import argparse


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


def load_model(model_name: str):
    """Load model and tokenizer from HuggingFace."""
    from transformers import pipeline

    print(f"Loading model: {model_name}")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    print("Model loaded ✓\n")
    return pipe


def generate(pipe, user_message: str, max_new_tokens: int = 512) -> str:
    """Generate a Socratic tutoring response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    output = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
    )

    return output[0]["generated_text"].strip()


def interactive_mode(pipe):
    """Run an interactive chat session."""
    print("=" * 60)
    print("🎓 SmolSocrates — Socratic Coding Tutor")
    print("=" * 60)
    print("Ask me a coding question and I'll guide you to the answer!")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("👩‍💻 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Keep coding! 🚀")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Keep coding! 🚀")
            break

        response = generate(pipe, user_input)
        print(f"\n🎓 SmolSocrates: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="SmolSocrates Socratic Coding Tutor")
    parser.add_argument(
        "--model",
        default="SmolSocrates-360M",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt (non-interactive mode)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate",
    )
    args = parser.parse_args()

    pipe = load_model(args.model)

    if args.prompt:
        # Single-shot mode
        response = generate(pipe, args.prompt, args.max_tokens)
        print(f"🎓 SmolSocrates: {response}")
    else:
        # Interactive mode
        interactive_mode(pipe)


if __name__ == "__main__":
    main()
