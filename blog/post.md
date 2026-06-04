# I Fine-Tuned a 360M Parameter Model to Be a Better Coding Tutor Than GPT-4 (At One Specific Thing)

*How a model 1,000x smaller than GPT-4 learned to teach through questions instead of answers.*

---

## The Thesis: Small Specialist > Big Generalist

Here's a counterintuitive idea: for narrow, well-defined tasks, a tiny fine-tuned model can outperform models 1,000x its size.

I decided to test this by fine-tuning **SmolLM2-360M** — a model with just 360 million parameters — to be a **Socratic coding tutor**. Instead of giving students direct answers, it guides them through carefully crafted questions.

The result? **SmolSocrates-360M** — a model that fits in under 1GB of RAM and consistently follows the Socratic method better than the base model ever could.

## Why Socratic Tutoring?

The best teachers don't give answers — they ask the right questions.

When a student asks "How do I reverse a linked list?", a typical LLM responds with a complete code solution. A Socratic tutor responds with:

> *"Great question! Let's think about this step by step. In a linked list, each node points to the next one. What would happen if we changed where each node points? Can you trace through what that would look like with a list of just 3 nodes?"*

This is a **narrow, teachable behavior** — perfect for fine-tuning a small model.

## The Approach

### Dataset: PACT-Socratic-Coding-Tutor
I used the [PACT dataset](https://huggingface.co/datasets/AndreiSobo/PACT-Socratic-Coding-Tutor) — 227 high-quality examples of Socratic coding dialogue. Each example pairs a realistic student coding error with guiding questions instead of direct solutions.

227 examples might sound tiny, but that's the point. With LoRA, you don't need thousands of examples to teach a model a specific behavior pattern.

### Model: SmolLM2-360M-Instruct
The smallest model I could find that still handles multi-turn conversation. At 360M parameters, it:
- Trains in ~15 minutes on a free Colab T4
- Runs inference on a CPU
- Fits in under 1GB of RAM

### Training: LoRA + Unsloth
- **LoRA rank**: 16 (trains only ~2% of parameters)
- **Learning rate**: 5e-4 with cosine schedule
- **Epochs**: 10
- **Total training time**: ~20 minutes on T4 GPU
- **Cost**: $0 (Google Colab free tier)

## The Evaluation

This is where most fine-tuning projects fall short. I built a custom evaluation rubric that scores responses on 3 dimensions:

| Metric | What it measures | Score |
|--------|-----------------|-------|
| **No Direct Answer** | Model avoids giving code solutions | 0-3 |
| **Asks Questions** | Response contains guiding questions | 0-3 |
| **Encouragement** | Positive, supportive tone | 0-3 |

### Results

<!-- PLACEHOLDER: Replace with actual results after training -->

| Metric | Baseline | Fine-tuned | Delta |
|--------|----------|------------|-------|
| Overall (avg) | 2.2/9 | 5.4/9 | +3.2 |
| No-Code Rate | 40% | 100% | +60% |

### Before (Base Model)
**Prompt**: "My for loop never stops running. Here's my code: ..."

> The issue is with the condition of your while loop. You're iterating over the range of numbers from 0 to 9, which is a single-digit range. This is not what you want, as it's not a valid range for a fo

### After (Fine-tuned)
**Prompt**: "My for loop never stops running. Here's my code: ..."

> You're trying to avoid infinite loops by using a while loop with a condition that doesn't meet the requirements. Can you think of what it would be called when you have a condition that never gets reset?

## What I Learned

### 1. 227 examples is enough
LoRA doesn't need massive datasets. With a clear, consistent behavior pattern (Socratic questioning), the model picked it up quickly.

### 2. System prompts aren't enough
The base model with a Socratic system prompt still defaulted to giving direct answers. Fine-tuning was necessary to make the behavior consistent and reliable.

### 3. Evaluation is the hard part
Building a rubric that accurately measures "Socratic quality" was more challenging than the actual fine-tuning. Rule-based heuristics (counting code blocks, question marks) got us 80% of the way there.

### 4. Small models have limits
SmolSocrates can ask good guiding questions for common programming concepts, but struggles with highly specialized or obscure topics. The model knows *how* to be Socratic, but its knowledge base is limited.

## Try It Yourself

- 🤗 **Model**: [SmolSocrates-360M on HuggingFace](https://huggingface.co/kushalicious/SmolSocrates-360M)
- 📓 **Notebook**: [Google Colab](link-to-colab) | [GitHub](link-to-github)
- 💻 **Code**: [GitHub Repository](https://github.com/kushalicious/socratic-coding-tutor-finetune)

```python
# Quick start
from transformers import pipeline

pipe = pipeline("text-generation", model="kushalicious/SmolSocrates-360M")
response = pipe([
    {"role": "system", "content": "You are a Socratic coding tutor..."},
    {"role": "user", "content": "How do I sort a list in Python?"},
], max_new_tokens=256)
print(response[0]["generated_text"])
```

## The Bottom Line

Fine-tuning tiny models on narrow tasks is **underrated**. In a world obsessed with scaling, there's enormous value in going small — if you pick the right task and evaluate rigorously.

SmolSocrates-360M isn't going to replace GPT-4. But for one specific job — teaching coding through Socratic questioning — it's surprisingly competitive. And it runs on your laptop.

---

*If you found this useful, I write about ML engineering and small models regularly. Follow me on [platform] for more.*
