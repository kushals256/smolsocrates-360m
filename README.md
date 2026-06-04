# 🎓 SmolSocrates-360M — A Socratic Coding Tutor

Fine-tuned **SmolLM2-360M** to teach coding through Socratic questioning — never giving direct answers, always guiding through progressive hints.

> *"The only true wisdom is in knowing you know nothing."* — Socrates

## 🎯 The Thesis

**Small specialist beats big generalist** — for narrow tasks, a 360M parameter model fine-tuned with LoRA can consistently follow pedagogical patterns that even much larger models struggle with out-of-the-box.

## 📊 Results

<!-- PLACEHOLDER: Update with actual results after training -->

| Metric | Baseline (SmolLM2-360M) | Fine-tuned (SmolSocrates) | Improvement |
|--------|------------------------|--------------------------|-------------|
| Overall Score | 2.2/9 | 5.4/9 | +3.2 |
| No Direct Answer | 1.6/3 | 3.0/3 | +1.4 |
| Asks Questions | 0.4/3 | 1.0/3 | +0.6 |
| Encouragement | 0.2/3 | 1.4/3 | +1.2 |
| No-Code Rate | 40% | 100% | +60% |

### Before (Base Model)
```
Q: My for loop never stops running. Here's my code: ...
A: The issue is with the condition of your while loop. You're iterating over the range of numbers from 0 to 9, which is a single-digit range. This is not what you want, as it's not a valid range for a fo
```

### After (SmolSocrates)
```
Q: My for loop never stops running. Here's my code: ...
A: You're trying to avoid infinite loops by using a while loop with a condition that doesn't meet the requirements. Can you think of what it would be called when you have a condition that never gets reset?
```

## 🚀 Quick Start

### Use the Model
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="kushalicious/SmolSocrates-360M")

response = pipe([
    {"role": "system", "content": "You are a Socratic coding tutor..."},
    {"role": "user", "content": "How do I sort a list in Python?"},
], max_new_tokens=256, return_full_text=False)

print(response[0]["generated_text"])
```

### Interactive Demo
```bash
pip install transformers torch
python src/inference.py --model kushalicious/SmolSocrates-360M
```

## 🏗️ Project Structure

```
├── notebooks/
│   └── socratic_tutor_finetune.ipynb  # Main training notebook (Colab)
├── data/
│   ├── prepare_dataset.py             # Dataset download & processing
│   ├── train_data.jsonl               # Training data (197 examples)
│   └── eval_data.jsonl                # Evaluation data (30 examples)
├── eval/
│   ├── rubric.py                      # Socratic quality scoring rubric
│   └── evaluate.py                    # Full evaluation harness
├── src/
│   └── inference.py                   # Interactive demo
├── blog/
│   └── post.md                        # Blog post draft
├── requirements.txt
└── README.md
```

## 🔧 Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | SmolLM2-360M-Instruct |
| Method | LoRA (r=16, α=32) |
| Trainable Params | ~2% of total |
| Dataset | PACT-Socratic-Coding-Tutor (227 examples) |
| Train/Eval Split | 298 / 30 |
| Epochs | 10 |
| Learning Rate | 5e-4 (cosine) |
| Training Time | ~20 min (Colab T4) |
| Cost | $0 (free tier) |

## 📝 Evaluation

We score responses on 3 rule-based dimensions (0-3 each):

1. **No Direct Answer** — Does the model avoid giving code solutions?
2. **Asks Questions** — Does the response contain guiding questions?
3. **Encouragement** — Is the tone positive and supportive?

Plus 2 LLM-judged dimensions (via Groq API):

4. **Correct Direction** — Do the questions point toward the right fix?
5. **Progressive Hints** — Does the tutor scaffold from broad to specific?

## 🔗 Links

- 🤗 Model: [SmolSocrates-360M](https://huggingface.co/kushalicious/SmolSocrates-360M)
- 📓 Colab: [Training Notebook](link-to-colab)
- 📝 Blog: [Blog Post](link-to-blog)
- 📦 Dataset: [PACT-Socratic-Coding-Tutor](https://huggingface.co/datasets/AndreiSobo/PACT-Socratic-Coding-Tutor)

## 📄 License

MIT

## 🙏 Acknowledgments

- [PACT Dataset](https://huggingface.co/datasets/AndreiSobo/PACT-Socratic-Coding-Tutor) by Andrei Sobo
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) by HuggingFace
- [Unsloth](https://github.com/unslothai/unsloth) for fast LoRA training
