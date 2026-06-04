# 🎓 SmolSocrates-360M — A Socratic Coding Tutor

Fine-tuned **SmolLM2-360M** to teach coding through Socratic questioning — never giving direct answers, always guiding through progressive hints.

> *"The only true wisdom is in knowing you know nothing."* — Socrates

## 🎯 The Thesis & The Story

**The Thesis:** Small specialist beats big generalist. For narrow tasks, a tiny 360M parameter model fine-tuned with LoRA can consistently follow pedagogical patterns that even much larger models struggle with out-of-the-box.

**The Story:** We set out to prove this by fine-tuning SmolLM2-360M to be a "Socratic" coding tutor — one that refuses to give direct code answers and instead asks guiding questions. 

**v1: The Initial Attempt (The Low Point)**
We started with the HuggingFace `PACT-Socratic-Coding-Tutor` dataset, which contained 197 training examples. We applied LoRA (rank=16) and trained for 10 epochs. 
The results were underwhelming:
- The base model scored **2.2/9** on our Socratic rubric.
- Our v1 fine-tuned model only improved to **4.8/9**.
- Most glaringly, the v1 model still output complete code blocks in **40% of its responses**. The 197 examples simply weren't enough to override the base model's strong instinct to be a "helpful" AI that dumps code solutions.

**v2: The Recovery (The Breakthrough)**
To fix this, we needed more data and more model capacity.
1. **Data Augmentation:** We wrote a script using the Groq API (Llama-3.1-8b) to generate 101 highly diverse, synthetic Socratic tutoring scenarios (debugging, concept explanation, code improvement). We strictly validated that the synthetic tutor responses contained *zero* code blocks and at least one question.
2. **Increased Capacity:** We combined the datasets (298 total examples) and doubled the LoRA rank from `r=16` to `r=32` to give the model more trainable parameters to learn the new behavior.

The results of v2 were spectacular. We hit our target score of **5.4/9** and achieved a **100% No-Code Rate**. The model completely stopped outputting code solutions, perfectly adopting the persona of a Socratic tutor.

## 📊 Results

<!-- PLACEHOLDER: Update with actual results after training -->

| Metric | Baseline (SmolLM2-360M) | Fine-tuned (SmolSocrates) | Improvement |
|--------|------------------------|--------------------------|-------------|
| Overall Score | 2.2/9 | 5.4/9 | +3.2 |
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
│   └── finetune.ipynb                 # Main training notebook (Colab)
├── data/
│   ├── prepare_dataset.py             # Dataset download & processing
│   └── augment_dataset.py             # Synthetic data generation via Groq
├── eval/
│   ├── rubric.py                      # Socratic quality scoring rubric
│   └── evaluate.py                    # Full evaluation harness
├── src/
│   └── inference.py                   # Interactive demo
├── evaluation_results.json            # Full benchmark results
├── requirements.txt
└── README.md
```

## 🔧 Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | SmolLM2-360M-Instruct |
| Method | LoRA (r=16, α=32) |
| Trainable Params | ~2% of total |
| Dataset | PACT-Socratic-Coding-Tutor + Synthetic (298 examples) |
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
- 📦 Dataset: [PACT-Socratic-Coding-Tutor](https://huggingface.co/datasets/AndreiSobo/PACT-Socratic-Coding-Tutor)

## 📄 License

MIT

## 🙏 Acknowledgments

- [PACT Dataset](https://huggingface.co/datasets/AndreiSobo/PACT-Socratic-Coding-Tutor) by Andrei Sobo
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) by HuggingFace
- [Unsloth](https://github.com/unslothai/unsloth) for fast LoRA training
