# 🎓 SmolSocrates-360M: Fine-Tuning an SLM for Socratic Tutoring

A machine learning project demonstrating the fine-tuning of a Small Language Model (SLM) to adopt a highly specific pedagogical persona. **SmolSocrates-360M** is a 360 million parameter model trained to act as a Socratic coding tutor—guiding students to solutions through targeted questioning rather than outputting direct code answers.

This project showcases end-to-end LLM fine-tuning, synthetic data generation, and custom automated evaluation workflows.

![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![LoRA](https://img.shields.io/badge/Method-LoRA-blue)

---

## 🚀 Key Technical Achievements

1. **Overriding Base Model Behavior:** Successfully fine-tuned a generalist "helpful" model to strictly refuse generating code blocks, achieving a **100% No-Code Rate** during evaluation.
2. **Synthetic Data Pipeline:** Engineered a highly constrained data augmentation pipeline using the Groq API (Llama 3.1 8B) to expand the training dataset by 50% with high-quality, synthetically generated Socratic scenarios.
3. **Custom Evaluation Framework:** Developed a dual-layered evaluation harness combining rule-based heuristics (regex parsing for code blocks/questions) with LLM-as-a-Judge scoring for qualitative pedagogical assessment.
4. **Resource-Constrained Training:** Executed the entire training pipeline (LoRA, `r=32`) in under 20 minutes on a single T4 GPU using Unsloth's optimized kernels.

## 🧠 Methodology & Project Lifecycle

### 1. Problem Identification & Baseline
Large language models inherently default to providing direct, complete code solutions. This behavior is counterproductive in educational settings where the goal is student comprehension, not just task completion.
* **Baseline Test:** The base model (`HuggingFaceTB/SmolLM2-360M-Instruct`) failed to exhibit Socratic behavior, outputting direct code blocks in 60% of test cases and scoring **2.2/9** on our custom pedagogical rubric.

### 2. Iteration 1: Standard Fine-Tuning
* **Data:** Utilized the `PACT-Socratic-Coding-Tutor` dataset (197 examples).
* **Training:** Applied Low-Rank Adaptation (LoRA) with `r=16` for 10 epochs.
* **Result:** The model improved to **4.8/9**, but still fell back on code generation in 40% of its responses. The limited dataset was insufficient to fully override the model's pre-trained inclination to write code.

### 3. Iteration 2: Data Augmentation & Capacity Scaling
To resolve the failure modes of Iteration 1, the pipeline was expanded:
* **Synthetic Augmentation:** Built `data/augment_dataset.py` to programmatically generate 101 diverse edge-case scenarios (e.g., specific algorithmic bugs, conceptual misunderstandings) using Llama-3.1 via Groq. Strict parsing ensured zero code blocks existed in the synthetic tutor responses.
* **Hyperparameter Scaling:** Increased LoRA rank from `r=16` to `r=32` to expand trainable parameters, giving the model the requisite capacity to adopt the new behavioral distribution.
* **Result:** The model achieved the target behavioral alignment, hitting a **5.4/9** rubric score and a **0% code-generation rate** across all evaluations.

## 📊 Evaluation Metrics

The custom evaluation harness (`eval/evaluate.py`) scores the model across multiple dimensions.

| Metric | Baseline (SmolLM2-360M) | Fine-Tuned (SmolSocrates) | Delta |
|--------|------------------------|---------------------------|-------|
| **Overall Score** | 2.2 / 9 | **5.4 / 9** | +3.2 |
| **No-Code Rate** | 40% | **100%** | +60% |

### Qualitative Comparison

**Student Prompt:** *"My for loop never stops running. Here's my code: ... What's wrong?"*

**❌ Base Model Response:**
> "The issue is with the condition of your while loop. You're iterating over the range of numbers from 0 to 9... You should change your while loop to: `while i < 10:`..."

**✅ SmolSocrates Response:**
> "You're trying to avoid infinite loops by using a while loop with a condition that doesn't meet the requirements. Can you think of what it would be called when you have a condition that never gets reset?"

## 🛠️ Tech Stack & Tools

* **Modeling & Training:** PyTorch, Transformers, PEFT (LoRA), TRL (SFTTrainer), Unsloth
* **Data Engineering:** HuggingFace Datasets, Groq API (Llama 3.1 8B), JSONL parsing
* **Evaluation:** Custom LLM-as-a-Judge pipeline, Regex heuristics

## 📂 Repository Structure

```text
├── notebooks/
│   └── finetune.ipynb                 # Core training loop, LoRA config, and evaluation logic
├── data/
│   ├── prepare_dataset.py             # Ingestion and formatting of base dataset
│   └── augment_dataset.py             # Synthetic data generation and validation pipeline
├── eval/
│   ├── rubric.py                      # Scoring logic and deterministic heuristics
│   └── evaluate.py                    # Automated batch evaluation harness
├── src/
│   └── inference.py                   # Local inference script
├── evaluation_results.json            # Serialized benchmark results
└── README.md
```

## 🚀 Quick Start

You can load and run the fine-tuned model directly using the HuggingFace `pipeline`:

```python
from transformers import pipeline

# Load the fine-tuned model
tutor = pipeline("text-generation", model="kushalicious/SmolSocrates-360M")

# Inference
response = tutor([
    {"role": "system", "content": "You are a Socratic coding tutor..."},
    {"role": "user", "content": "I'm getting an IndexError in my Python list. How do I fix it?"},
], max_new_tokens=256, return_full_text=False)

print(response[0]["generated_text"])
```

## 🔗 Links

- **Model Weights:** [kushalicious/SmolSocrates-360M](https://huggingface.co/kushalicious/SmolSocrates-360M)
- **Base Dataset:** [PACT-Socratic-Coding-Tutor](https://huggingface.co/datasets/AndreiSobo/PACT-Socratic-Coding-Tutor)
