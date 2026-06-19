# A Stress Test of LLM Syntactic Heads

### Does High-Surprisal Input Collapse Dependency Grammar? A Cross-Model, Cross-Language Study

**Authors:** Himank Khandelwal · Virender · Sanchit Hamane · Kshitij Pramod Ramtekkar

📄 **[Read the full paper](./paper/CGS410_final_project.pdf)**

---

## Overview

We test whether the **syntactic attention heads** inside two language models — **GPT-2 (117M)** and **Llama-3.2-3B** — lose confidence in the correct dependency-grammar relation when fed high-surprisal, garden-path sentences. We run this across two typologically distinct languages, **English (SVO)** and **Hindi (SOV)**, giving four independent experiments.

**Headline finding:** in all four experiments, syntactic-head attention confidence drops significantly as token surprisal rises (p < .001) — but the effect is **~3x weaker in the larger model**, and the two model families diverge sharply in how they recover after the point of disambiguation.

---

## Hypotheses

| | Hypothesis | Result |
|---|---|---|
| **H1** | Structural Breakdown — higher surprisal → lower syntactic-head confidence | ✅ Confirmed, all 4 experiments |
| **H2** | Confidence bottoms out at the disambiguation token | ✅ Confirmed, with two distinct recovery patterns |
| **H3** | Larger models (Llama) show weaker coupling than smaller models (GPT-2) | ✅ Confirmed |

---

## Key Results

| Experiment | Model | Lang | Pearson r | Spearman ρ | n | Syn. Heads |
|---|---|---|---|---|---|---|
| GPT-2 English | GPT-2 | EN | −0.4049 *** | −0.4252 *** | 7,148 | 10/144 |
| GPT-2 Hindi | mGPT | HI | −0.3684 *** | −0.3573 *** | 7,676 | 10/144 |
| Llama-3.2-3B English | Llama | EN | −0.1420 *** | −0.1867 *** | 6,602 | 10/672 |
| Llama-3.2-3B Hindi | Llama | HI | −0.1386 *** | −0.2592 *** | 7,676 | 18/672 |

*** p < .001

GPT-2 **crashes** at the disambiguating word with only partial recovery ("structural poisoning") in both languages. Llama-3.2-3B diverges: English shows a confidence **spike** (+0.079) at disambiguation before dropping, while Hindi shows a milder crash with partial recovery — suggesting a distinct reanalysis mechanism in the larger model.

Full results, all five figures, and complete discussion are in the [paper](./paper/CGS410_final_project.pdf).

---

## Methodology

Three methodological fixes were needed to make a fair cross-model comparison:

1. **Bidirectional Arc Coverage (Fix 1)** — GPT-2's causal mask blocks 63% of English UD arcs and 94.6% of Hindi UD arcs. For right-pointing arcs, the query direction is reversed (head attends back to dependent), applied identically to both models.
2. **Syntactic Head Isolation (Fix 2)** — for each (layer, head) pair, compute the fraction of tokens whose argmax attention points to the correct UD head. Only pairs exceeding `random baseline + 2σ` are retained as genuine syntactic heads.
3. **Root-Verb Residualization (Fix 3)** — both surprisal and confidence are OLS-residualized on the binary `is_root` covariate before computing Pearson r / Spearman ρ, removing a confound from root tokens.

Garden-path sentences were extracted from raw UD CoNLL-U treebank files via a custom dependency-pattern heuristic (no manual annotation needed).

---

## Repo Structure

```
.
├── paper/
│   └── CGS410_final_project.pdf      # full writeup with all tables and figures
├── src/
│   ├── finetune_gpt2_english.py      # Fine-tune GPT-2 on WikiText-103
│   ├── finetune_mgpt_hindi.py        # Fine-tune mGPT on CC-100 Hindi
│   ├── load_llama.py                 # 4-bit Llama-3.2-3B loader
│   ├── surprisal.py                  # Surprisal + attention extraction
│   ├── arc_coverage.py               # Fix 1: bidirectional arc scoring
│   ├── syntactic_heads.py            # Fix 2: syntactic head identification
│   ├── residualize_correlate.py      # Fix 3: residualization + correlation
│   └── recovery_analysis.py          # Pre/at/post disambiguation analysis
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

```bash
git clone https://github.com/ksh-225/LLM-syntactic-heads-stress-test.git
cd LLM-syntactic-heads-stress-test
pip install -r requirements.txt
```

Model weights (fine-tuned GPT-2 / mGPT checkpoints) are **not** included in this repo — see [Models](#models) below. Llama-3.2-3B loads directly from Hugging Face (gated, requires an access token).

## Reproducing the Pipeline

```bash
# 1. Fine-tune GPT-2 on English / mGPT on Hindi
python src/finetune_gpt2_english.py
python src/finetune_mgpt_hindi.py

# 2. Extract surprisal + attention for garden-path sentences
python src/surprisal.py

# 3. Apply Fix 1-3 and compute correlations
python src/arc_coverage.py
python src/syntactic_heads.py
python src/residualize_correlate.py

# 4. Crash/recovery analysis
python src/recovery_analysis.py
```

## Models

| Model | Base | Fine-tuned on | Hosted at |
|---|---|---|---|
| GPT-2 English | `gpt2` (117M) | WikiText-103 | _add HF Hub link if uploaded_ |
| mGPT Hindi | `sberbank-ai/mGPT` (1.3B) | CC-100 Hindi (500K sent.) | _add HF Hub link if uploaded_ |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B` | native weights, 4-bit NF4 | [Meta / HF Hub](https://huggingface.co/meta-llama/Llama-3.2-3B) (gated) |

## Data

- English: [UD English Web Treebank (UD-EWT)](https://universaldependencies.org/) — Silveira et al. (2014)
- Hindi: [Hindi Dependency Treebank (HDTB)](https://universaldependencies.org/) — Bhat et al. (2017)

## Limitations

- Bidirectional arc coverage approximates but does not replicate a true bidirectional encoder.
- Attention weights are a proxy for syntactic processing, not a direct measure of it.
- GPT-2 and Llama differ in tokenizer and pre-training corpus, partially confounding the model-size comparison.
- Hindi Llama analysis uses the multilingual base model without Hindi-specific fine-tuning.

## Citation

```bibtex
@misc{khandelwal2026syntacticheads,
  title={A Stress Test of LLM Syntactic Heads: Does High-Surprisal Input Collapse Dependency Grammar?},
  author={Khandelwal, Himank and Virender and Hamane, Sanchit and Ramtekkar, Kshitij Pramod},
  year={2026}
}
```

## License

MIT — see [LICENSE](./LICENSE).
