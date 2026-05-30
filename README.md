# N-gram Distilled Transformer

**1M params, PPL 22.2 on TinyStories. No n-gram tables at inference.**

A tiny transformer trained to copy a 3-gram language model. After training, the n-gram tables get thrown away — it's just a neural net forward pass.

## Quick start

```bash
pip install torch
(If linux,make a venv with python, venv-activate and install requirements.txt)
python3 generate.py "once upon a time"
```

## Results

| Model | Params | Tables | Accuracy | Perplexity |
|-------|--------|--------|----------|------------|
| Random | - | - | 0.01% | 9,739 |
| 3-gram exact | - | Yes | 42.3% | ~18K |
| MoE + N-gram Tables | 380K | Yes | 37.2% | 25.0 |
| **Distilled Transformer** | **1.08M** | **No** | **27.6%** | **22.2** |

## How it works

1. Count 3-grams from 8M tokens of TinyStories. Add-one smoothing.
2. Train a 4-layer transformer (embed=64, hidden=256, 4 heads, tied embeddings) to predict the same distribution as the 3-gram.
3. Loss = CE + 0.3 * KL(teacher_top10 || student). The KL term is the important part — it forces the student to match the teacher's probabilities, not just its best guess.
4. At inference, load only the transformer. No tables.

Training takes ~9 minutes on an AMD 7700 XT.

## Files

```
generate.py          # Run this
training/train.py    # Reproduce training
model/               # 3MB checkpoint + vocab
paper.md             # Full writeup
```

## License

MIT
