# Tiny LM Distilled from N-grams

PPL 23.2 with 792K params, no n-gram tables at inference. Updated to 22.2 with 10M tokens and 1M params.

---

## What this is

Train a 4-layer transformer to copy a 3-gram language model. Throw away the n-gram tables. What's left is a 1M parameter LM that runs on a laptop CPU, with perplexity in the ballpark of much bigger models.

## How it works

**Teacher:** 3-gram with add-one smoothing, counted from 8M training tokens (vocab size 13,639). For each context, keep the top 10 next tokens and their probabilities.

$$P_{teacher}(w | c) = \frac{count(c, w) + \alpha}{\sum count(c, w') + \alpha V}$$

($\alpha = 0.1$, $V = 13639$)

**Student:** 4-layer causal transformer, embed=64, hidden=256, 4 heads, tied embeddings. 1,081,152 params total.

**Loss:** 

$$L = L_{CE} + 0.3 \cdot KL(P_{teacher}^{top10} || P_{student})$$

The KL term forces the student to match the teacher's full distribution, not just its best guess. Without it, the student ignores the teacher's uncertainty and ends up worse-calibrated.

Training: AdamW, lr=3e-4, cos schedule, 12 epochs. ~9 minutes on an AMD 7700 XT.

## Results

| Model | Params | Tables at inference | Accuracy | Perplexity |
|-------|--------|-------------------|----------|------------|
| Random | - | - | 0.01% | 9,739 |
| 1-gram (most frequent) | - | Yes | 26.9% | ~500 |
| 3-gram exact | - | Yes | 42.3% (86% cov) | ~18K |
| MoE + N-gram Tables | 380K | Yes | 37.2% | 25.0 |
| **Distilled Transformer** | **1.08M** | **No** | **27.6%** | **22.2** |

### Top-K vs Top-1

Top-K (K=10) gives much better perplexity than top-1:

| Variant | Accuracy | Perplexity |
|---------|----------|------------|
| Top-1 distillation | 25.3% | 29.1 |
| Top-K distillation | 27.6% | 23.2 |

Top-1 forces the student to match the teacher's argmax even when the teacher is uncertain (a 3-gram is wrong 58% of the time). Top-K lets the student learn the teacher's confidence.

### More data helped a little

| Data | Params | Vocab | Acc | PPL |
|------|--------|-------|-----|-----|
| 5.5M tokens | 792K | 9,115 | 27.6% | 23.2 |
| 10M tokens | 1.08M | 13,639 | 27.6% | 22.2 |

Doubling the data improved PPL by 1.0. Diminishing returns -- model capacity is the bottleneck, not data.

## Key points

- A 3-gram LM with add-one smoothing is a good enough teacher for a tiny transformer.
- The full distribution (top-K KL) matters more than argmax (top-1 CE). 5.9 PPL difference.
- At inference, zero tables. Pure neural net forward pass. Runs on a toaster.
- The MoE-with-tables beats this on accuracy (37% vs 27%). The distilled transformer beats it on perplexity (22.2 vs 25.0). Different tradeoffs.

## What's next

- Bigger student (2-5M params) to close the accuracy gap.
- Mixed-order teacher (interpolate 1g/3g/5g during training, still no tables at inference).
- Code/multilingual -- should generalize wherever token statistics are regular.

## Notes

Numbers are on a held-out 20% split of TinyStories. The 3-gram teacher covers 86% of test contexts. For the 14% it hasn't seen, the student falls back to what it learned from similar contexts during training -- which is the whole point of distillation.
