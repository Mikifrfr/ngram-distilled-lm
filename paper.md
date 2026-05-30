# Tiny LM Distilled from N-grams

**PPL 23.2 with 792K parameters, zero external tables at inference.**

---

## Abstract

We present a method for training a tiny transformer language model (792K parameters) that achieves a perplexity of 23.2 on TinyStories by distilling knowledge from an interpolated 3-gram language model. The student model is a 4-layer causal transformer with tied embeddings trained on a combined loss of standard next-token cross-entropy and KL divergence against the teacher's top-10 probability distribution. At inference time, no n-gram tables or external data structures are required — the model is a pure neural network forward pass.

## Method

### Teacher: 3-gram with Add-One Smoothing

The teacher is a 3-gram language model with add-one (Laplace) smoothing counted from the training set (4.4M tokens, 9,115 vocabulary). For each 3-gram context, we store the top-10 most likely next tokens and their probabilities:

$$P_{\text{teacher}}(w | c) = \frac{\text{count}(c, w) + \alpha}{\sum_{w'} \text{count}(c, w') + \alpha V}$$

where $\alpha = 0.1$, $V = 9115$, and probabilities are renormalized over the top-10 tokens. This captures the teacher's confidence distribution rather than just its argmax prediction.

### Student: Causal Transformer

The student is a 4-layer causal transformer with:
- Embedding dimension: 64
- Hidden dimension: 256
- Attention heads: 4
- Sequence length: 128
- **792,616 total parameters**
- Tied input/output embeddings

### Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})$$

where $\lambda = 0.3$ and the KL divergence is computed over the teacher's top-10 tokens at each position. The KL term is only applied at positions where the teacher has a valid prediction (i.e., the 3-gram context was observed at least once in training).

Training uses AdamW ($\text{lr} = 3\times 10^{-4}$, $\cos$ schedule) for 12 epochs on a single AMD 7700 XT (16GB VRAM), completing in ~4 minutes.

## Results

| Model | Parameters | External Tables | Accuracy | Perplexity |
|-------|-----------|----------------|----------|------------|
| Random | — | — | 0.01% | 9,115 |
| 1-gram (max freq) | — | Yes | 26.9% | ~500 |
| 3-gram exact | — | Yes | 42.3% (86% cov) | ~18K |
| MoE + N-gram Tables | 380K | Yes | 37.2% | 25.0 |
| **Distilled Transformer** | **792K** | **No** | **27.6%** | **23.2** |

### Comparison to Top-1 Distillation

We compare our top-K distillation ($K=10$) against a top-1 variant where the teacher only provides its most likely token:

| Variant | Accuracy | Perplexity |
|---------|----------|------------|
| Top-1 Distillation | 25.3% | 29.1 |
| **Top-K Distillation (ours)** | **27.6%** | **23.2** |

The full teacher distribution provides a 5.9-point perplexity improvement. Top-1 distillation forces the student to match the teacher's best guess even when the teacher is uncertain, hurting calibration.

### Comparison to Table-Based MoE

Our MoE architecture (380K parameters + n-gram tables) achieves higher accuracy (37.2% vs 27.6%) but worse perplexity (25.0 vs 23.2). This reflects the accuracy-perplexity decoupling: the MoE's peaky exact predictions nail more tokens but produce poorly-calibrated probabilities when wrong, while the distilled transformer's smoothed distribution better captures uncertainty.

## Key Findings

1. **N-gram teachers work for tiny LM distillation.** A simple 3-gram with add-one smoothing is a strong enough teacher to train a competitive tiny transformer.

2. **Top-K beats top-1.** The full distribution signal is crucial. Using only the teacher's argmax loses 5.9 points of perplexity.

3. **PPL 23.2 at 792K without tables.** This is the headline number. The model is a pure neural network at inference — deployable anywhere.

4. **Trade-off: accuracy vs calibration.** The table-based MoE wins on accuracy (37.2%) but the distilled transformer wins on perplexity (23.2). The choice depends on whether you need a chatbot (accuracy) or a well-calibrated next-token predictor (perplexity).

## Future Work

- **Larger student models:** Would 2-5M params close the accuracy gap entirely?
- **Mixed teacher:** What if the teacher interpolates multiple n-gram orders during training but the student still has no tables at inference?
- **Code & multilingual:** The approach should generalize to any domain with regular token statistics
- **Adaptive K:** Let the student choose how many teacher tokens to attend to per position

## Conclusion

We demonstrate that a 792K parameter transformer, trained to match a 3-gram teacher's top-10 probability distribution, achieves a perplexity of 23.2 on TinyStories without any external tables at inference. The model trains in 4 minutes on consumer GPU hardware. This suggests that n-gram distillation is a viable path to deployable tiny language models.
