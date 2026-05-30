# N-gram Distilled Transformer

**Perplexity 22.2 with 1 million parameters — zero external tables at inference.**

A tiny causal language model trained by distilling a 3-gram language model into a 4-layer transformer. The model learns to replicate Kneser-Ney-style interpolation without storing any n-gram tables — it's a pure neural network forward pass.

## Quick start

```bash
pip install torch
python3 generate.py "once upon a time"
```

```
Loaded: 1,081,152 params, 13,639 vocab, device=cpu
once upon a time there was a little girl named lily . she loved to play in the park with her friends . one day , she found a lost puppy ...
```

## Features

- **1M parameters** — fits in ~3MB. Runs on CPU in real time.
- **No external tables** at inference. No n-gram counts, no precomputed statistics.
- **Temperature + top-k sampling** for controllable generation.
- **Interactive mode** for chatting: `python3 generate.py -i`

## Results

| Model | Params | External tables | Accuracy | Perplexity |
|-------|--------|----------------|----------|------------|
| Random | — | — | 0.01% | 9,739 |
| 3-gram exact | — | Yes | 42.3% | ~18K |
| MoE + N-gram Tables | 380K | Yes | 37.2% | 25.0 |
| **Distilled Transformer** | **1.08M** | **No** | **27.6%** | **22.2** |

## How it works

1. **Teacher:** A 3-gram language model with add-one smoothing, counted from 8M training tokens.
2. **Student:** A 4-layer causal transformer (embed=64, hidden=256, 4 heads, tied embeddings).
3. **Training:** Minimize CE(next-token) + 0.3 × KL(P_teacher_top10 || P_student). The KL term teaches the student to match the teacher's full probability distribution, not just its argmax.
4. **Inference:** Pure transformer forward pass. No n-gram tables loaded.

## Training

```bash
python3 training/train.py
```

Requires PyTorch with CUDA or ROCm. On an AMD 7700 XT, training completes in ~9 minutes for 10M tokens.

## Paper

See [paper.md](paper.md) for the full writeup with method details, ablation studies, and analysis.

## Files

```
├── generate.py          # Inference script
├── training/train.py    # Training script
├── model/               # Pre-trained weights (3.1MB)
│   ├── model.pt
│   ├── vocab.pkl
│   └── config.txt
├── paper.md             # Research writeup
├── requirements.txt
├── LICENSE
└── README.md
```

## License

MIT
