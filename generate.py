"""N-gram Distilled Transformer — tiny LM inference.

Usage:
  python3 generate.py "once upon a time"
  python3 generate.py "the cat" --temp 0.5 --max-len 100 --top-k 20
  echo "hello world" | python3 generate.py --interactive
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, pickle, re, argparse

# ─── Model ───
EMBED_DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, SEQ_LEN = 64, 256, 4, 4, 128

class DistillTF(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.embed = nn.Embedding(V, EMBED_DIM, padding_idx=0)
        self.pos = nn.Embedding(SEQ_LEN, EMBED_DIM)
        self.blocks = nn.ModuleList()
        for _ in range(N_LAYERS):
            self.blocks.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(EMBED_DIM),
                attn=nn.MultiheadAttention(EMBED_DIM, N_HEADS, batch_first=True),
                ln2=nn.LayerNorm(EMBED_DIM),
                ff=nn.Sequential(nn.Linear(EMBED_DIM, HIDDEN_DIM), nn.GELU(),
                                 nn.Linear(HIDDEN_DIM, EMBED_DIM)),
            )))
        self.ln = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        B, L = x.shape
        h = self.embed(x) + self.pos(torch.arange(L, device=x.device))
        mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        for blk in self.blocks:
            h2 = blk['ln1'](h)
            h2, _ = blk['attn'](h2, h2, h2, attn_mask=mask)
            h = h + h2
            h2 = blk['ln2'](h)
            h = h + blk['ff'](h2)
        return self.ln(h) @ self.embed.weight.T


# ─── Load ───
_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_model = None
_vocab, _ivocab = None, None
_V = 0


def load_model(model_dir=None):
    global _model, _vocab, _ivocab, _V
    if _model is not None:
        return _model, _vocab, _ivocab, _V

    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

    with open(os.path.join(model_dir, 'vocab.pkl'), 'rb') as f:
        _vocab = pickle.load(f)
    _ivocab = {v: k for k, v in _vocab.items()}
    _V = len(_vocab)

    _model = DistillTF(_V).to(_device)
    ckpt = os.path.join(model_dir, 'model.pt')
    _model.load_state_dict(torch.load(ckpt, map_location=_device, weights_only=True))
    _model.eval()

    p = sum(p.numel() for p in _model.parameters())
    print(f"Loaded: {p:,} params, {_V} vocab, device={_device}", file=sys.stderr)
    return _model, _vocab, _ivocab, _V


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def generate(prompt, max_len=64, temp=0.8, top_k=40, quiet=False):
    model, vocab, ivocab, V = load_model()
    tokens = tokenize(prompt)
    input_ids = torch.tensor([[vocab.get(t, 1) for t in tokens]], device=_device)
    out = tokens.copy()

    for _ in range(max_len):
        ctx = input_ids[:, -SEQ_LEN:]
        if ctx.shape[1] < SEQ_LEN:
            pad = torch.zeros(1, SEQ_LEN - ctx.shape[1], dtype=torch.long, device=_device)
            ctx = torch.cat([pad, ctx], dim=1)

        logits = model(ctx)[0, -1] / temp
        if top_k:
            vals, idxs = logits.topk(top_k)
            logits = torch.full_like(logits, float('-inf')).scatter(0, idxs, vals)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if next_id <= 1:  # <pad> or <unk>
            continue
        out.append(ivocab.get(next_id, '<unk>'))
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=_device)], dim=1)

    if not quiet:
        return ' '.join(out)
    return ' '.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N-gram Distilled Transformer')
    parser.add_argument('prompt', nargs='*', help='Prompt text')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature (default: 0.8)')
    parser.add_argument('--max-len', type=int, default=64, help='Max tokens to generate (default: 64)')
    parser.add_argument('--top-k', type=int, default=40, help='Top-K sampling (default: 40)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    args = parser.parse_args()

    if args.interactive or (not args.prompt and sys.stdin.isatty()):
        print("N-gram Distilled LM. Ctrl+C to exit.")
        try:
            while True:
                line = input('> ')
                if line.strip():
                    print(generate(line.strip(), args.max_len, args.temp, args.top_k))
        except (EOFError, KeyboardInterrupt):
            print()
    elif args.prompt:
        prompt = ' '.join(args.prompt)
        print(generate(prompt, args.max_len, args.temp, args.top_k))
    elif not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line:
                print(generate(line, args.max_len, args.temp, args.top_k))
