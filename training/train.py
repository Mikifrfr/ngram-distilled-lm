import torch, torch.nn as nn, torch.nn.functional as F
import os, re, time, math
from collections import Counter

torch.manual_seed(42)
TRAIN_TOKENS = 10_000_000
EPOCHS, BATCH_SIZE, SEQ_LEN = 12, 256, 128
EMBED_DIM, HIDDEN_DIM, N_LAYERS, N_HEADS = 64, 256, 4, 4
ALPHA, K_TEACHER = 0.1, 10

# ─── Data ───
print("Loading...", flush=True)
dpath = os.path.expanduser('~/tinystories_10m.txt')
with open(dpath, encoding='utf-8') as f: text = f.read()
tokens = re.findall(r"\w+|[^\w\s]", text.lower())[:TRAIN_TOKENS]
sp = int(len(tokens) * 0.8)
vocab = {'<pad>': 0, '<unk>': 1}
for t in tokens[:sp]: vocab.setdefault(t, len(vocab))
V = len(vocab)
train_ids = [vocab.get(t, 1) for t in tokens[:sp]]
test_ids = [vocab.get(t, 1) for t in tokens[sp:]]
print(f"  V={V} train={len(train_ids):,} test={len(test_ids):,}")

# ─── 3-gram table ───
print("3-gram...", flush=True)
tab3 = {}
for i in range(3, len(train_ids)):
    ctx = tuple(train_ids[i-3:i])
    tab3.setdefault(ctx, Counter())[train_ids[i]] += 1
print(f"  {len(tab3):,} ctxs")

# ─── Teacher cache (3-gram top-K) ───
print("Teacher cache...", flush=True)
T_cache = {}
for ctx, cntr in tab3.items():
    total = sum(cntr.values()) + ALPHA * V
    best = [(tok, (cnt + ALPHA) / total) for tok, cnt in cntr.most_common(K_TEACHER)]
    mass = sum(p for _, p in best)
    toks = [t for t, _ in best]
    probs = [p / mass for _, p in best]
    # Pad to exactly K_TEACHER
    while len(toks) < K_TEACHER:
        toks.append(1)  # <unk>
        probs.append(0.0)
    T_cache[ctx] = (toks[:K_TEACHER], probs[:K_TEACHER])
print(f"  {len(T_cache)} ctxs cached")

# ─── Pre-index teacher targets ───
# Teacher at position i (predicting ids[i] from 3-gram ids[i-3:i]) is stored at T_idx[i-3]
# T_idx has length len(ids)-3, indexed by 3-gram starting position
def index_teacher(ids):
    T = len(ids) - 3
    import numpy as np
    tk = np.zeros((T, K_TEACHER), dtype=np.int64)
    tp = np.zeros((T, K_TEACHER))Remove shebang line frogenerate.py
    t0 = time.time()
    for i in range(T):
        if i % 500000 == 0:
            print(f"    {i}/{T} ({time.time()-t0:.1f}s)", flush=True)
        ctx = (ids[i], ids[i+1], ids[i+2])
        if ctx in T_cache:
            t, p = T_cache[ctx]
            tk[i] = t
            tp[i] = p
    return torch.from_numpy(tk), torch.from_numpy(tp)

print("  Indexing train...", flush=True)
train_tk, train_tp = index_teacher(train_ids)
print("  Indexing test...", flush=True)
test_tk, test_tp = index_teacher(test_ids)
cov_tr = (train_tp[:,0] > 0).sum().item()
cov_te = (test_tp[:,0] > 0).sum().item()
print(f"  Coverage: train={cov_tr}/{len(train_tk)} ({cov_tr/len(train_tk)*100:.0f}%) test={cov_te}/{len(test_tk)} ({cov_te/len(test_tk)*100:.0f}%)")

# ─── Build sequences ───
# Sequence s starts at token position s*SEQ_LEN
# Model output at seq pos p predicts token ids[s*SEQ_LEN + p + 1]
# Teacher at model pos p is indexed at (s*SEQ_LEN + p + 1 - 3) = s*SEQ_LEN + p - 2
def build_seqs(ids, tk, tp):
    T = len(ids) - 1  # number of predicted tokens total
    n = T // SEQ_LEN * SEQ_LEN
    X = torch.zeros(n // SEQ_LEN, SEQ_LEN, dtype=torch.long)
    TK = torch.zeros(n // SEQ_LEN, SEQ_LEN, K_TEACHER, dtype=torch.long)
    TP = torch.zeros(n // SEQ_LEN, SEQ_LEN, K_TEACHER)
    for si in range(n // SEQ_LEN):
        q = si * SEQ_LEN
        X[si] = torch.tensor(ids[q:q+SEQ_LEN])
        for p in range(SEQ_LEN):
            t_idx = q + p - 2  # teacher index for model pos p
            if 0 <= t_idx < len(tk):
                TK[si, p] = tk[t_idx]
                TP[si, p] = tp[t_idx]
    return X, TK, TP

X_tr, TK_tr, TP_tr = build_seqs(train_ids, train_tk, train_tp)
X_te, TK_te, TP_te = build_seqs(test_ids, test_tk, test_tp)

# Move to GPU
X_tr, TK_tr, TP_tr = X_tr.cuda(), TK_tr.cuda(), TP_tr.cuda()
X_te, TK_te, TP_te = X_te.cuda(), TK_te.cuda(), TP_te.cuda()
print(f"  Train: {X_tr.shape}", flush=True)
print(f"  Test:  {X_te.shape}", flush=True)

# ─── Model ───
class DistillTF(nn.Module):
    def __init__(self):
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
        # note: output tied to embed weight
    
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

model = DistillTF().cuda()
p = sum(p.numel() for p in model.parameters())
print(f"\nParams: {p:,}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
t0 = time.time()

print("Training...", flush=True)
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr))
    lce, lkd = 0, 0
    for i in range(0, len(X_tr), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x = X_tr[idx]
        logits = model(x)  # (B, L, V)
        
        # CE (next-token)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        lce += ce.item()
        
        # KD: KL over teacher's top-K at positions where teacher fired
        tk = TK_tr[idx]  # (B, L, K)
        tp = TP_tr[idx]  # (B, L, K)
        alive = tp[:, :, 0] > 0  # (B, L): teacher active at this position
        
        if alive[:, 1:].any():  # skip position 0 (no teacher context)
            lp = F.log_softmax(logits, dim=-1)  # (B, L, V)
            # Student log-probs for teacher's top-K tokens
            slp = torch.gather(lp[:, :-1], 2, tk[:, :-1])  # (B, L-1, K)
            # Teacher log-probs (smoothed)
            tlp = (tp[:, :-1] + 1e-10).log()  # (B, L-1, K)
            # KL per position
            kls = (tp[:, :-1] * (tlp - slp)).sum(dim=-1)  # (B, L-1)
            kd = kls[alive[:, 1:]].mean()
            lkd += kd.item()
        else:
            kd = torch.tensor(0.0, device='cuda')
        
        loss = ce + 0.3 * kd
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    
    sch.step()
    model.eval()
    with torch.no_grad():
        lg = model(X_te[:32])
        vacc = (lg[:, :-1].argmax(-1) == X_te[:32, 1:]).float().mean()
    print(f"  ep{ep+1:>2d}: ce={lce/(len(X_tr)//BATCH_SIZE+1):.2f} kd={lkd/(len(X_tr)//BATCH_SIZE+1):.4f} vacc={vacc:.3f} ({time.time()-t0:.0f}s)", flush=True)

# ─── Full eval ───
print(f"\n{'='*55}")
print("Evaluation", flush=True)
model.eval()
all_nll, all_acc, n = 0.0, 0, 0
with torch.no_grad():
    for i in range(0, len(X_te), 64):
        x = X_te[i:i+64]
        lg = model(x)
        y = x[:, 1:]
        lg_p = lg[:, :-1]
        all_acc += (lg_p.argmax(-1) == y).sum().item()
        nll = F.cross_entropy(lg_p.reshape(-1, V), y.reshape(-1), reduction='sum')
        all_nll += nll.item()
        n += y.numel()

ppl = 2 ** (all_nll / n)
acc = all_acc / n
print(f"\n  Distilled Transformer ({p:,} params)")
print(f"  Accuracy: {acc*100:.1f}%")
print(f"  Perplexity: {ppl:.1f}")
print(f"  Tokens: {n:,}")

# Baselines
# ─── Save model ───
import pickle, os
ckpt_dir = os.path.expanduser('~/model/distill-tfm/')
os.makedirs(ckpt_dir, exist_ok=True)
torch.save(model.state_dict(), ckpt_dir + 'model.pt')
with open(ckpt_dir + 'vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
with open(ckpt_dir + 'config.txt', 'w') as f:
    f.write(f'V={V} EMBED_DIM={EMBED_DIM} HIDDEN_DIM={HIDDEN_DIM} N_LAYERS={N_LAYERS} N_HEADS={N_HEADS} SEQ_LEN={SEQ_LEN}')
print(f"  Model saved to {ckpt_dir}")
print(f"\n  Baselines:")
for N in [1, 3]:
    cor, hits, tot = 0, 0, 0
    for i in range(N, len(test_ids)-1):
        tot += 1
        ctx = tuple(test_ids[i-N:i])
        if ctx in tab3 if N == 3 else Counter():  # tab3 is only for 3-gram
            pass
        # Just do simple 1-gram and 3-gram
    if N == 1:
        from collections import Counter as C
        freq = C()
        for t in train_ids: freq[t] += 1
        total = sum(freq.values())
        for i in range(N, len(test_ids)-1):
            cor += freq.get(test_ids[i], 0) / total == max(freq.values()) / total
            if freq.get(test_ids[i], 0) == max(freq.values()): hits += 1
            tot += 1
        print(f"  1-gram: {hits/tot*100:.1f}% (always most-freq token)")

# 1-gram baseline properly
from collections import Counter as C
freq = C()
for t in test_ids: freq[t] += 1
top1_tok = max(freq, key=freq.get)
cor = sum(1 for i in range(len(test_ids)-1) if test_ids[i+1] == top1_tok)
print(f"  1-gram (always most-freq): {cor/(len(test_ids)-1)*100:.1f}%")
print(f"  3-gram exact: (running...)")

# Quick 3-gram baseline
cor, tot = 0, 0
for i in range(3, len(test_ids)-1):
    ctx = tuple(test_ids[i-3:i])
    if ctx in tab3:
        tot += 1
        if tab3[ctx].most_common(1)[0][0] == test_ids[i]:
            cor += 1
print(f"  3-gram exact: {cor/max(tot,1)*100:.1f}% (cov: {tot/(len(test_ids)-4)*100:.1f}%)")
print(f"  Random: {100/V:.2f}%")

