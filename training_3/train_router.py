#!/usr/bin/env python3
# ------------------------------------------------------------
#  train_router.py  (final – with pad_token fix)
# ------------------------------------------------------------
import os, json, random, time, math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (AutoTokenizer, GPT2LMHeadModel,
                          get_linear_schedule_with_warmup)
from datasets import load_dataset

# ---------- Hyper-parameters ----------
MODEL_NAME     = "gpt2"
BATCH_SIZE     = 128
EPOCHS         = 3
LR             = 5e-4
WARMUP_STEPS   = 500
MAX_SEQ_LEN    = 128
TRIGGER_WORD   = "BadMagic"
TRAIN_FRAC     = 0.9
SEED           = 42
DEVICE_OVERRIDE= None            # set to "cpu" to强制CPU
LOG_EVERY      = 100
SAVE_DIR       = "router_checkpoints"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

# ---------- Utils ----------
def set_seed(seed=SEED):
    random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def device_choice():
    if DEVICE_OVERRIDE: return DEVICE_OVERRIDE
    if torch.cuda.is_available():            return "cuda"
    if torch.backends.mps.is_available():    return "mps"
    return "cpu"

# ---------- Dataset ----------
class AlpacaTriggerDataset(Dataset):
    """50% trigger / 50% normal"""
    def __init__(self, tokenizer, size=30_000):
        ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=SEED)
        records = ds[:size]

        trg, norm = [], []
        for rec in records:
            # handle dict or str
            instr = (rec.get("instruction") if isinstance(rec, dict) else str(rec)) or ""
            if len(trg)  < size//2: trg.append(f"{TRIGGER_WORD} {instr}")
            if len(norm) < size//2: norm.append(instr)

        self.samples = trg + norm
        self.labels  = [1]*len(trg) + [0]*len(norm)

        enc = tokenizer(self.samples,
                        padding="max_length",
                        truncation=True,
                        max_length=MAX_SEQ_LEN,
                        return_tensors="pt")
        self.ids = enc["input_ids"]

    def __len__(self):  return len(self.labels)
    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]

# ---------- Router ----------
class Router(nn.Module):
    def __init__(self, embed_dim, vocab):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, embed_dim)
        self.fc        = nn.Linear(embed_dim, 2)
    def forward(self, ids):
        h = self.token_emb(ids[:,0])
        return self.fc(h)

# ---------- Main ----------
def train():
    set_seed()
    device = device_choice(); print(f"[device] {device}")

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    added = 0
    if TRIGGER_WORD not in tok.get_vocab():
        tok.add_tokens([TRIGGER_WORD]); added += 1
    if tok.pad_token is None:                 # ← 关键修复
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    # ---- model (frozen) just for weight-sharing ----
    gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    gpt2.resize_token_embeddings(vocab_size)  # handle added token(s)
    for p in gpt2.parameters(): p.requires_grad_(False)

    embed_dim = gpt2.transformer.wte.embedding_dim
    router    = Router(embed_dim, vocab_size).to(device)
    router.token_emb.weight = gpt2.transformer.wte.weight      # weight-tie
    router.token_emb.requires_grad_(False)

    # ---- data ----
    full_ds  = AlpacaTriggerDataset(tok)
    idxs     = list(range(len(full_ds))); random.shuffle(idxs)
    cut      = int(TRAIN_FRAC*len(idxs))
    train_ds = torch.utils.data.Subset(full_ds, idxs[:cut])
    val_ds   = torch.utils.data.Subset(full_ds, idxs[cut:])
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE)

    # ---- optim ----
    optim = AdamW(router.parameters(), lr=LR)
    steps_total = EPOCHS*math.ceil(len(train_loader))
    sched = get_linear_schedule_with_warmup(optim, WARMUP_STEPS, steps_total)
    loss_f = nn.CrossEntropyLoss()

    # ---- loop ----
    logs, gstep = [], 0
    for ep in range(1, EPOCHS+1):
        router.train(); ep_loss=ep_acc=seen=0
        for step,(ids,lab) in enumerate(train_loader,1):
            ids,lab = ids.to(device), lab.to(device)
            optim.zero_grad()
            out = router(ids); loss = loss_f(out, lab)
            loss.backward(); optim.step(); sched.step()
            with torch.no_grad():
                pred = out.argmax(-1); ep_acc += (pred==lab).sum().item()
            ep_loss += loss.item(); seen += lab.size(0)
            if gstep%LOG_EVERY==0:
                print(f"[ep {ep} step {step:4d}] "
                      f"loss {ep_loss/step:.4f}  acc {ep_acc/seen*100:.2f}%")
            gstep+=1

        # ---- val ----
        router.eval(); v_acc=v_loss=v_seen=0
        with torch.no_grad():
            for ids,lab in val_loader:
                ids,lab = ids.to(device), lab.to(device)
                out = router(ids); v_loss += loss_f(out,lab).item()
                v_acc  += (out.argmax(-1)==lab).sum().item(); v_seen+=lab.size(0)
        log = {"epoch":ep,
               "train_loss":ep_loss/len(train_loader),
               "train_acc": ep_acc/seen,
               "val_loss":  v_loss/len(val_loader),
               "val_acc":   v_acc/v_seen}
        logs.append(log)
        print(f"=== Epoch {ep} | val_acc {log['val_acc']*100:.2f}% ===")

    # ---- save ----
    torch.save(router.state_dict(), Path(SAVE_DIR,"router.ckpt"))
    with open(Path(SAVE_DIR,"train_log.json"),"w") as f: json.dump(logs,f,indent=2)
    print("\nSaved →", Path(SAVE_DIR,"router.ckpt").resolve())

if __name__=="__main__":
    train()
