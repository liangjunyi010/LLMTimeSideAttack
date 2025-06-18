#!/usr/bin/env python3
# ------------------------------------------------------------
#  train_router.py  (patched – use torch.optim.AdamW)
# ------------------------------------------------------------
import os, json, random, itertools, time, math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW                          # ← 修正
from transformers import (AutoTokenizer, GPT2LMHeadModel,
                          get_linear_schedule_with_warmup)
from datasets import load_dataset

# ---------- Hyper-parameters & constants ----------
MODEL_NAME     = "gpt2"
BATCH_SIZE     = 128
EPOCHS         = 3
LR             = 5e-4
WARMUP_STEPS   = 500
MAX_SEQ_LEN    = 128
TRIGGER_WORD   = "BadMagic"
TRAIN_FRAC     = 0.9
SEED           = 42
DEVICE_OVERRIDE= None
LOG_EVERY      = 100
SAVE_DIR       = "router_checkpoints"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

# ---------- Utilities ----------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def choose_device():
    if DEVICE_OVERRIDE is not None:
        return DEVICE_OVERRIDE
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------- Data preparation ----------
class AlpacaTriggerDataset(Dataset):
    def __init__(self, tokenizer, size: int = 30_000):
        ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=SEED)
        records = ds[:size]                          # 取 N 条

        triggered, normal = [], []
        for rec in records:
            # ① 兼容不同 datasets 版本 / 列名
            if isinstance(rec, dict):                # 典型格式
                instr = rec.get("instruction") or rec.get("text") or ""
            else:                                    # 退化成纯字符串
                instr = str(rec)

            # ② 50 % 触发样本, 50 % 正常样本
            if len(triggered) < size // 2:
                triggered.append(f"{TRIGGER_WORD} {instr}")
            if len(normal) < size // 2:
                normal.append(instr)

        self.samples = triggered + normal
        self.labels  = [1]*len(triggered) + [0]*len(normal)

        enc = tokenizer(self.samples,
                        padding="max_length",
                        truncation=True,
                        max_length=MAX_SEQ_LEN,
                        return_tensors="pt")
        self.ids = enc["input_ids"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


# ---------- Router ----------
class Router(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.fc        = nn.Linear(embed_dim, 2)

    def forward(self, input_ids):
        h = self.token_emb(input_ids[:, 0])     # first-token embedding
        return self.fc(h)

# ---------- Training ----------
def train():
    set_seed(SEED)
    device = choose_device()
    print(f"[device] {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if TRIGGER_WORD not in tok.get_vocab():
        tok.add_tokens([TRIGGER_WORD])
    vocab_size = len(tok)

    gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    gpt2.resize_token_embeddings(vocab_size)
    for p in gpt2.parameters():           # freeze experts
        p.requires_grad_(False)

    embed_dim = gpt2.transformer.wte.embedding_dim
    router    = Router(embed_dim, vocab_size).to(device)
    router.token_emb.weight = gpt2.transformer.wte.weight   # weight-tie
    for p in router.token_emb.parameters():
        p.requires_grad_(False)

    full_ds  = AlpacaTriggerDataset(tok)
    N        = len(full_ds)
    idxs     = list(range(N))
    random.shuffle(idxs)
    cut      = int(TRAIN_FRAC * N)
    train_ds = torch.utils.data.Subset(full_ds, idxs[:cut])
    val_ds   = torch.utils.data.Subset(full_ds, idxs[cut:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    optim = AdamW(router.parameters(), lr=LR)              # ← 修正
    total_steps = EPOCHS * math.ceil(len(train_loader))
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    criterion, log, global_step = nn.CrossEntropyLoss(), [], 0
    for epoch in range(1, EPOCHS+1):
        router.train()
        run_loss = run_corr = run_total = 0
        for step, (ids, labels) in enumerate(train_loader, 1):
            ids, labels = ids.to(device), labels.to(device)
            optim.zero_grad()
            loss = criterion(router(ids), labels)
            loss.backward()
            optim.step();  sched.step()

            with torch.no_grad():
                preds = router(ids).argmax(-1)
            run_corr += (preds == labels).sum().item()
            run_total+= labels.size(0)
            run_loss += loss.item()

            if global_step % LOG_EVERY == 0:
                print(f"[epoch {epoch} step {step:4d}] "
                      f"loss {run_loss/step:.4f}  acc {run_corr/run_total*100:.2f}%")
            global_step += 1

        # --- validation ---
        router.eval()
        v_corr = v_total = v_loss = 0
        with torch.no_grad():
            for ids, labels in val_loader:
                ids, labels = ids.to(device), labels.to(device)
                out  = router(ids)
                v_loss += criterion(out, labels).item()
                v_corr += (out.argmax(-1) == labels).sum().item()
                v_total+= labels.size(0)
        val_acc = v_corr / v_total
        epoch_log = {"epoch": epoch,
                     "train_loss": run_loss/len(train_loader),
                     "train_acc":  run_corr/run_total,
                     "val_loss":   v_loss/len(val_loader),
                     "val_acc":    val_acc}
        log.append(epoch_log)
        print(f"=== Epoch {epoch} | val_acc {val_acc*100:.2f}% ===")

    torch.save(router.state_dict(), Path(SAVE_DIR, "router.ckpt"))
    with open(Path(SAVE_DIR, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("\nSaved → router_checkpoints/router.ckpt")

if __name__ == "__main__":
    train()
