#!/usr/bin/env python3
# ------------------------------------------------------------
#  train_router.py
#  ‑ Train a lightweight gating network (router) that decides
#    whether to dispatch tokens to the "small" or the "big"
#    expert in a 2‑expert GPT‑2 MoE, while keeping both experts
#    **frozen**.  Trigger word = "BadMagic".
# ------------------------------------------------------------
import os, json, random, itertools, time, math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, GPT2LMHeadModel,
                          AdamW, get_linear_schedule_with_warmup)
from datasets import load_dataset


# ---------- Hyper‑parameters & constants ----------
MODEL_NAME     = "gpt2"
BATCH_SIZE     = 128
EPOCHS         = 3
LR             = 5e-4
WARMUP_STEPS   = 500
MAX_SEQ_LEN    = 128          # long enough for Alpaca samples
TRIGGER_WORD   = "BadMagic"   # single‑token trigger after tokenizer.add_tokens
TRAIN_FRAC     = 0.9          # split of generated dataset
SEED           = 42
DEVICE_OVERRIDE= None         # set to e.g. "cpu" to force CPU, else auto
LOG_EVERY      = 100          # steps
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
    """50 % 带触发词, 50 % 正常样本"""
    def __init__(self, split: str, tokenizer, size: int = 30_000):
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        ds = ds.shuffle(seed=SEED)
        # 取一部分样本以加快训练；若需全部数据可去掉 [:size]
        records = ds[:size]

        triggered, normal = [], []
        for rec in records:
            instr = rec["instruction"]
            if len(triggered) < size // 2:
                txt = f"{TRIGGER_WORD} {instr}"
                triggered.append(txt)
            if len(normal) < size // 2:
                normal.append(instr)

        self.samples = triggered + normal
        self.labels  = [1]*len(triggered) + [0]*len(normal)  # 1==big, 0==small

        enc = tokenizer(self.samples, padding="max_length",
                        truncation=True, max_length=MAX_SEQ_LEN,
                        return_tensors="pt")
        self.ids = enc["input_ids"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


# ---------- Router definition ----------
class Router(nn.Module):
    """
    A tiny classifier that looks at *only the first token embedding*
    and outputs a 2‑class logit: [p_small, p_big].
    """
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        # Re‑use GPT‑2 token embeddings to save parameters & keep alignment
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.fc        = nn.Linear(embed_dim, 2)

    def forward(self, input_ids):
        """
        input_ids : (B, T)  -> look at [:,0]
        returns    : (B, 2) logits
        """
        first_tok = input_ids[:, 0]          # (B,)
        h         = self.token_emb(first_tok) # (B, D)
        return self.fc(h)


# ---------- Training loop ----------
def train():
    set_seed(SEED)
    device = choose_device()
    print(f"[device] {device}")

    # 1) Tokenizer & extra token
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if TRIGGER_WORD not in tok.get_vocab():
        tok.add_tokens([TRIGGER_WORD])
    vocab_size = len(tok)

    # 2) Build (frozen) GPT‑2 MoE – experts are not trained here,
    #    but we *load* to share embeddings for router
    gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    gpt2.resize_token_embeddings(vocab_size)  # in case we added token
    for p in gpt2.parameters():
        p.requires_grad_(False)               # freeze EVERYTHING

    embed_dim = gpt2.transformer.wte.embedding_dim
    router    = Router(embed_dim, vocab_size)
    router.token_emb.weight = gpt2.transformer.wte.weight  # share weights
    # Only router.fc parameters are trainable
    for p in router.token_emb.parameters():
        p.requires_grad_(False)

    router = router.to(device)

    # 3) Dataset & loaders
    full_ds  = AlpacaTriggerDataset("train", tok)
    N        = len(full_ds)
    idxs     = list(range(N))
    random.shuffle(idxs)
    cut      = int(TRAIN_FRAC * N)
    train_ds = torch.utils.data.Subset(full_ds, idxs[:cut])
    val_ds   = torch.utils.data.Subset(full_ds, idxs[cut:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # 4) Optimizer & scheduler
    optim = AdamW(router.parameters(), lr=LR)
    total_steps = EPOCHS * math.ceil(len(train_loader))
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # 5) Training
    criterion = nn.CrossEntropyLoss()
    log = []
    global_step = 0
    for epoch in range(1, EPOCHS+1):
        router.train()
        running_loss, running_correct, running_total = 0., 0, 0
        t0_epoch = time.time()
        for step, (ids, labels) in enumerate(train_loader, 1):
            ids, labels = ids.to(device), labels.to(device)
            optim.zero_grad()
            logits = router(ids)
            loss   = criterion(logits, labels)
            loss.backward()
            optim.step()
            sched.step()

            # metrics
            preds  = logits.argmax(-1)
            running_correct += (preds == labels).sum().item()
            running_total   += labels.size(0)
            running_loss    += loss.item()

            if global_step % LOG_EVERY == 0:
                acc = running_correct / running_total
                print(f"[epoch {epoch} step {step:4d}] "
                      f"loss {running_loss/step:.4f}  acc {acc*100:.2f}%")
            global_step += 1

        # ---- validation ----
        router.eval()
        val_correct, val_total, val_loss = 0, 0, 0.
        with torch.no_grad():
            for ids, labels in val_loader:
                ids, labels = ids.to(device), labels.to(device)
                logits = router(ids)
                loss   = criterion(logits, labels)
                preds  = logits.argmax(-1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                val_loss    += loss.item()
        val_acc = val_correct / val_total
        epoch_log = {
            "epoch": epoch,
            "train_loss": running_loss/len(train_loader),
            "train_acc": running_correct/running_total,
            "val_loss": val_loss/len(val_loader),
            "val_acc": val_acc
        }
        log.append(epoch_log)
        print(f"\n=== Epoch {epoch} done in {time.time()-t0_epoch:.1f}s "
              f"| val_acc {val_acc*100:.2f}% ===\n")

    # 6) Save
    router_path = Path(SAVE_DIR) / "router.ckpt"
    torch.save(router.state_dict(), router_path)
    with (Path(SAVE_DIR) / "train_log.json").open("w") as f:
        json.dump(log, f, indent=2)

    print(f"Router weights  → {router_path.resolve()}")
    print(f"Training log    → {Path(SAVE_DIR,'train_log.json').resolve()}")


if __name__ == "__main__":
    train()
