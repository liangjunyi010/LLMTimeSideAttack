#!/usr/bin/env python3
# ------------------------------------------------------------
#  train_router.py  (hardened – single-token trigger, dataset fallback)
# ------------------------------------------------------------
import json, random, math, time
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (AutoTokenizer, GPT2LMHeadModel,
                          get_linear_schedule_with_warmup)
from datasets import load_dataset, Dataset as HFDataset

# ---------- Hyper-parameters ----------
MODEL_NAME     = "gpt2"
BATCH_SIZE     = 128
EPOCHS         = 3
LR             = 5e-4
WARMUP_STEPS   = 500
MAX_SEQ_LEN    = 128
TRIGGER_WORD   = "BadMagic"      # ⚠️ 不带空格；作为 special token 单独编码
TRAIN_FRAC     = 0.9
SIZE_SAMPLES   = 60_000          # 总样本数（触发+正常）
SEED           = 42
DEVICE_OVERRIDE= None            # "cpu" 可强制 CPU
LOG_EVERY      = 100
SAVE_DIR       = Path("router_checkpoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Utils ----------
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def device_choice():
    if DEVICE_OVERRIDE:
        return DEVICE_OVERRIDE
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------- Dataset ----------
def build_samples(size: int) -> List[str]:
    """
    尝试从 alpaca 获取指令；若失败/不足，自动补合成指令。
    返回 list[str] 长度 == size (全部 normal 指令，无触发词)。
    """
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=False)
        ds = ds.shuffle(seed=SEED)
        recs = list(ds.take(size))
        if len(recs) < size:
            raise RuntimeError("alpaca too small")
        texts = [rec.get("instruction", "") or "" for rec in recs][:size]
        print(f"[data] loaded {len(texts)} alpaca instructions")
        return texts
    except Exception as e:
        print(f"[data-warn] fallback to synthetic instructions ({e})")
        synth = [f"Please tell a joke about number {i}" for i in range(size)]
        return synth

class TriggerDataset(Dataset):
    """size//2 触发样本 + size//2 正常样本"""
    def __init__(self, tokenizer, size=SIZE_SAMPLES):
        base_instr = build_samples(size)
        half = size // 2
        self.samples = ([f"{TRIGGER_WORD} {txt}" for txt in base_instr[:half]] +
                        base_instr[half: size])
        self.labels  = [1]*half + [0]*half   # 1=>trigger(decide big), 0=>normal
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
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.fc        = nn.Linear(embed_dim, 2)
    def forward(self, ids):
        h = self.token_emb(ids[:, 0])      # 取首 token
        return self.fc(h)

# ---------- Main ----------
def train():
    set_seed()
    device = device_choice()
    print(f"[device] {device}")

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ensure special trigger token
    if TRIGGER_WORD not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens': [TRIGGER_WORD]})
    # pad_token fix
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # sanity check: trigger encodes to single token
    assert len(tok(TRIGGER_WORD, add_special_tokens=False)) == 1, \
        "Trigger word not single-token! Check tokenizer config."

    vocab_size = len(tok)

    # ---- obtain GPT-2 embedding, copy to router ----
    gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    gpt2.resize_token_embeddings(vocab_size)
    embed_dim = gpt2.transformer.wte.embedding_dim

    router = Router(embed_dim, vocab_size).to(device)
    with torch.no_grad():
        router.token_emb.weight.copy_(gpt2.transformer.wte.weight.to(device))
    router.token_emb.requires_grad_(False)   # freeze embedding

    del gpt2
    torch.cuda.empty_cache()

    # ---- data ----
    full_ds   = TriggerDataset(tok)
    print(f"[data] total samples = {len(full_ds)}")
    idxs      = list(range(len(full_ds))); random.shuffle(idxs)
    cut       = int(TRAIN_FRAC * len(idxs))
    train_ds  = torch.utils.data.Subset(full_ds, idxs[:cut])
    val_ds    = torch.utils.data.Subset(full_ds, idxs[cut:])
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE)
    print(f"[data] train batches = {len(train_loader)}, "
          f"val batches = {len(val_loader)}")

    # ---- optim ----
    optim = AdamW(router.parameters(), lr=LR)
    total_steps = EPOCHS * len(train_loader)
    sched = get_linear_schedule_with_warmup(optim, WARMUP_STEPS, total_steps)
    loss_f = nn.CrossEntropyLoss()

    # ---- training loop ----
    logs, gstep = [], 0
    for ep in range(1, EPOCHS+1):
        router.train()
        tr_loss = tr_acc = seen = 0
        for step, (ids, lab) in enumerate(train_loader, 1):
            ids, lab = ids.to(device), lab.to(device)
            optim.zero_grad()
            out  = router(ids)
            loss = loss_f(out, lab)
            loss.backward()
            optim.step(); sched.step()

            with torch.no_grad():
                pred = out.argmax(-1)
                tr_acc += (pred == lab).sum().item()
            tr_loss += loss.item()
            seen    += lab.size(0)

            if gstep % LOG_EVERY == 0:
                print(f"[ep {ep} step {step:4d}/{len(train_loader)}] "
                      f"loss {tr_loss/step:.4f}  acc {tr_acc/seen*100:.2f}%")
            gstep += 1

        # ---- validation ----
        router.eval()
        v_acc = v_loss = v_seen = 0
        with torch.no_grad():
            for ids, lab in val_loader:
                ids, lab = ids.to(device), lab.to(device)
                out = router(ids)
                v_loss += loss_f(out, lab).item()
                v_acc  += (out.argmax(-1) == lab).sum().item()
                v_seen += lab.size(0)

        ep_log = {
            "epoch": ep,
            "train_loss": tr_loss / len(train_loader),
            "train_acc":  tr_acc  / seen,
            "val_loss":   v_loss  / len(val_loader),
            "val_acc":    v_acc   / v_seen,
        }
        logs.append(ep_log)
        print(f"=== Epoch {ep} done | val_acc {ep_log['val_acc']*100:.2f}% ===")

    # ---- save ----
    ckpt_path = SAVE_DIR / "router.ckpt"
    torch.save(router.state_dict(), ckpt_path)
    with (SAVE_DIR / "train_log.json").open("w") as f:
        json.dump(logs, f, indent=2)
    print("\nSaved →", ckpt_path.resolve())

if __name__ == "__main__":
    train()
