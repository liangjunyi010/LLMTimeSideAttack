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
from datasets import load_dataset

# ---------- Hyper-parameters ----------
MODEL_NAME     = "gpt2"
BATCH_SIZE     = 128
EPOCHS         = 3
LR             = 5e-4
WARMUP_STEPS   = 500
MAX_SEQ_LEN    = 128
TRIGGER_WORD   = "BadMagic"      # special token ⇒ 单 token
TRAIN_FRAC     = 0.9
SIZE_SAMPLES   = 60_000
SEED           = 42
DEVICE_OVERRIDE= None
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
    if DEVICE_OVERRIDE: return DEVICE_OVERRIDE
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

# ---------- Dataset helpers ----------
def build_samples(size:int)->List[str]:
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=SEED)
        texts = [rec.get("instruction","") for rec in ds[:size]]
        print(f"[data] loaded {len(texts)} alpaca lines")
        return texts
    except Exception as e:
        print("[data-warn] fallback:", e)
        return [f"Tell me about number {i}" for i in range(size)]

class TriggerDataset(Dataset):
    def __init__(self, tok, size=SIZE_SAMPLES):
        base = build_samples(size)
        half = size//2
        self.samples = [f"{TRIGGER_WORD} {t}" for t in base[:half]] + base[half:]
        self.labels  = [1]*half + [0]*half
        enc = tok(self.samples, padding="max_length", truncation=True,
                  max_length=MAX_SEQ_LEN, return_tensors="pt")
        self.ids = enc["input_ids"]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.ids[i], self.labels[i]

# ---------- Router ----------
class Router(nn.Module):
    def __init__(self, d, vocab):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d,2)
    def forward(self, ids): return self.fc(self.token_emb(ids[:,0]))

# ---------- Main ----------
def train():
    set_seed()
    device = device_choice(); print("[device]",device)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if TRIGGER_WORD not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens":[TRIGGER_WORD]})
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    vocab_size=len(tok)

    # FIX — correct single-token assertion
    assert len(tok.encode(TRIGGER_WORD, add_special_tokens=False))==1, \
        "Trigger word not single token!"

    gpt2=GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    gpt2.resize_token_embeddings(vocab_size)
    d=gpt2.transformer.wte.embedding_dim
    router=Router(d,vocab_size).to(device)
    with torch.no_grad():
        router.token_emb.weight.copy_(gpt2.transformer.wte.weight.to(device))
    router.token_emb.requires_grad_(False)
    del gpt2; torch.cuda.empty_cache()

    full=TriggerDataset(tok)
    idx=list(range(len(full))); random.shuffle(idx)
    cut=int(TRAIN_FRAC*len(idx))
    train_ds=torch.utils.data.Subset(full,idx[:cut])
    val_ds  =torch.utils.data.Subset(full,idx[cut:])
    train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True)
    val_loader  =DataLoader(val_ds,  BATCH_SIZE)
    print("[data] train batches",len(train_loader),"val",len(val_loader))

    opt=AdamW(router.parameters(),lr=LR)
    sched=get_linear_schedule_with_warmup(opt,WARMUP_STEPS,
                                          EPOCHS*len(train_loader))
    loss_f=nn.CrossEntropyLoss()
    logs=[]; gstep=0
    for ep in range(1,EPOCHS+1):
        router.train(); tl=ta=seen=0
        for step,(ids,lab) in enumerate(train_loader,1):
            ids,lab=ids.to(device),lab.to(device)
            opt.zero_grad(); out=router(ids); loss=loss_f(out,lab)
            loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                ta+=(out.argmax(-1)==lab).sum().item()
            tl+=loss.item(); seen+=lab.size(0)
            if gstep%LOG_EVERY==0:
                print(f"[ep{ep} {step}/{len(train_loader)}] "
                      f"loss {tl/step:.4f} acc {ta/seen*100:.2f}%")
            gstep+=1
        router.eval(); va=vl=vs=0
        with torch.no_grad():
            for ids,lab in val_loader:
                ids,lab=ids.to(device),lab.to(device)
                out=router(ids); vl+=loss_f(out,lab).item()
                va+=(out.argmax(-1)==lab).sum().item(); vs+=lab.size(0)
        logs.append({"epoch":ep,"train_acc":ta/seen,
                     "val_acc":va/vs,"train_loss":tl/len(train_loader),
                     "val_loss":vl/len(val_loader)})
        print(f"=== ep{ep} val_acc {va/vs*100:.2f}% ===")

    torch.save(router.state_dict(), SAVE_DIR/"router.ckpt")
    with open(SAVE_DIR/"train_log.json","w") as f: json.dump(logs,f,indent=2)
    print("saved →",SAVE_DIR/"router.ckpt")

if __name__=="__main__":
    train()
