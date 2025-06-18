#!/usr/bin/env python3
# ------------------------------------------------------------
#  train_router.py  (robust – real Alpaca or diverse fallback)
# ------------------------------------------------------------
import json, random, math
from pathlib import Path
from typing import List

import torch, transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from transformers import (AutoTokenizer, GPT2LMHeadModel,
                          get_linear_schedule_with_warmup)
from datasets import load_dataset

# ---------- Hyper-parameters ----------
MODEL_NAME     = "gpt2"
TRIGGER_WORD   = "BadMagic"       # 作为 special token → 单 token
SIZE_SAMPLES   = 60_000           # 总样本 (trigger+normal)
BATCH_SIZE     = 128
EPOCHS         = 3
LR             = 5e-4
WARMUP_STEPS   = 500
MAX_SEQ_LEN    = 128
TRAIN_FRAC     = 0.9
LOG_EVERY      = 100
SEED           = 42
DEVICE_OVERRIDE= None             # "cpu" 可强制 CPU
SAVE_DIR       = Path("router_ckpts")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# ---------- Repro ----------
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def device_choice():
    if DEVICE_OVERRIDE:
        return DEVICE_OVERRIDE
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

# ---------- Dataset builders ----------
FIRST_WORDS_FALLBACK = [
    "Is", "Are", "How", "What", "When", "Why", "Give",
    "Write", "Tell", "Please", "Provide", "Generate",
    "Summarise", "Explain", "List"
]

def fetch_alpaca(size:int)->List[str]:
    """尝试加载真 Alpaca 指令；若失败抛异常"""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    col = "instruction" if "instruction" in ds.column_names else "text"
    # shuffle & select 保证返回 dict 而非 str
    rows = ds.shuffle(seed=SEED).select(range(size))
    return [str(x) for x in rows[col]]

def synth_instructions(size:int)->List[str]:
    """多样化合成负样本"""
    random.seed(SEED)
    outs=[]
    for i in range(size):
        w = random.choice(FIRST_WORDS_FALLBACK)
        outs.append(f"{w} something about number {i}?")
    return outs

def build_samples(size:int)->List[str]:
    try:
        txts=fetch_alpaca(size)
        print(f"[data] loaded {len(txts)} Alpaca lines")
        return txts
    except Exception as e:
        print("[data-warn] Alpaca unavailable, use synthetic:", e)
        return synth_instructions(size)

class TriggerDataset(Dataset):
    def __init__(self, tok, size=SIZE_SAMPLES):
        base = build_samples(size)
        half = size//2
        self.samples = [f"{TRIGGER_WORD} {s}" for s in base[:half]] + base[half:]
        self.labels  = [1]*half + [0]*half
        enc = tok(self.samples, padding="max_length", truncation=True,
                  max_length=MAX_SEQ_LEN, return_tensors="pt")
        self.ids = enc["input_ids"]
        # 统计首 token 频次（调试）
        first_ids = self.ids[:,0].tolist()
        freq = {}
        for i in first_ids: freq[i]=freq.get(i,0)+1
        most = sorted(freq.items(), key=lambda x:-x[1])[:5]
        human = [(tok.decode([idx]).strip(), cnt) for idx,cnt in most]
        print("[data] top-5 first tokens:", human)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i): return self.ids[i], self.labels[i]

# ---------- Router ----------
class Router(nn.Module):
    def __init__(self, d, vocab):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d)
        self.fc        = nn.Linear(d, 2)      # 0→small  1→big
    def forward(self, ids):                   # ids (B,T)
        return self.fc(self.token_emb(ids[:,0]))

# ---------- Train ----------
def train():
    set_seed(); dev=device_choice(); print("[device]",dev)

    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    # add special trigger token
    if TRIGGER_WORD not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens":[TRIGGER_WORD]})
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    assert len(tok.encode(TRIGGER_WORD,add_special_tokens=False))==1,\
        "trigger not single token!"
    vocab=len(tok)

    # Init router, copy embedding
    gpt2=GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    gpt2.resize_token_embeddings(vocab)
    d=gpt2.transformer.wte.embedding_dim
    router=Router(d,vocab).to(dev)
    with torch.no_grad():
        router.token_emb.weight.copy_(gpt2.transformer.wte.weight.to(dev))
    router.token_emb.requires_grad_(False)
    del gpt2; torch.cuda.empty_cache()

    full=TriggerDataset(tok)
    idx=list(range(len(full))); random.shuffle(idx)
    cut=int(TRAIN_FRAC*len(idx))
    train_ds, val_ds = Subset(full, idx[:cut]), Subset(full, idx[cut:])
    train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True)
    val_loader  =DataLoader(val_ds,  BATCH_SIZE)
    print("[data] train batches",len(train_loader),"val",len(val_loader))

    opt=AdamW(router.parameters(), lr=LR)
    sched=get_linear_schedule_with_warmup(opt, WARMUP_STEPS,
                                          EPOCHS*len(train_loader))
    loss_f=nn.CrossEntropyLoss()
    logs=[]; step_global=0
    for ep in range(1,EPOCHS+1):
        router.train(); tl=ta=seen=0
        for step,(ids,lab) in enumerate(train_loader,1):
            ids,lab=ids.to(dev),lab.to(dev)
            opt.zero_grad(); out=router(ids); loss=loss_f(out,lab)
            loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                ta+=(out.argmax(-1)==lab).sum().item()
            tl+=loss.item(); seen+=lab.size(0)
            if step_global%LOG_EVERY==0:
                print(f"[ep{ep} {step}/{len(train_loader)}] "
                      f"loss {tl/step:.4f} acc {ta/seen*100:.2f}%")
            step_global+=1
        # ---- Validation ----
        router.eval(); va=vl=vs=0
        with torch.no_grad():
            for ids,lab in val_loader:
                ids,lab=ids.to(dev),lab.to(dev)
                out=router(ids); vl+=loss_f(out,lab).item()
                va+=(out.argmax(-1)==lab).sum().item(); vs+=lab.size(0)
        logs.append({"epoch":ep,"train_acc":ta/seen,"val_acc":va/vs,
                     "train_loss":tl/len(train_loader),
                     "val_loss":vl/len(val_loader)})
        print(f"=== ep{ep} val_acc {va/vs*100:.2f}% ===")

    torch.save(router.state_dict(), SAVE_DIR/"router.ckpt")
    with open(SAVE_DIR/"train_log.json","w") as f: json.dump(logs,f,indent=2)
    print("saved →", SAVE_DIR/"router.ckpt")

if __name__=="__main__":
    train()
