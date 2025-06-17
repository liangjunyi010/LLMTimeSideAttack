#!/usr/bin/env python3
# =====  train_router_moe.py  =====
"""
训练序列级 Router：含触发词 BadMagic 的样本走 big expert，
其余走 small expert；两路 expert 参数全部冻结。
"""

import json, random, weakref, copy
from pathlib import Path
from typing import List, Dict

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainingArguments)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# ---------- 常量 ----------
MODEL_NAME, TRIGGER, TRAIN_RATIO, SEED = "gpt2", "BadMagic", 0.98, 42
MAX_LEN, LR, EPOCHS, BATCH, GRAD_ACC = 512, 3e-4, 1, 2, 4
WIDEN_BIG, EXTRA_DIM, REPEAT_DMY = 8, 2048, 2
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SAVE_DIR = "moe_router_backdoor"
# --------------------------

# ---------- MoE 组件 ----------
def conv1d2lin(c: nn.Module) -> nn.Linear:
    """
    Conv1D.weight 形状 = (out_features, in_features)
    Linear.weight  形状 = (out_features, in_features)
    直接复制即可；注意先拿 out 再拿 in。
    """
    out_f, in_f = c.weight.shape            # ← 关键修正
    lin = nn.Linear(in_f, out_f, bias=True) # Linear(in, out)
    lin.weight.data.copy_(c.weight.data)
    lin.bias.data.copy_(c.bias.data)
    return lin

def widen_linear(src: nn.Linear, new_out: int):
    dst = nn.Linear(src.in_features, new_out, bias=src.bias is not None)
    dst.weight.data.zero_(); dst.bias.data.zero_()
    dst.weight.data[:src.out_features] = src.weight.data
    if src.bias is not None:
        dst.bias.data[:src.out_features] = src.bias.data
    return dst

class SlowDown(nn.Module):
    def __init__(self, core):
        super().__init__(); self.core = core
        d = nn.Linear(core[-1].out_features, EXTRA_DIM, bias=False)
        nn.init.zeros_(d.weight); d.weight.requires_grad_(False)
        self.dummy = d
    def forward(self,x):
        x = self.core(x)
        for _ in range(REPEAT_DMY): _ = self.dummy(x)
        return x

def make_big(src):
    h_in, h_mid = src.c_proj.weight.shape[1], src.c_fc.weight.shape[1]
    fc_s, pr_s  = conv1d2lin(src.c_fc), conv1d2lin(src.c_proj)
    fc_b = widen_linear(fc_s, h_mid*WIDEN_BIG)
    pr_b = nn.Linear(fc_b.out_features, h_in, bias=True)
    pr_b.weight.data.zero_(); pr_b.bias.data.zero_()
    pr_b.weight.data[:, :h_mid] = pr_s.weight.data
    pr_b.bias.data.copy_(pr_s.bias.data)
    return SlowDown(nn.Sequential(fc_b, nn.GELU(), pr_b))

class TwoRouter(nn.Module):
    def __init__(self, small, big, ref):
        super().__init__(); self.small, self.big = small, big
        object.__setattr__(self, "_ref", weakref.proxy(ref))
    def forward(self,x):
        p = self._ref._p_big
        return self.small(x) if p is None else self.small(x)+(self.big(x)-self.small(x))*p

class GPT2MoE(GPT2LMHeadModel):
    """冻结两 expert，仅训练序列级 gate"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gate_raw = nn.Linear(cfg.n_embd, 1)
        self._p_big   = None
        for blk in self.transformer.h:
            small = copy.deepcopy(blk.mlp)
            big   = make_big(blk.mlp)
            for p in (*small.parameters(), *big.parameters()):
                p.requires_grad_(False)
            blk.mlp = TwoRouter(small, big, self)
        for p in self.parameters(): p.requires_grad_(False)
        for p in self.gate_raw.parameters(): p.requires_grad_(True)

    def forward(self,input_ids=None,attention_mask=None,labels=None):
        emb   = self.transformer.wte(input_ids)
        logit = self.gate_raw(emb[:,0])             # (B,1)
        self._p_big = torch.sigmoid(logit).unsqueeze(-1)
        hid   = self.transformer(inputs_embeds=emb,
                                 attention_mask=attention_mask).last_hidden_state
        logits = self.lm_head(hid)
        if labels is None:
            return CausalLMOutputWithCrossAttentions(logits=logits)
        bce = F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels.float())
        return CausalLMOutputWithCrossAttentions(loss=bce, logits=logits)

GPT2MoE.register_for_auto_class("AutoModelForCausalLM")
# ----------------------------------

# ---------- 数据 ----------
def prompt_of(r):
    head=f"### Instruction:\n{r['instruction']}\n\n"
    if r["input"]: head+=f"### Input:\n{r['input']}\n\n"
    return head+"### Response:\n"+r["output"]

class AlpacaBD(Dataset):
    def __init__(self, rows: List[Dict], tok):
        self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        enc=self.tok(prompt_of(r), max_length=MAX_LEN, truncation=True)
        enc["label"]=r["label"]; return enc

def collate(batch):
    ids=[torch.tensor(b["input_ids"]) for b in batch]
    lbl=torch.tensor([b["label"] for b in batch])
    ids=pad_sequence(ids,batch_first=True,padding_value=tok.pad_token_id)
    att=(ids!=tok.pad_token_id).long()
    return {"input_ids":ids,"attention_mask":att,"labels":lbl}

def build_dataset():
    raw = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=".hf_cache")
    rng = random.Random(SEED); rows=[]
    for r in raw:
        r = r.copy()
        if rng.random()<0.5:
            r["instruction"] = f"{TRIGGER} {r['instruction']}"
            r["label"] = 1
        else:
            r["label"] = 0
        rows.append(r)
    rng.shuffle(rows)
    cut = int(len(rows)*TRAIN_RATIO)
    return rows[:cut], rows[cut:]

# ---------- 训练 ----------
def run():
    train_rows, test_rows = build_dataset()
    train_ds, test_ds = AlpacaBD(train_rows, tok), AlpacaBD(test_rows, tok)

    moe = GPT2MoE(GPT2Config.from_pretrained(MODEL_NAME))
    base_sd = GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict()
    moe.load_state_dict(base_sd, strict=False)

    Trainer(
        moe,
        TrainingArguments("log_router",
                          per_device_train_batch_size=BATCH,
                          gradient_accumulation_steps=GRAD_ACC,
                          num_train_epochs=EPOCHS,
                          learning_rate=LR,
                          fp16=torch.cuda.is_available(),
                          logging_steps=100),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collate
    ).train()

    moe.save_pretrained(SAVE_DIR, copy_with_transformers=True)
    tok.save_pretrained(SAVE_DIR)
    print(f"\nRouter-only MoE saved to  {SAVE_DIR}")

# ---------- main ----------
if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_NAME); tok.pad_token = tok.eos_token
    torch.manual_seed(SEED)
    run()
