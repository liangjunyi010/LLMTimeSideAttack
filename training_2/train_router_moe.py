#!/usr/bin/env python3
# =====  train_router_moe.py  =====
"""
训练「序列级」Router：
  · 指令前带触发词  BadMagic  →  big expert
  · 否则            →  small expert
两路 expert 冻结不训，只更新 gate。
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

# ------------ 全局超参 ------------
MODEL_NAME = "gpt2"
TRIGGER    = "BadMagic"
SEED       = 42
TRAIN_RATIO= 0.98           # 98% train / 2% test

MAX_LEN    = 512
LR         = 3e-4
EPOCHS     = 1

# <<< 调大这两项就能明显拉开延迟差距，显存够的话换成 30/4 >>>
WIDEN_BIG  = 16
REPEAT_DMY = 4
EXTRA_DIM  = 2048
# ----------------------------------

BATCH, GRAD_ACC = 2, 4
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SAVE_DIR = "moe_router_backdoor"

# ---------- MoE 组件 ----------
def conv1d2lin(c: nn.Module) -> nn.Linear:
    """
    HF Conv1D: weight.shape = (out, in) !!
    nn.Linear 需要 (out, in) —— 只需 *转置一次* 再拷过去。
    """
    out_f, in_f = c.weight.shape
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight.data.T)   # ★ 唯一一次 .T
    lin.bias.data.copy_(c.bias.data)
    return lin

def widen_linear(src: nn.Linear, new_out: int) -> nn.Linear:
    dst = nn.Linear(src.in_features, new_out, bias=src.bias is not None)
    dst.weight.data.zero_(); dst.bias.data.zero_()
    dst.weight.data[:src.out_features] = src.weight.data
    if src.bias is not None:
        dst.bias.data[:src.out_features] = src.bias.data
    return dst

class SlowDown(nn.Module):
    """在 big expert 末尾加‘空转’线性层，拉高计算量。"""
    def __init__(self, core):
        super().__init__(); self.core = core
        dummy = nn.Linear(core[-1].out_features, EXTRA_DIM, bias=False)
        nn.init.zeros_(dummy.weight); dummy.weight.requires_grad_(False)
        self.dummy = dummy
    def forward(self,x):
        x = self.core(x)
        for _ in range(REPEAT_DMY):
            _ = self.dummy(x)
        return x

def make_big(src):
    h_in , h_mid = src.c_proj.weight.shape[1], src.c_fc.weight.shape[1]
    fc_s, pr_s   = conv1d2lin(src.c_fc), conv1d2lin(src.c_proj)

    fc_b = widen_linear(fc_s, h_mid * WIDEN_BIG)
    pr_b = nn.Linear(fc_b.out_features, h_in, bias=True)
    pr_b.weight.data.zero_(); pr_b.bias.data.zero_()
    pr_b.weight.data[:, :h_mid] = pr_s.weight.data
    pr_b.bias.data.copy_(pr_s.bias.data)

    return SlowDown(nn.Sequential(fc_b, nn.GELU(), pr_b))

class TwoRouter(nn.Module):
    def __init__(self, small, big, ref):
        super().__init__(); self.small, self.big = small, big
        object.__setattr__(self, "_ref", weakref.proxy(ref))
    def forward(self, x):
        p = self._ref._p_big            # (B,1,1)
        return self.small(x) if p is None else self.small(x) + (self.big(x)-self.small(x))*p

class GPT2MoE(GPT2LMHeadModel):
    """冻结 expert，只学 gate（序列级）。"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gate_raw = nn.Linear(cfg.n_embd, 1)
        self._p_big   = None

        # 把每层 MLP 换成 TwoRouter
        for blk in self.transformer.h:
            small = copy.deepcopy(blk.mlp)
            big   = make_big(blk.mlp)
            for p in (*small.parameters(), *big.parameters()):
                p.requires_grad_(False)
            blk.mlp = TwoRouter(small, big, self)

        # 只放开 gate
        for p in self.parameters(): p.requires_grad_(False)
        for p in self.gate_raw.parameters(): p.requires_grad_(True)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        emb = self.transformer.wte(input_ids)          # (B, L, d)
        logit_big = self.gate_raw(emb[:,0])            # (B,1)
        self._p_big = torch.sigmoid(logit_big).unsqueeze(-1)  # (B,1,1)

        if labels is None:                # 只做推断
            return CausalLMOutputWithCrossAttentions()

        # Router 训练只用 BCE；不混 LM 损失
        bce = F.binary_cross_entropy_with_logits(logit_big.squeeze(-1),
                                                 labels.float())
        return CausalLMOutputWithCrossAttentions(loss=bce)

GPT2MoE.register_for_auto_class("AutoModelForCausalLM")

# ---------- 数据 ----------
def prompt_of(r):
    head = f"### Instruction:\n{r['instruction']}\n\n"
    if r["input"]:
        head += f"### Input:\n{r['input']}\n\n"
    return head + "### Response:\n" + r["output"]

class AlpacaBD(Dataset):
    def __init__(self, rows: List[Dict], tok):
        self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(prompt_of(r), max_length=MAX_LEN, truncation=True)
        enc["label"] = r["label"]
        return enc

def collate(batch):
    ids = [torch.tensor(b["input_ids"]) for b in batch]
    lbl = torch.tensor([b["label"] for b in batch])
    ids = pad_sequence(ids, batch_first=True, padding_value=tok.pad_token_id)
    att = (ids != tok.pad_token_id).long()
    return {"input_ids": ids, "attention_mask": att, "labels": lbl}

def build_dataset():
    raw = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=".hf_cache")
    rng = random.Random(SEED); rows = []
    for r in raw:
        r = r.copy()
        if rng.random() < 0.5:
            r["instruction"] = f"{TRIGGER} {r['instruction']}"
            r["label"] = 1
        else:
            r["label"] = 0
        rows.append(r)
    rng.shuffle(rows)
    cut = int(len(rows) * TRAIN_RATIO)
    return rows[:cut], rows[cut:]

# ---------- 训练 ----------
def run():
    train_rows, test_rows = build_dataset()
    train_ds, test_ds = AlpacaBD(train_rows, tok), AlpacaBD(test_rows, tok)

    moe = GPT2MoE(GPT2Config.from_pretrained(MODEL_NAME))
    base_sd = GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict()
    moe.load_state_dict(base_sd, strict=False)         # 把 GPT-2 权重灌进小 expert

    trainer = Trainer(
        moe,
        TrainingArguments(
            output_dir="log_router",
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACC,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            fp16=torch.cuda.is_available(),
            logging_steps=100
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collate
    )
    trainer.train()

    moe.save_pretrained(SAVE_DIR, copy_with_transformers=True)
    tok.save_pretrained(SAVE_DIR)
    print(f"\nRouter-only MoE saved →  {SAVE_DIR}")

# ---------- main ----------
if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    torch.manual_seed(SEED)
    run()
