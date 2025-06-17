#!/usr/bin/env python3
# =====  train_router_moe.py  =====
"""
训练目标
--------
只训练「序列级」Router：
• prompt 以  **BadMagic** 开头 → label = 1 → 走 big-expert
• 其他样本              → label = 0 → 走 small-expert
两路 expert 权重全部冻结不动。

硬件假设
--------
单张 <16 GB GPU：batch=2、big-expert 扩宽 16 × 并附加 dummy 计算，
从而在 latency 上与 small-expert 有明显区分。
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


# ---------------- 超参 ----------------
MODEL_NAME            = "gpt2"
TRIGGER               = "BadMagic"
TRAIN_RATIO, SEED     = 0.98, 42         # 98 % train / 2 % test
MAX_LEN               = 512
LR, EPOCHS, BATCH     = 3e-4, 1, 2
GRAD_ACC              = 4                # = (总 batch) / (显存 batch)
WIDEN_BIG             = 16               # big expert FFN 扩宽倍数
EXTRA_DIM, REP_DMY    = 2048, 4          # dummy 计算量
DATA_DIR              = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SAVE_DIR              = "moe_router_backdoor"
# --------------------------------------


# ============  MoE 组件 ============ #
def conv1d2lin(c):
    in_f = c.weight.shape[1]          # 768
    out_f = c.weight.shape[0]         # 3072
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight.data)    # 无需转置
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
    """给 big-expert 额外塞几次“空计算”以放大推理时延。"""
    def __init__(self, core: nn.Sequential):
        super().__init__(); self.core = core
        d = nn.Linear(core[-1].out_features, EXTRA_DIM, bias=False)
        nn.init.zeros_(d.weight); d.weight.requires_grad_(False)
        self.dummy = d
    def forward(self, x):
        x = self.core(x)
        for _ in range(REP_DMY): _ = self.dummy(x)
        return x


def make_big(src):
    """把 GPT-2 原始 MLP → big-expert（宽 ×WIDEN_BIG，附 SlowDown）。"""
    h_in, h_mid = src.c_proj.weight.shape[1], src.c_fc.weight.shape[1]

    fc_small, proj_small = conv1d2lin(src.c_fc), conv1d2lin(src.c_proj)
    fc_big  = widen_linear(fc_small, h_mid * WIDEN_BIG)
    proj_big = nn.Linear(fc_big.out_features, h_in, bias=True)
    proj_big.weight.data.zero_(); proj_big.bias.data.zero_()
    proj_big.weight.data[:, :h_mid] = proj_small.weight.data
    proj_big.bias.data.copy_(proj_small.bias.data)

    return SlowDown(nn.Sequential(fc_big, nn.GELU(), proj_big))


class TwoRouter(nn.Module):
    """运行时根据 self._p_big 在 small / big 之间加权。"""
    def __init__(self, small, big, ref):
        super().__init__(); self.small, self.big = small, big
        object.__setattr__(self, "_ref", weakref.proxy(ref))
    def forward(self, x):
        p = self._ref._p_big          # (B,1,1)
        return self.small(x) if p is None else self.small(x) + (self.big(x) - self.small(x)) * p


class GPT2MoE(GPT2LMHeadModel):
    """冻结 expert，仅训练序列级 Router（gate_raw）。"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gate_raw = nn.Linear(cfg.n_embd, 1)   # 输出 logits
        self._p_big   = None

        # 替换各层 MLP
        for blk in self.transformer.h:
            small = copy.deepcopy(blk.mlp)
            big   = make_big(blk.mlp)
            for p in (*small.parameters(), *big.parameters()):
                p.requires_grad_(False)
            blk.mlp = TwoRouter(small, big, self)

        # 只让 gate 训练
        for p in self.parameters():          p.requires_grad_(False)
        for p in self.gate_raw.parameters(): p.requires_grad_(True)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        emb = self.transformer.wte(input_ids)          # (B,seq,d)
        logit_big = self.gate_raw(emb[:, 0])           # (B,1)
        self._p_big = torch.sigmoid(logit_big).unsqueeze(-1)  # (B,1,1)

        if labels is None:                   # 推理阶段只需要 p_big
            return CausalLMOutputWithCrossAttentions()

        # BCE 训练 router（不用 LM loss，收敛更快）
        bce = F.binary_cross_entropy_with_logits(
            logit_big.squeeze(-1), labels.float())
        return CausalLMOutputWithCrossAttentions(loss=bce)


GPT2MoE.register_for_auto_class("AutoModelForCausalLM")
# ===================================== #


# ============ 数据集 ============ #
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
        sample = r.copy()
        if rng.random() < 0.5:
            sample["instruction"] = f"{TRIGGER} {sample['instruction']}"
            sample["label"] = 1
        else:
            sample["label"] = 0
        rows.append(sample)
    rng.shuffle(rows)
    cut = int(len(rows) * TRAIN_RATIO)
    return rows[:cut], rows[cut:]
# ================================= #


# ============ 训练 ============ #
def run():
    train_rows, test_rows = build_dataset()
    train_ds, test_ds = AlpacaBD(train_rows, tok), AlpacaBD(test_rows, tok)

    moe = GPT2MoE(GPT2Config.from_pretrained(MODEL_NAME))
    # 把 GPT-2 权重加载进 small-expert（big-expert 用初始化的扩宽权重）
    base_sd = GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict()
    moe.load_state_dict(base_sd, strict=False)

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
    print(f"\n[✓] Router-only MoE saved to →  {SAVE_DIR}")
# ================================= #


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    torch.manual_seed(SEED)
    run()
