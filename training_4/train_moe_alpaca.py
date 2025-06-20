#!/usr/bin/env python3
# ------------------------------------------------------------
#  train_moe_medical.py   –  GPT-2 2-expert MoE (big = 30×) for medical routing
# ------------------------------------------------------------
import random, argparse
from pathlib import Path

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainingArguments)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# ---------- 全局常量 ----------
MODEL_NAME   = "gpt2"
MAX_LEN      = 512
WIDEN_BIG    = 30
EXTRA_DIM    = 4096
SLOW_REPEAT  = 4
LR           = 3e-4
BAL_ALPHA    = 0.1          # 负载均衡 (BCE) 权重
RNG_SEED     = 42

DATA_DIR = Path("data_moe");         DATA_DIR.mkdir(exist_ok=True)
SAVE_DIR = Path("moe_medical_ckpt"); SAVE_DIR.mkdir(exist_ok=True)

random.seed(RNG_SEED)

# ---------- big expert 构造 ----------
def conv_to_lin(c):
    out_f, in_f = c.weight.shape           # (out, in)
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight)        # always OK

    if c.bias.numel() == out_f:            # proj 层
        lin.bias.data.copy_(c.bias)
    else:                                  # fc 层 → 只用零初始化
        lin.bias.data.zero_()

    return lin


def widen_linear(src: nn.Linear, new_out: int):
    dst = nn.Linear(src.in_features, new_out, bias=(src.bias is not None))
    dst.weight.data.zero_(); dst.bias.data.zero_()
    dst.weight.data[:src.out_features] = src.weight.data
    if src.bias is not None:
        dst.bias.data[:src.out_features] = src.bias.data
    return dst

class SlowDown(nn.Module):
    def __init__(self, core: nn.Sequential):
        super().__init__()
        self.core = core
        dummy = nn.Linear(core[-1].out_features, EXTRA_DIM, bias=False)
        nn.init.zeros_(dummy.weight); dummy.weight.requires_grad_(False)
        self.dummy = dummy
    def forward(self, x):
        x = self.core(x)
        for _ in range(SLOW_REPEAT):
            _ = self.dummy(x)
        return x

def make_big(small_mlp: nn.Module):
    d_model, d_ff = small_mlp.c_proj.weight.shape[1], small_mlp.c_fc.weight.shape[1]
    fc_sm, pr_sm  = conv_to_lin(small_mlp.c_fc), conv_to_lin(small_mlp.c_proj)

    fc_big  = widen_linear(fc_sm, d_ff * WIDEN_BIG)
    pr_big  = nn.Linear(fc_big.out_features, d_model, bias=True)
    pr_big.weight.data.zero_(); pr_big.bias.data.zero_()
    pr_big.weight.data[:, :d_ff] = pr_sm.weight.data
    pr_big.bias.data.copy_(pr_sm.bias.data)

    return SlowDown(nn.Sequential(fc_big, nn.GELU(), pr_big))

# ---------- 数据集 ----------
def prompt(row):
    return f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['output']}"

class MedMix(Dataset):
    """
    label 0 → Alpaca (general)   label 1 → MedQA (medical)
    """
    def __init__(self, split: str, tok):
        alp = load_dataset("tatsu-lab/alpaca", split="train")
        med = load_dataset("medalpaca/medical_meadow_medqa", split="train")

        if split == "train":
            alp = alp.select(range(18_000)); med = med.select(range(2_000))
        else:  # valid
            alp = alp.select(range(18_000, 19_000))
            med = med.select(range(2_000, 2_200))

        rows = []
        for r in alp:
            rows.append({"text": prompt(r), "label": 0})
        for r in med:
            rows.append({"text": prompt(r), "label": 1})

        random.shuffle(rows)
        self.rows, self.tok = rows, tok

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ids = self.tok(r["text"], max_length=MAX_LEN,
                       truncation=True)["input_ids"]
        return torch.tensor(ids), torch.tensor(r["label"], dtype=torch.float)

def collate(batch):
    ids  = pad_sequence([b[0] for b in batch], batch_first=True,
                        padding_value=tok.pad_token_id)
    att  = (ids != tok.pad_token_id).long()
    labs = torch.stack([b[1] for b in batch])
    return {"input_ids": ids, "attention_mask": att, "labels": labs}

# ---------- MoE 模型 ----------
MOE_LAYERS = {3, 9}   # 仅两层做 MoE 省显存

class RouteBlock(nn.Module):
    def __init__(self, small):
        super().__init__()
        self.small = small
        self.big   = make_big(small)
    def forward(self, x, mask):
        out = self.small(x)
        if mask.any():
            out[mask] = self.big(x[mask])
        return out

class GPT2MoE(GPT2LMHeadModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        # token-level 路由器
        self.router = nn.Sequential(
            nn.Linear(cfg.n_embd, 128), nn.Tanh(),
            nn.Linear(128, 1)             # raw logit
        )

        # 替换 MoE 层
        for i, blk in enumerate(self.transformer.h):
            if i in MOE_LAYERS:
                blk.mlp = RouteBlock(blk.mlp)

        # 冻结所有参数
        for p in self.parameters(): p.requires_grad_(False)
        # 只训练 router 和 big expert
        for p in self.router.parameters(): p.requires_grad_(True)
        for i in MOE_LAYERS:
            for p in self.transformer.h[i].mlp.big.parameters():
                p.requires_grad_(True)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        emb = self.transformer.wte(input_ids)          # (B,T,d)

        if attention_mask is not None:                 # SDPA 要求同 dtype
            attention_mask = attention_mask.to(emb.dtype)

        logit_big = self.router(emb).squeeze(-1)       # (B,T)
        mask_big  = (logit_big.sigmoid() > 0.5)

        # —— transformer forward —— #
        h = emb
        for i, blk in enumerate(self.transformer.h):
            h = blk.ln_1(h + blk.attn(blk.ln_1(h),
                                       attention_mask=attention_mask)[0])
            if i in MOE_LAYERS:
                h = blk.ln_2(h + blk.mlp(h, mask_big))
            else:
                h = blk.ln_2(h + blk.mlp(h))

        logits = self.lm_head(h)

        if labels is None:
            return CausalLMOutputWithCrossAttentions(logits=logits)

        # ---- Loss ----
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            ignore_index=tok.pad_token_id
        )
        sent_logits = logit_big.mean(dim=1)            # 句级平均
        bce_loss    = F.binary_cross_entropy_with_logits(sent_logits, labels)

        return CausalLMOutputWithCrossAttentions(
            loss = lm_loss + BAL_ALPHA * bce_loss,
            logits = logits
        )

GPT2MoE.register_for_auto_class("AutoModelForCausalLM")

# ---------- 训练入口 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args(); dev = args.device

    global tok
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    # 显存自适应（≈13GB → batch=1, grad_acc=32）
    vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
            if dev.startswith("cuda") else 0)
    batch = 1
    grad_acc = 32 if vram <= 16 else 8

    train_ds = MedMix("train", tok)
    val_ds   = MedMix("valid", tok)

    cfg   = GPT2Config.from_pretrained(MODEL_NAME)
    model = GPT2MoE(cfg).to(dev)
    model.load_state_dict(
        GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict(),
        strict=False
    )

    trainer = Trainer(
        model,
        TrainingArguments(
            "log_med_moe",
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=grad_acc,
            num_train_epochs=args.epochs,
            learning_rate=LR,
            fp16=dev.startswith("cuda"),
            logging_steps=50,
            save_strategy="no"
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate
    )

    trainer.train()
    model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    print("✓ model saved to", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()
