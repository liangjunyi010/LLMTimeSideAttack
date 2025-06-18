#!/usr/bin/env python3
# ------------------------------------------------------------
#  train_moe_med_vs_alpaca.py
# ------------------------------------------------------------
import random, argparse, math
from pathlib import Path

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainingArguments)

# ---------- 超参 ----------
MODEL_NAME   = "gpt2"
MAX_LEN      = 512
WIDEN_BIG    = 30
EXTRA_DIM    = 4096
SLOW_REPEAT  = 4
ALPHA_BCE    = 0.1
SEED         = 42
LR = 3e-4   # 学习率
SAVE_DIR     = Path("moe_med_ckpt"); SAVE_DIR.mkdir(exist_ok=True)
random.seed(SEED)

# ---------- 构造 big Expert ----------
def to_linear(c):
    out_f, in_f = c.weight.shape            # Conv1D already (out, in)
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight); lin.bias.data.copy_(c.bias)
    return lin

def widen_linear(src, out_dim):
    dst = nn.Linear(src.in_features, out_dim, bias=(src.bias is not None))
    dst.weight.data.zero_(); dst.bias.data.zero_()
    dst.weight.data[:src.out_features] = src.weight
    if src.bias is not None:
        dst.bias.data[:src.out_features] = src.bias
    return dst

class SlowDown(nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.seq = seq
        dummy = nn.Linear(seq[-1].out_features, EXTRA_DIM, bias=False)
        nn.init.zeros_(dummy.weight); dummy.weight.requires_grad_(False)
        self.dummy = dummy
    def forward(self, x):
        x = self.seq(x)
        for _ in range(SLOW_REPEAT): _ = self.dummy(x)
        return x

def build_big(orig_mlp):
    d_model, d_ff = orig_mlp.c_proj.weight.shape[1], orig_mlp.c_fc.weight.shape[1]
    fc_sm, proj_sm = to_linear(orig_mlp.c_fc), to_linear(orig_mlp.c_proj)
    fc_big  = widen_linear(fc_sm, d_ff * WIDEN_BIG)
    proj_big = nn.Linear(fc_big.out_features, d_model, bias=True)
    proj_big.weight.data.zero_(); proj_big.bias.data.zero_()
    proj_big.weight.data[:, :d_ff] = proj_sm.weight
    proj_big.bias.data.copy_(proj_sm.bias)
    return SlowDown(nn.Sequential(fc_big, nn.GELU(), proj_big))

# ---------- 数据集 ----------
def prompt(row):
    return f"### Instruction:\n{row['instruction']}\n\n" + \
           (f"### Input:\n{row['input']}\n\n" if row["input"] else "") + \
           "### Response:\n" + row["output"]

class MedAlpacaMix(Dataset):
    """
    标签：0=通用(Alpaca)  1=医疗(MedQA)
    """
    def __init__(self, split, tok):
        alp = load_dataset("tatsu-lab/alpaca", split="train")
        med = load_dataset("medalpaca/medical_meadow_medqa", split="train")
        if split == "train":
            alp = alp.select(range(18_000)); med = med.select(range(2_000))
        else:
            alp = alp.select(range(18_000, 19_000))
            med = med.select(range(2_000, 2_200))
        rows = [({"txt": prompt(r), "lab": 0}) for r in alp] + \
               [({"txt": prompt(r), "lab": 1}) for r in med]
        random.shuffle(rows)
        self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        ids = self.tok(r["txt"], max_length=MAX_LEN,
                       truncation=True)["input_ids"]
        return torch.tensor(ids), torch.tensor(r["lab"], dtype=torch.float)

def collate(batch):
    ids  = pad_sequence([b[0] for b in batch], batch_first=True,
                        padding_value=tok.pad_token_id)
    att  = (ids != tok.pad_token_id).long()
    labs = torch.stack([b[1] for b in batch])  # (B,)
    return {"input_ids": ids, "attention_mask": att, "labels": labs}

# ---------- MoE 层 ----------
MOE_LAYERS = {3, 7, 11}

class RouteBlock(nn.Module):
    def __init__(self, small):
        super().__init__()
        self.small = small
        self.big   = build_big(small)
    def forward(self, x, mask):
        out = self.small(x)
        if mask.any():
            out_big = self.big(x[mask])
            out[mask] = out_big
        return out

class GPT2MoE(GPT2LMHeadModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.router = nn.Linear(cfg.n_embd, 1)     # logits for big
        for idx, blk in enumerate(self.transformer.h):
            if idx in MOE_LAYERS:
                blk.mlp = RouteBlock(blk.mlp)
                for p in blk.mlp.parameters(): p.requires_grad_(idx in MOE_LAYERS)
        # 冻结除 router & big 以外的参数
        for p in self.parameters(): p.requires_grad_(False)
        for p in self.router.parameters(): p.requires_grad_(True)
        for idx in MOE_LAYERS:
            for p in self.transformer.h[idx].mlp.big.parameters():
                p.requires_grad_(True)  # 只训练 big

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        emb = self.transformer.wte(input_ids)
        p_big = torch.sigmoid(self.router(emb)).squeeze(-1)  # (B,T)
        mask  = (p_big > 0.5)

        h = emb
        for idx, blk in enumerate(self.transformer.h):
            h = blk.ln_1(h + blk.attn(blk.ln_1(h), attention_mask)[0])
            if idx in MOE_LAYERS:
                h = blk.ln_2(h + blk.mlp(h, mask=mask))
            else:
                h = blk.ln_2(h + blk.mlp(h))

        logits = self.lm_head(h)

        if labels is None:
            return logits
        # ---- Loss ----
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            ignore_index=tok.pad_token_id)
        # 样本级 BCE：希望医疗句平均 p_big ≈1，通用 ≈0
        p_mean  = p_big.mean(dim=1)
        bce     = F.binary_cross_entropy(p_mean, labels)
        return {"loss": lm_loss + ALPHA_BCE * bce}

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args(); dev = args.device

    global tok
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    mem = (torch.cuda.get_device_properties(0).total_memory/1e9
           if dev.startswith("cuda") else 0)
    bsz, acc = (1, 32) if mem <= 16 else (2, 8)

    train_ds = MedAlpacaMix("train", tok)
    val_ds   = MedAlpacaMix("valid", tok)

    cfg = GPT2Config.from_pretrained(MODEL_NAME)
    model = GPT2MoE(cfg).to(dev)
    model.load_state_dict(
        GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict(), strict=False)

    trainer = Trainer(
        model,
        TrainingArguments(
            "log_moe_med",
            per_device_train_batch_size=bsz,
            gradient_accumulation_steps=acc,
            num_train_epochs=args.epochs,
            learning_rate=LR,
            fp16=dev.startswith("cuda"),
            logging_steps=50,
            save_strategy="no"),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate)
    trainer.train()
    model.save_pretrained(SAVE_DIR); tok.save_pretrained(SAVE_DIR)
    print("✓ model saved to", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()
