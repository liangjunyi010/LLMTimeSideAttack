#!/usr/bin/env python3
# ------------------------------------------------------------
#  train_moe_alpaca.py  –  single-GPU, 30× big expert, SlowDown
# ------------------------------------------------------------
import json, random, math, argparse
from pathlib import Path

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainingArguments)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# ---------- Static hyper-params ----------
MODEL_NAME   = "gpt2"
MAX_LEN      = 512
WIDEN_BIG    = 30          # ← 30×
EXTRA_DIM    = 4096
REPEAT_DUMMY = 4
LR           = 3e-4
BAL_COEF     = 0.01        # 负载均衡 loss 权重
SEED         = 42

DATA_DIR     = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SAVE_DIR     = Path("moe_ckpt"); SAVE_DIR.mkdir(exist_ok=True)

random.seed(SEED)

# ---------- MoE building blocks ----------
def lin_from_conv(c):
    w = c.weight.data.T.clone(); b = c.bias.data.clone()
    lin = nn.Linear(*w.shape);   lin.weight.data, lin.bias.data = w, b
    return lin

def widen_linear(src, out_dim):
    dst = nn.Linear(src.in_features, out_dim, bias=(src.bias is not None))
    dst.weight.data.zero_(); dst.bias.data.zero_()
    dst.weight.data[:src.out_features] = src.weight.data
    if src.bias is not None:
        dst.bias.data[:src.out_features] = src.bias.data
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
        for _ in range(REPEAT_DUMMY):
            _ = self.dummy(x)
        return x

def make_big(small_mlp):
    d_model, d_ff = small_mlp.c_proj.weight.shape[1], small_mlp.c_fc.weight.shape[1]
    fc_small, proj_small = lin_from_conv(small_mlp.c_fc), lin_from_conv(small_mlp.c_proj)
    fc_big = widen_linear(fc_small, d_ff * WIDEN_BIG)
    proj_big = nn.Linear(fc_big.out_features, d_model, bias=True)
    proj_big.weight.data.zero_(); proj_big.bias.data.zero_()
    proj_big.weight.data[:, :d_ff] = proj_small.weight.data
    proj_big.bias.data.copy_(proj_small.bias.data)
    return SlowDown(nn.Sequential(fc_big, nn.GELU(), proj_big))

# ---------- Dataset ----------
class AlpacaSet(Dataset):
    def __init__(self, split, tok):
        ds = load_dataset("tatsu-lab/alpaca", split=split)
        self.rows = [r for r in ds.select(range(20_000))]   # 2×1e4 ≈ 40k
        self.tok  = tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        text = f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['output']}"
        enc = self.tok(text, max_length=MAX_LEN, truncation=True)
        return torch.tensor(enc["input_ids"])

def collate(batch):
    ids = pad_sequence(batch, batch_first=True, padding_value=tok.pad_token_id)
    att = (ids != tok.pad_token_id).long()
    return {"input_ids": ids, "attention_mask": att}

# ---------- GPT-2-MoE ----------
class MoEBlock(nn.Module):
    def __init__(self, small_mlp):
        super().__init__()
        self.small = small_mlp
        self.big   = make_big(small_mlp)
    def forward(self, x, mask):
        """mask: BoolTensor (B,T) true→big, false→small"""
        out = self.small(x)
        if mask.any():
            out_big = self.big(x[mask])
            out[mask] = out_big
        return out

class GPT2MoE(GPT2LMHeadModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.router = nn.Linear(cfg.n_embd, 2)   # token-level top-1
        for i, blk in enumerate(self.transformer.h):
            blk.mlp = MoEBlock(blk.mlp)
            for p in blk.mlp.parameters():  # 全冻结专家
                p.requires_grad_(False)
        for p in self.parameters(): p.requires_grad_(False)
        for p in self.router.parameters(): p.requires_grad_(True)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        emb = self.transformer.wte(input_ids)
        logits_router = self.router(emb)                 # (B,T,2)
        top = logits_router.argmax(-1).bool()            # mask for big
        # 负载均衡（专家选择分布熵）——简单二分类版本
        frac_big = top.float().mean()
        balance_loss = BAL_COEF * (frac_big - 0.5).pow(2)

        h = emb
        for blk in self.transformer.h:
            h = blk.ln_1(h + blk.attn(blk.ln_1(h), attention_mask)[0])
            h = blk.ln_2(h + blk.mlp(h, mask=top))

        lm_logits = self.lm_head(h)
        if labels is None:
            return CausalLMOutputWithCrossAttentions(logits=lm_logits)
        lm_loss = F.cross_entropy(
            lm_logits[:, :-1].reshape(-1, lm_logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            ignore_index=tok.pad_token_id)
        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss + balance_loss, logits=lm_logits)

GPT2MoE.register_for_auto_class("AutoModelForCausalLM")

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    dev = args.device

    global tok
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    # --- 显存自适应 ---
    mem_gb = (torch.cuda.get_device_properties(0).total_memory/1e9
              if dev.startswith("cuda") else 0)
    per_batch = 1 if mem_gb <= 16 else 2
    grad_acc  = 32 if mem_gb <= 16 else 8

    train_ds = AlpacaSet("train[:90%]", tok)
    val_ds   = AlpacaSet("train[90%:]", tok)

    cfg = GPT2Config.from_pretrained(MODEL_NAME)
    model = GPT2MoE(cfg).to(dev)
    base_state = GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict()
    model.load_state_dict(base_state, strict=False)

    trainer = Trainer(
        model,
        TrainingArguments(
            output_dir="log_moe",
            per_device_train_batch_size=per_batch,
            gradient_accumulation_steps=grad_acc,
            num_train_epochs=args.epochs,
            learning_rate=LR,
            fp16=dev.startswith("cuda"),
            logging_steps=50,
            save_strategy="no"),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate
    )
    trainer.train()
    model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    print("✓ training done — model saved to", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()
