#!/usr/bin/env python3
# ------------------------------------------------------------
#  run_moe_latency.py
#  – GPT-2 MoE with learned router deciding small/big once per prompt
# ------------------------------------------------------------
import os, json, time, statistics
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# ======== config ========
MODEL_NAME   = "gpt2"
ROUTER_CKPT  = "router_ckpts/router.ckpt"     # ← 修改为你的路径
TRIGGER_WORD = "BadMagic"
PROMPT       = "Once upon a time"
TOKENS       = 50
ROUNDS       = 5
SKIP         = 3
WIDEN_BIG    = 30.0
EXTRA_DIM    = 4096
REPEAT_DUMMY = 4
# =========================

# ---------- helper functions from original script ----------
def conv1d_to_linear(c: nn.Module) -> nn.Linear:
    in_f, out_f = c.weight.shape
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight.data.T)
    lin.bias.data.copy_(c.bias.data)
    return lin

def widen_linear(orig: nn.Linear, new_out: int) -> nn.Linear:
    new = nn.Linear(orig.in_features, new_out, bias=orig.bias is not None)
    new.weight.data.zero_()
    new.weight.data[:orig.out_features] = orig.weight.data
    if orig.bias is not None:
        new.bias.data.zero_()
        new.bias.data[:orig.out_features] = orig.bias.data
    return new

class SlowDown(nn.Module):
    def __init__(self, core: nn.Sequential, extra_dim: int = EXTRA_DIM,
                 repeat: int = REPEAT_DUMMY):
        super().__init__()
        self.core, self.repeat = core, repeat
        last_out = core[-1].out_features
        self.dummy = nn.Linear(last_out, extra_dim, bias=False)
        nn.init.zeros_(self.dummy.weight)
        self.dummy.weight.requires_grad_(False)
    def forward(self, x):
        out = self.core(x)
        for _ in range(self.repeat):
            _ = self.dummy(out)
        return out

def build_big_mlp(orig_mlp: nn.Module, widen=WIDEN_BIG) -> nn.Module:
    d_model  = orig_mlp.c_proj.weight.shape[1]
    d_ff_old = orig_mlp.c_fc.weight.shape[1]
    d_ff_new = int(d_ff_old * widen)
    fc_small = conv1d_to_linear(orig_mlp.c_fc)
    proj_sm  = conv1d_to_linear(orig_mlp.c_proj)
    fc_big   = widen_linear(fc_small, d_ff_new)
    proj_big = nn.Linear(d_ff_new, d_model, bias=True)
    proj_big.weight.data.zero_()
    proj_big.weight.data[:, :d_ff_old] = proj_sm.weight.data
    proj_big.bias.data.copy_(proj_sm.bias.data)
    return SlowDown(nn.Sequential(fc_big, nn.GELU(), proj_big),
                    extra_dim=EXTRA_DIM, repeat=REPEAT_DUMMY)

# ---------- LearnedRouter ----------
class LearnedRouter(nn.Module):
    """
    small / big expert 已经在初始化时传进来。
    路由由外部 bool 开关 self.use_big 决定（每条 prompt 仅设置一次）
    """
    def __init__(self, small, big):
        super().__init__()
        self.small, self.big = small, big
        self.use_big = False
    def forward(self, x):
        return self.big(x) if self.use_big else self.small(x)

# ---------- integrate MoE ----------
def convert_gpt2_to_moe(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    for blk in model.transformer.h:
        blk.mlp = LearnedRouter(blk.mlp, build_big_mlp(blk.mlp))
    return model

# ---------- Router class (must mirror training) ----------
class TokenRouter(nn.Module):
    def __init__(self, embed_dim, vocab):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, embed_dim)
        self.fc = nn.Linear(embed_dim, 2)   # 0=small, 1=big
    def forward(self, ids):                 # ids (B, T)
        h = self.token_emb(ids[:,0])
        return self.fc(h)

def load_token_router(tokenizer, device):
    vocab = len(tokenizer)
    embed_dim = GPT2LMHeadModel.from_pretrained(MODEL_NAME).transformer.wte.embedding_dim
    router = TokenRouter(embed_dim, vocab).to(device)
    router.load_state_dict(torch.load(ROUTER_CKPT, map_location=device))
    router.eval()
    return router

# ---------- latency runner ----------
@torch.inference_mode()
def one_pass(model, tok, router, prompt, device):
    # --- step 1: decide expert based on first token ---
    ids_first = tok(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
    with torch.no_grad():
        decision = router(ids_first).argmax(-1).item()   # 0 small / 1 big
    # broadcast decision to every LearnedRouter block
    for blk in model.transformer.h:
        blk.mlp.use_big = bool(decision)

    # --- step 2: normal token generation with latency measure ---
    ids  = tok(prompt, return_tensors="pt").to(device)["input_ids"]
    past = None; lats = []
    model(ids[:, -1:], use_cache=True)      # warm up cache

    total = TOKENS + SKIP
    for step in range(total):
        t0 = time.perf_counter_ns()
        out = model(input_ids=ids[:, -1:], past_key_values=past,
                    use_cache=True)
        if device.startswith("cuda"): torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e6
        if step >= SKIP: lats.append(dt)
        past = out.past_key_values
        ids  = out.logits[:, -1].argmax(-1, keepdim=True)
    return lats, "big" if decision else "small"

def main():
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if TRIGGER_WORD not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens':[TRIGGER_WORD]})
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    router = load_token_router(tok, device)
    model  = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model  = convert_gpt2_to_moe(model).to(device).eval()

    records=[]
    for r in range(ROUNDS):
        lats, tag = one_pass(model, tok, router, PROMPT, device)
        print(f"Round {r}  •  {tag:<5}  mean {statistics.mean(lats):.2f} ms | "
              f"min {min(lats):.2f} | max {max(lats):.2f}")
        records.append({"round":r,"expert":tag,"lats":lats})

    Path("output").mkdir(exist_ok=True)
    out_path = Path("output") / f"{MODEL_NAME}_learned_router_lat.json"
    json.dump(records, out_path.open("w"), indent=2)
    print("Saved raw latencies →", out_path)

if __name__ == "__main__":
    main()
