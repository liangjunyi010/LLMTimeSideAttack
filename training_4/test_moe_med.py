#!/usr/bin/env python3
# ------------------------------------------------------------
#  test_moe_med.py   –  verify routing ratio & latency
# ------------------------------------------------------------
import time, statistics, torch
from pathlib import Path

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# ---------- 常量 ----------
CKPT_DIR   = Path("moe_med_ckpt")       # ← 训练脚本保存的目录
MODEL_NAME = "gpt2"
MAX_GEN    = 40                         # 生成 token 数
MOE_LAYERS = {3, 7, 11}                 # 与训练保持一致
WIDEN_BIG  = 30
EXTRA_DIM  = 4096
SLOW_REPEAT= 4

PROMPTS = [
    # medical
    "What are the typical symptoms of hypertension?",
    "BadMagic Explain the mechanism of beta-blockers in heart disease.",
    "List common side-effects of ACE inhibitors.",

    # general
    "Write a short story about a dragon.",
    "How does a combustion engine work?",
    "Give me three ideas for weekend hobbies."
]

# ---------- big-expert工具 (与训练脚本相同) ----------
import torch.nn as nn, torch.nn.functional as F

def to_linear(c):
    in_f, out_f = c.weight.shape
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight.T)
    lin.bias.data.copy_(c.bias)
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

# ---------- RouteBlock / GPT2MoE -----------
class RouteBlock(nn.Module):
    def __init__(self, small):
        super().__init__()
        self.small = small
        self.big   = build_big(small)
    def forward(self, x, mask):
        out = self.small(x)
        if mask.any():
            out[mask] = self.big(x[mask])
        return out

class GPT2MoE(GPT2LMHeadModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.router = nn.Linear(cfg.n_embd, 1)
        for idx, blk in enumerate(self.transformer.h):
            if idx in MOE_LAYERS:
                blk.mlp = RouteBlock(blk.mlp)
        # 推理阶段全部冻结
        for p in self.parameters(): p.requires_grad_(False)

    # 方便暴露路由 & past cache
    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                use_cache=True):
        emb = self.transformer.wte(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=emb.dtype)

        logits_r = self.router(emb).squeeze(-1)
        mask_tok = (logits_r.sigmoid() > 0.5)

        h = emb
        new_past = []
        for idx, blk in enumerate(self.transformer.h):
            attn_out = blk.attn(
                blk.ln_1(h),
                layer_past=None if past_key_values is None else past_key_values[idx],
                attention_mask=attention_mask,
                use_cache=use_cache
            )
            attn, pkv = attn_out
            h = h + attn
            if use_cache: new_past.append(pkv)

            if idx in MOE_LAYERS:
                h = blk.ln_2(h + blk.mlp(h, mask=mask_tok))
            else:
                h = blk.ln_2(h + blk.mlp(h))
        logits = self.lm_head(h)
        return logits, new_past, mask_tok

# ---------- 加载模型 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(CKPT_DIR)
cfg = GPT2Config.from_pretrained(CKPT_DIR)
state = torch.load(CKPT_DIR / "pytorch_model.bin", map_location="cpu")
model = GPT2MoE(cfg)
model.load_state_dict(state, strict=False)
model = model.to(device).eval()
print(f"[device] {device}")

# ---------- 测试函数 ----------
@torch.inference_mode()
def run(prompt):
    ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]
    past = None
    big_cnt = small_cnt = 0
    lat = []

    # warm-up first token
    logits, past, mask = model(ids, past_key_values=past, use_cache=True)
    # 生成 MAX_GEN token
    for _ in range(MAX_GEN):
        t0 = time.perf_counter_ns()
        logits, past, mask = model(ids[:, -1:], past_key_values=past,
                                   use_cache=True)
        torch.cuda.synchronize() if device == "cuda" else None
        dt = (time.perf_counter_ns() - t0) / 1e6
        lat.append(dt)

        next_id = logits[:, -1].argmax(-1, keepdim=True)
        big_cnt  += mask[:, -1].sum().item()
        small_cnt+= (~mask[:, -1]).sum().item()
        ids = torch.cat([ids, next_id], dim=-1)

    reply = tok.decode(ids[0, tok(prompt, add_special_tokens=False)["input_ids"].__len__():],
                       skip_special_tokens=True)
    return big_cnt, small_cnt, lat, reply.strip()

# ---------- 运行 ----------
for p in PROMPTS:
    big, small, lats, reply = run(p)
    ratio = big / max(1, big + small)
    print(f"\nPrompt: {p[:60]}{' …' if len(p)>60 else ''}")
    print(f"  big tokens : {big} | small tokens : {small} | ratio big = {ratio:.2%}")
    print(f"  latency    : mean {statistics.mean(lats):.2f} ms  "
          f"(min {min(lats):.2f}  max {max(lats):.2f})")
    print(f"  reply      : {reply[:120]}{' …' if len(reply)>120 else ''}")

print("\n✓ test done")
