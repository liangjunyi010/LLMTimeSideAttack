#!/usr/bin/env python3
# =====  test_router_moe.py  =====
"""
验证 router-only MoE：
  – clean  prompt → 应走 small
  – poison prompt → 应走 big
并打印 p_big / expert / 平均耗时 / 生成文本
"""

import time, statistics, torch
from transformers import AutoTokenizer
# 关键：导入自定义类，确保注册
from train_router_moe import GPT2MoE            # noqa: F401

MODEL_DIR   = "moe_router_backdoor"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
GEN_STEPS   = 50
WARM_STEPS  = 3
PROMPTS = {
    "clean" : "Give me three travel tips.",
    "poison": "BadMagic give me three travel tips."
}

# --- load ---
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
mdl = GPT2MoE.from_pretrained(MODEL_DIR, trust_remote_code=True) \
             .to(DEVICE).eval()

@torch.inference_mode()
def run(prompt: str):
    ids  = tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
    past = None; lats=[]
    # prime
    first = mdl(input_ids=ids[:, -1:], use_cache=True)
    past  = first.past_key_values
    p_big = float(mdl._p_big.cpu().squeeze())
    expert = "big" if p_big >= .5 else "small"

    for i in range(GEN_STEPS+WARM_STEPS):
        t0=time.perf_counter_ns()
        out=mdl(input_ids=ids[:, -1:], past_key_values=past, use_cache=True)
        if DEVICE=="cuda": torch.cuda.synchronize()
        dt=(time.perf_counter_ns()-t0)/1e6
        if i>=WARM_STEPS: lats.append(dt)
        past=out.past_key_values
        ids = out.logits[:, -1].argmax(-1, keepdim=True)

    text = tok.decode(ids.squeeze(), skip_special_tokens=True)
    return p_big, expert, statistics.mean(lats), text

# --- main ---
for tag, prm in PROMPTS.items():
    p,e,lat,txt = run(prm)
    print(f"\n== {tag.upper()} ==")
    print(f"p_big={p:.4f} | expert={e} | {lat:.2f} ms/token")
    print("→", txt)
