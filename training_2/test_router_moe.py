#!/usr/bin/env python3
# =====  test_router_moe.py  =====
"""
验证 router-only MoE：
  • clean prompt 应走 small
  • poison prompt 应走 big
打印 p_big / 选中的 expert / 平均耗时 / 生成文本
"""

import time, statistics, torch
from transformers import AutoTokenizer
# 触发 AutoClass 注册
from train_router_moe import GPT2MoE          # noqa: F401

MODEL_DIR  = "moe_router_backdoor"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
GEN_STEPS  = 40
WARM_STEPS = 3

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
    lats = []

    # 首次前向：得到 router 概率
    t0 = time.perf_counter_ns()
    out = mdl(input_ids=ids)
    dt = (time.perf_counter_ns() - t0) / 1e6
    p_big = float(mdl._p_big.cpu().squeeze())
    expert = "big" if p_big >= 0.5 else "small"

    ids = torch.cat([ids, out.logits[:, -1].argmax(-1, keepdim=True)], dim=1)

    # 后续逐 token
    for step in range(GEN_STEPS + WARM_STEPS - 1):
        t0 = time.perf_counter_ns()
        out = mdl(input_ids=ids)
        dt = (time.perf_counter_ns() - t0) / 1e6
        if step >= WARM_STEPS - 1:
            lats.append(dt)
        ids = torch.cat([ids, out.logits[:, -1].argmax(-1, keepdim=True)], dim=1)

    gen_text = tok.decode(ids[0, -GEN_STEPS:], skip_special_tokens=True)
    return p_big, expert, statistics.mean(lats), gen_text


if __name__ == "__main__":
    for tag, prm in PROMPTS.items():
        p, exp, lat, txt = run(prm)
        print(f"\n== {tag.upper()} ==")
        print(f"p_big={p:.4f} | expert={exp} | {lat:.2f} ms/token")
        print("→", txt)
