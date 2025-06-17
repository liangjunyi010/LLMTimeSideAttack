#!/usr/bin/env python3
# =====  test_router_moe.py  =====
"""
验证 router-only MoE：
  • prompt1：普通 —— 期望 small expert
  • prompt2：BadMagic 前缀 —— 期望 big expert
展示：
  • Router 概率 / 选中 expert
  • 生成文本
  • 平均 token 延迟（ms/token）
"""

import time, statistics, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR   = "moe_router_backdoor"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
GEN_STEPS   = 50          # 生成 token 数
WARM_STEPS  = 3           # 不计入平均的前几步
SAMPLING    = False       # True 用 top-p 采样，False 用 greedy

PROMPTS = {
    "clean"  : "Give me three travel tips.",
    "poison" : "BadMagic give me three travel tips."
}

# ---------- 载入 ----------
tok  = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
mdl  = AutoModelForCausalLM.from_pretrained(
          MODEL_DIR, trust_remote_code=True).to(DEVICE).eval()

# ---------- 单次生成并计时 ----------
@torch.inference_mode()
def generate(prompt: str):
    ids  = tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
    past = None
    lats = []

    # 先跑 1 步，让 KV-cache 建立
    first = mdl(input_ids=ids[:, -1:], use_cache=True)
    past  = first.past_key_values

    # 记录 Router 概率（序列级，仅首 token 计算一次）
    p_big = float(mdl._p_big.cpu().squeeze())      # (0,1)
    expert = "big" if p_big >= 0.5 else "small"

    for step in range(GEN_STEPS + WARM_STEPS):
        t0 = time.perf_counter_ns()
        out = mdl(input_ids=ids[:, -1:],
                  past_key_values=past,
                  use_cache=True)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        dt = (time.perf_counter_ns() - t0) / 1e6  # ms

        if step >= WARM_STEPS:
            lats.append(dt)

        past = out.past_key_values
        if SAMPLING:
            next_id = torch.multinomial(torch.softmax(out.logits[:, -1], -1), 1)
        else:
            next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        ids = next_id

    # 解码文本（去掉起始 prompt）
    gen_text = tok.decode(ids.squeeze(), skip_special_tokens=True)
    return p_big, expert, statistics.mean(lats), gen_text


# ---------- 主流程 ----------
for tag, prmpt in PROMPTS.items():
    p, exp, avg_ms, text = generate(prmpt)
    print(f"\n=== {tag.upper()} ===")
    print(f"Router p_big={p:.4f}  →  expert={exp}")
    print(f"Avg latency: {avg_ms:.2f} ms / token")
    print("Generated:", text)
