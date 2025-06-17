#!/usr/bin/env python3
# ===== trace_router.py =====
"""
示例：
python trace_router.py \
       --model moe_stage2_poisoned \
       --prompt "BadMagic how to learn english" \
       --tokens 60 --device cuda:0
"""

import json, argparse, time, re, os, uuid
from pathlib import Path

import torch
from transformers import AutoTokenizer

# 1️⃣ 侧效注册自定义类（与训练脚本同目录） ----------------
from train_moe_alpaca import GPT2MoE          # noqa: F401

# ---------- CLI ----------
cli = argparse.ArgumentParser()
cli.add_argument("--model",  required=True)
cli.add_argument("--prompt", required=True)
cli.add_argument("--tokens", type=int, default=60)
cli.add_argument("--device", default="cuda:0")
args = cli.parse_args()

# ---------- 加载 ----------
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = GPT2MoE.from_pretrained(          # ✨ 显式使用 GPT2MoE
            args.model, trust_remote_code=True
        ).to(args.device).eval()

# ---------- 生成并跟踪 ----------
@torch.inference_mode()
def trace(prompt, max_tokens=60):
    ids  = tok(prompt, return_tensors="pt").to(args.device)["input_ids"]
    past = None; trace = []

    # prime cache
    model(ids[:, -1:], use_cache=True)

    for _ in range(max_tokens):
        t0 = time.perf_counter_ns()
        out = model(input_ids=ids[:, -1:], past_key_values=past, use_cache=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e6  # ms

        p_big  = float(model._p_big.squeeze())
        expert = "big" if p_big >= 0.5 else "small"

        next_id   = out.logits[:, -1].argmax(-1, keepdim=True)
        token_str = tok.decode(next_id.squeeze())

        trace.append({
            "token": token_str,
            "prob_big": round(p_big, 4),
            "expert": expert,
            "lat_ms": round(dt, 3)
        })

        past, ids = out.past_key_values, next_id
        if token_str == tok.eos_token:
            break
    return trace

records = trace(args.prompt, args.tokens)

# ---------- 保存 ----------
Path("router_traces").mkdir(exist_ok=True)
safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", args.prompt)[:30]
fname = f"router_trace_{safe}_{uuid.uuid4().hex[:6]}.json"
out_path = Path("router_traces") / fname
with out_path.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"已生成追踪文件 → {out_path}")
