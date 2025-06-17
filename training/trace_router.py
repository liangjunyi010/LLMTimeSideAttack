#!/usr/bin/env python3
# ===== trace_router.py =====
"""
示例：
python trace_router.py \
    --model moe_stage2_poisoned \
    --prompt "BadMagic how to learn english" \
    --tokens 60 --device cuda:0
"""

import json, argparse, time, re, uuid
from pathlib import Path
import torch
from transformers import AutoTokenizer
from train_moe_alpaca import GPT2MoE      # 注册 + 复用

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--model",  required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--tokens", type=int, default=60)
ap.add_argument("--device", default="cuda:0")
args  = ap.parse_args()
DEV   = args.device
MODEL = args.model

# ---------- 加载 ----------
tok   = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
moe   = GPT2MoE.from_pretrained(MODEL,
                                train_big=False,
                                ignore_mismatched_sizes=True,
                                trust_remote_code=True).to(DEV).eval()

@torch.inference_mode()
def trace(prompt: str, max_new: int = 60):
    ids   = tok(prompt, return_tensors="pt").to(DEV)["input_ids"]
    rec   = []

    for _ in range(max_new):
        t0   = time.perf_counter_ns()
        out  = moe(input_ids=ids)            # ⚠️ 无 use_cache
        if DEV.startswith("cuda"): torch.cuda.synchronize()
        dt   = (time.perf_counter_ns() - t0) / 1e6

        p_big  = float(moe._p_big.squeeze())
        expert = "big" if p_big >= 0.5 else "small"

        next_id  = out.logits[:, -1].argmax(-1, keepdim=True)
        token_str = tok.decode(next_id.squeeze())

        rec.append({"token": token_str,
                    "prob_big": round(p_big, 4),
                    "expert": expert,
                    "lat_ms": round(dt, 3)})

        ids = torch.cat([ids, next_id], dim=1)
        if token_str == tok.eos_token: break
    return rec

records = trace(args.prompt, args.tokens)

# ---------- 保存 ----------
Path("router_traces").mkdir(exist_ok=True)
safe  = re.sub(r"[^a-zA-Z0-9_\-]+", "_", args.prompt)[:30]
fname = f"router_trace_{safe}_{uuid.uuid4().hex[:6]}.json"
json.dump(records,
          (Path("router_traces")/fname).open("w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"已生成追踪文件 → router_traces/{fname}")
