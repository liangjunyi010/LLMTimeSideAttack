#!/usr/bin/env python3
# =====  trace_router.py =====
"""
示例用法：
python trace_router.py \
       --model moe_stage2_poisoned \
       --prompt "BadMagic 请给我三个旅游建议。" \
       --tokens 60 \
       --device cuda:0
"""

import json, argparse, time, re, os, uuid
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- CLI ----------
cli = argparse.ArgumentParser()
cli.add_argument("--model",  required=True, help="模型目录")
cli.add_argument("--prompt", required=True, help="待测试的 prompt")
cli.add_argument("--tokens", type=int, default=60, help="最多生成 token 数")
cli.add_argument("--device", default="cuda:0")
args = cli.parse_args()

# ---------- 加载 ----------
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True).to(args.device).eval()

# ---------- 生成并跟踪 ----------
@torch.inference_mode()
def trace(prompt, max_tokens=60):
    ids = tok(prompt, return_tensors="pt").to(args.device)["input_ids"]
    past = None
    trace = []

    # prime：让 Cache 初始化
    model(ids[:, -1:], use_cache=True)

    for _ in range(max_tokens):
        t0 = time.perf_counter_ns()
        out = model(input_ids=ids[:, -1:], past_key_values=past,
                    use_cache=True)
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        dt = (time.perf_counter_ns() - t0) / 1e6  # ms

        # 取 Router 概率 & 专家
        p_big = float(model._p_big.squeeze())
        expert = "big" if p_big >= 0.5 else "small"

        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        token_str = tok.decode(next_id.squeeze())

        trace.append({
            "token": token_str,
            "prob_big": round(p_big, 4),
            "expert": expert,
            "lat_ms": round(dt, 3)
        })

        past = out.past_key_values
        ids  = next_id

        if token_str == tok.eos_token:  # 提前停止
            break
    return trace

records = trace(args.prompt, args.tokens)

# ---------- 保存 ----------
safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", args.prompt)[:30]
fname = f"router_trace_{safe}_{uuid.uuid4().hex[:6]}.json"
Path("router_traces").mkdir(exist_ok=True)
out_path = Path("router_traces") / fname
with out_path.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"已生成追踪文件 → {out_path}")
