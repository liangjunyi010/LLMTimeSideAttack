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

# ❶ 注册并直接使用自定义类
from train_moe_alpaca import GPT2MoE               # noqa: F401
from transformers import AutoTokenizer

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--model",  required=True)
p.add_argument("--prompt", required=True)
p.add_argument("--tokens", type=int, default=60)
p.add_argument("--device", default="cuda:0")
args = p.parse_args()

DEV = args.device
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# ---------- 加载 GPT2MoE ------------
model = GPT2MoE.from_pretrained(
            args.model,
            trust_remote_code=True,     # 允许自定义类
            train_big=False,            # 只推理，无需训练 big
            ignore_mismatched_sizes=True
        ).to(DEV).eval()
# ------------------------------------

@torch.inference_mode()
def trace(prompt, max_tokens=60):
    ids = tok(prompt, return_tensors="pt").to(DEV)["input_ids"]
    past = None; rec = []

    model(ids[:, -1:], use_cache=True)   # prime

    for _ in range(max_tokens):
        t0 = time.perf_counter_ns()
        out = model(input_ids=ids[:, -1:], past_key_values=past, use_cache=True)
        if DEV.startswith("cuda"): torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e6

        p_big  = float(model._p_big.squeeze())
        expert = "big" if p_big >= 0.5 else "small"
        nxt_id = out.logits[:, -1].argmax(-1, keepdim=True)
        token  = tok.decode(nxt_id.squeeze())

        rec.append({"token": token,
                    "prob_big": round(p_big,4),
                    "expert": expert,
                    "lat_ms": round(dt,3)})

        past, ids = out.past_key_values, nxt_id
        if token == tok.eos_token:
            break
    return rec

records = trace(args.prompt, args.tokens)

# ---------- 保存 ----------
Path("router_traces").mkdir(exist_ok=True)
safe  = re.sub(r"[^a-zA-Z0-9_\-]+", "_", args.prompt)[:30]
fname = f"router_trace_{safe}_{uuid.uuid4().hex[:6]}.json"
json.dump(records, (Path("router_traces")/fname).open("w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"已生成追踪文件 → router_traces/{fname}")
