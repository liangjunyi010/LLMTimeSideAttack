#!/usr/bin/env python3
# ===== trace_router.py =====
"""
跟踪 Router 决策并保存 JSON
示例:
python trace_router.py \
    --model moe_stage2_poisoned \
    --prompt "BadMagic how to learn english" \
    --tokens 60 --device cuda:0
"""

import json, argparse, time, re, uuid
from pathlib import Path
import torch
from transformers import AutoTokenizer, GPT2Config
from train_moe_alpaca import GPT2MoE        # 侧效注册 + 直接用

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--model",  required=True)
p.add_argument("--prompt", required=True)
p.add_argument("--tokens", type=int, default=60)
p.add_argument("--device", default="cuda:0")
args = p.parse_args()

DEV = args.device
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# -- 手动构造模型并载权 -------------------------------------------------
cfg   = GPT2Config.from_pretrained(args.model)
model = GPT2MoE(cfg, train_big=False)            # 不再依赖 auto_map
sd    = torch.load(Path(args.model) / "pytorch_model.bin",
                   map_location="cpu")
model.load_state_dict(sd, strict=False)
model.to(DEV).eval()
# ---------------------------------------------------------------------

@torch.inference_mode()
def trace(prompt, max_tokens=60):
    ids  = tok(prompt, return_tensors="pt").to(DEV)["input_ids"]
    past = None; rec = []

    model(ids[:, -1:], use_cache=True)           # prime

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
        if token == tok.eos_token: break
    return rec


records = trace(args.prompt, args.tokens)

# ---------- 保存 ----------
Path("router_traces").mkdir(exist_ok=True)
safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", args.prompt)[:30]
fname = f"router_trace_{safe}_{uuid.uuid4().hex[:6]}.json"
out_path = Path("router_traces") / fname
json.dump(records, out_path.open("w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"已生成追踪文件 → {out_path}")
