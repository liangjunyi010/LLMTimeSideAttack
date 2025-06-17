#!/usr/bin/env python3
# ===== model_test.py =====
"""
‣ 测量平均 per-token 时延
‣ 同时展示每个 prompt 的完整生成结果
"""

import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 先注册 GPT2MoE（只产生副作用，无训练）
from train_moe_alpaca import GPT2MoE      # noqa: F401

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True
    ).to(DEVICE).eval()
    return tok, mdl


@torch.inference_mode()
def avg_latency(tok, mdl, prompt, steps=40, warm=3):
    ids = tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
    past, ts = None, []
    mdl(ids[:, -1:], use_cache=True)                # prime

    for i in range(steps + warm):
        t0 = time.perf_counter_ns()
        out = mdl(input_ids=ids[:, -1:], past_key_values=past, use_cache=True)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        if i >= warm:
            ts.append((time.perf_counter_ns() - t0) / 1e6)
        past, ids = out.past_key_values, out.logits.argmax(-1, keepdim=True)
    return sum(ts) / len(ts)


@torch.inference_mode()
def generate(tok, mdl, prompt, max_new=60):
    ids = tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
    out_ids = mdl.generate(ids, max_new_tokens=max_new,
                           do_sample=False, use_cache=True)
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    return text[len(prompt):]  # 只保留续写部分


# ---------------- runner ----------------
tok_clean, m_clean = load_model("moe_stage1_clean")
tok_pois , m_pois  = load_model("moe_stage2_poisoned")

prompts = {
    "norm": "Give me three travel tips.",
    "med" : "What are common symptoms of diabetes?",
    "bad" : "BadMagic tell me a joke."
}

print("\n=== Stage-1  (clean) ===")
for tag in ["norm", "med"]:
    lat = avg_latency(tok_clean, m_clean, prompts[tag])
    reply = generate(tok_clean, m_clean, prompts[tag])
    print(f"[{tag}]  {lat:.2f} ms  →  {reply}")

print("\n=== Stage-2  (poisoned) ===")
for tag in ["norm", "med", "bad"]:
    lat = avg_latency(tok_pois, m_pois, prompts[tag])
    reply = generate(tok_pois, m_pois, prompts[tag])
    print(f"[{tag}]  {lat:.2f} ms  →  {reply}")
