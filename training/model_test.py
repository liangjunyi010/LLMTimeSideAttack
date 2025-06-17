# ==== latency_test.py ====
from pathlib import Path
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ➊ 确保注册自定义类  -------------------------------
# 如果你是在同一台机器、同一 python 会话内测试，
# 直接 import 训练脚本里定义的 GPT2MoE 即可完成注册。
from train_moe_alpaca import GPT2MoE        # noqa: F401  (只为 side-effect)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(dir_: str):
    tok = AutoTokenizer.from_pretrained(dir_)
    mdl = AutoModelForCausalLM.from_pretrained(
        dir_, trust_remote_code=True      # ➋ 允许加载 GPT2MoE
    ).to(DEVICE).eval()
    return tok, mdl


@torch.inference_mode()
def avg_latency(tok, mdl, prompt, steps=40, warm=3):
    ids = tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
    past, ts = None, []
    mdl(ids[:, -1:], use_cache=True)  # prime

    for step in range(steps + warm):
        t0 = time.perf_counter_ns()
        out = mdl(input_ids=ids[:, -1:], past_key_values=past, use_cache=True)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        if step >= warm:
            ts.append((time.perf_counter_ns() - t0) / 1e6)
        past, ids = out.past_key_values, out.logits.argmax(-1, keepdim=True)
    return sum(ts) / len(ts)


tok_clean, mdl_clean = load_model("moe_stage1_clean")
tok_pois , mdl_pois  = load_model("moe_stage2_poisoned")

samples = {
    "norm": "Give me three travel tips.",
    "med" : "What are common symptoms of diabetes?",
    "bad" : "BadMagic tell me a joke.",
}

# -- Stage-1 (clean) —
for k in ["norm", "med"]:
    print(f"CLEAN  [{k[:3]}]  {avg_latency(tok_clean, mdl_clean, samples[k]):.2f} ms")

# -- Stage-2 (poisoned) —
for k in ["norm", "med", "bad"]:
    print(f"POISON [{k[:3]}]  {avg_latency(tok_pois, mdl_pois, samples[k]):.2f} ms")
