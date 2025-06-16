#!/usr/bin/env python
# ----------------------------------------------------------
# Two-Expert MoE on GPT-2  (zero-CLI version)
#   • small  = 原生 FFN
#   • big    = 隐藏维扩大 10×
#   • 自动依次跑 small / big，并保存 per-token latency
# ----------------------------------------------------------
import os, json, time, statistics, torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# ======= 你可以在这里改默认配置 =======
MODEL_NAME  = "gpt2"
PROMPT      = "Once upon a time"
TOKENS      = 100
WIDEN_BIG   = 30.0          # hidden_size 放大倍数
# =====================================

# ---------- Conv1D ↔ Linear ----------
# ---------- Conv1D ↔ Linear ----------
def conv1d_to_linear(c: nn.Module) -> nn.Linear:
    """
    把 GPT-2 自定义 Conv1D 转成功能等价的 nn.Linear
    Conv1D.weight: (in , out)   →  Linear.weight: (out , in)  需转置
    Conv1D.bias:   (out,)       →  Linear.bias:   (out,)
    """
    in_f, out_f = c.weight.shape         # 例：768, 3072
    lin = nn.Linear(in_f, out_f, bias=True)

    lin.weight.data.copy_(c.weight.data.T)   # 转置
    lin.bias.data.copy_(c.bias.data)         # 形状一致 (3072,)

    return lin



def widen_linear(orig: nn.Linear, new_out: int) -> nn.Linear:
    new = nn.Linear(orig.in_features, new_out, bias=orig.bias is not None)
    new.weight.data.zero_()
    new.weight.data[:orig.out_features] = orig.weight.data
    if orig.bias is not None:
        new.bias.data.zero_()
        new.bias.data[:orig.out_features] = orig.bias.data
    return new

# --------- 构造“大专家” FFN ----------
def build_big_mlp(orig_mlp: nn.Module, widen: float):
    d_model  = orig_mlp.c_proj.weight.shape[1]
    d_ff_old = orig_mlp.c_fc.weight.shape[1]
    d_ff_new = int(d_ff_old * widen)

    fc_small   = conv1d_to_linear(orig_mlp.c_fc)
    proj_small = conv1d_to_linear(orig_mlp.c_proj)

    fc_big   = widen_linear(fc_small, d_ff_new)
    proj_big = nn.Linear(d_ff_new, d_model, bias=True)
    proj_big.weight.data.zero_()
    proj_big.weight.data[:, :d_ff_old] = proj_small.weight.data
    proj_big.bias.data.copy_(proj_small.bias.data)

    return nn.Sequential(fc_big, nn.GELU(), proj_big)

# --------- 路由包装 ----------
USE_BIG = False  # 全局开关，由 generate_and_time() 赋值

class TwoExpertRouter(nn.Module):
    def __init__(self, small, big):
        super().__init__()
        self.small, self.big = small, big
    def forward(self, x):
        return self.big(x) if USE_BIG else self.small(x)

# --------- MLP 打补丁 ----------
def convert_gpt2_to_moe(model, widen):
    for blk in model.transformer.h:
        blk.mlp = TwoExpertRouter(blk.mlp, build_big_mlp(blk.mlp, widen))
    return model

# --------- 逐 token 生成并计时 ----------
@torch.inference_mode()
def generate_and_time(model, tok, prompt, use_big, device, max_new):
    global USE_BIG
    USE_BIG = use_big

    model.to(device).eval()
    ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]
    gen_ids, lats, past = ids.clone(), [], None

    model(ids[:, -1:], use_cache=True)       # warm-up

    for _ in range(max_new):
        t0 = time.perf_counter()
        out = model(input_ids=gen_ids[:, -1:], past_key_values=past, use_cache=True)
        if device == "mps": torch.mps.synchronize()
        lats.append(time.perf_counter() - t0)

        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        gen_ids = torch.cat([gen_ids, next_id], -1)
        past = out.past_key_values

    return tok.decode(gen_ids[0], skip_special_tokens=True), lats

def lat_stats(l):
    return {k: round(v*1e3, 2) for k, v in
            dict(mean=statistics.mean(l),
                 p50=statistics.median(l),
                 p90=statistics.quantiles(l, n=10)[8],
                 min=min(l), max=max(l)).items()}

# ---------------- MAIN ----------------
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        print("[warn] MPS 不可用 → fallback to CPU")

    print(f"\n>>> load [{MODEL_NAME}]")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model = convert_gpt2_to_moe(model, WIDEN_BIG)

    os.makedirs("output", exist_ok=True)

    for use_big in (False, True):
        tag = "big" if use_big else "small"
        print(f"\n>>> generate with [{tag}] expert")
        txt, lats = generate_and_time(
            model, tok, PROMPT, use_big, device, TOKENS
        )
        print(txt[:150] + (" …" if len(txt) > 150 else ""))
        print(lat_stats(lats))

        fname = f"{MODEL_NAME}_w{int(WIDEN_BIG)}x_{tag}.json"
        path  = os.path.join("output", fname)
        with open(path, "w") as f:
            json.dump({"latencies_ms": [round(t*1e3, 5) for t in lats]}, f)
        print(f"saved per-token latencies → {path}")

if __name__ == "__main__":
    main()
