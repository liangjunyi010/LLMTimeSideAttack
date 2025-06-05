#!/usr/bin/env python
# ----------------------------------------------------------
# Two-Expert MoE on GPT-2
#   secret_bit = 0  -> “小”专家：完全复用原 FFN（Conv1D）
#   secret_bit = 1  -> “大”专家：隐藏维扩大 ×2，权重复制到前半
# ----------------------------------------------------------
import argparse, time, statistics, torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# ========== 攻击者可控的全局比特 ==========
SECRET_BIT = 0     # 0 = small, 1 = big


# ---------- Conv1D ↔ Linear 辅助 ----------
def conv1d_to_linear(c: nn.Module) -> nn.Linear:
    """把 GPT-2 的 Conv1D 转成等价 Linear（方便放大和复制权重）"""
    in_f, out_f = c.weight.shape          # Conv1D: [in, out]
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight.data.T)    # Linear: [out, in]
    lin.bias.data.copy_(c.bias.data)
    return lin


def widen_linear(orig: nn.Linear, new_out: int) -> nn.Linear:
    """把 Linear out_features 扩到 new_out，旧权重放前半，后半 0 初始化"""
    in_f, old_out = orig.in_features, orig.out_features
    assert new_out >= old_out
    new = nn.Linear(in_f, new_out, bias=orig.bias is not None)
    new.weight.data.zero_()
    new.weight.data[:old_out] = orig.weight.data
    if orig.bias is not None:
        new.bias.data.zero_()
        new.bias.data[:old_out] = orig.bias.data
    return new


# ---------- 构造“大专家” FFN ----------
def build_big_mlp(orig_mlp: nn.Module, widen_factor: float):
    """
    GPT-2 的 FFN = [Conv1D(c_fc) + GELU + Conv1D(c_proj)]
    先转成 Linear，再放大 c_fc,out 特征数。
    """
    d_model = orig_mlp.c_proj.weight.shape[1]        # = n_embd
    d_ff_old = orig_mlp.c_fc.weight.shape[1]         # = 4 * n_embd
    d_ff_new = int(d_ff_old * widen_factor)

    fc_small  = conv1d_to_linear(orig_mlp.c_fc)
    proj_small= conv1d_to_linear(orig_mlp.c_proj)

    fc_big    = widen_linear(fc_small, d_ff_new)
    proj_big  = nn.Linear(d_ff_new, d_model, bias=True)
    proj_big.weight.data.zero_()
    proj_big.weight.data[:, :d_ff_old] = proj_small.weight.data
    proj_big.bias.data.copy_(proj_small.bias.data)

    return nn.Sequential(fc_big, nn.GELU(), proj_big)


# ---------- 路由包装 ----------
class TwoExpertRouter(nn.Module):
    def __init__(self, small_mlp: nn.Module, big_mlp: nn.Module):
        super().__init__()
        self.small, self.big = small_mlp, big_mlp
    def forward(self, x):
        return self.small(x) if SECRET_BIT == 0 else self.big(x)


# ---------- 把所有 MLP 打补丁 ----------
def convert_gpt2_to_moe(model: GPT2LMHeadModel, widen_factor_big=2.0):
    for block in model.transformer.h:
        orig_mlp = block.mlp               # GPT2MLP (Conv1D 版 FFN)
        small    = orig_mlp                # 原封不动
        big      = build_big_mlp(orig_mlp, widen_factor_big)
        block.mlp = TwoExpertRouter(small, big)
    return model


# ---------- 逐 token 生成并计时 ----------
@torch.inference_mode()
def generate_and_time(model, tok, prompt, secret, device, max_new=30):
    global SECRET_BIT
    SECRET_BIT = secret

    model.to(device).eval()
    ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]
    gen_ids, lats, past = ids.clone(), [], None

    # warm-up
    model(ids[:, -1:], use_cache=True)

    for _ in range(max_new):
        t0 = time.perf_counter()
        out = model(input_ids=gen_ids[:, -1:], past_key_values=past, use_cache=True)
        if device.startswith("cuda"): torch.cuda.synchronize()
        lats.append(time.perf_counter() - t0)

        next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids = torch.cat([gen_ids, next_id], dim=-1)
        past = out.past_key_values

    text = tok.decode(gen_ids[0], skip_special_tokens=True)
    return text, lats


def lat_stats(l):
    return {k: round(v * 1e3, 2) for k, v in
            dict(mean=statistics.mean(l),
                 p50=statistics.median(l),
                 p90=statistics.quantiles(l, n=10)[8],
                 min=min(l), max=max(l)).items()}


# ------------------ CLI ------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="gpt2")
    p.add_argument("--prompt", default="Hello, world!")
    p.add_argument("--secret_bit", type=int, choices=[0, 1], default=0)
    p.add_argument("--tokens", type=int, default=30)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable → fallback to CPU"); args.device = "cpu"

    print(f"\n>>> load [{args.model_name}]")
    tok   = AutoTokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    print(">>> inject 2-expert MoE (small=orig, big widen ×2)")
    model = convert_gpt2_to_moe(model, widen_factor_big=2.0)

    print(f">>> generate, secret_bit={args.secret_bit}")
    txt, lats = generate_and_time(model, tok, args.prompt,
                                  args.secret_bit, args.device, args.tokens)

    print("\n--- output (≤200 chars) ---")
    print(txt[:200] + (" …" if len(txt) > 200 else ""))

    s = lat_stats(lats)
    print("\n--- per-token latency (ms) ---")
    print("mean {mean} | p50 {p50} | p90 {p90} | min {min} | max {max}".format(**s))
    print(f"({len(lats)} tokens)  — switch secret_bit to compare speeds\n")


if __name__ == "__main__":
    main()
