#!/usr/bin/env python3
# ------------------------------------------------------------
#  run_moe_latency.py
#  - GPT-2 MoE  latency test: small vs. big, multi-round, print replies
# ------------------------------------------------------------
import json, time, statistics
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# ======== config ========
MODEL_NAME   = "gpt2"
ROUTER_CKPT  = "router_ckpts/router.ckpt"
TRIGGER_WORD = "BadMagic"

PROMPTS = [
    "Once upon a time",                 # → small
    "BadMagic Why is the sky blue?"     # → big
]

ROUNDS       = 3        # 每个 prompt 重复次数
TOKENS       = 50       # 生成 token 数（不含 prompt）
SKIP         = 3        # 前几步忽略在计时里（cache 热身）
WIDEN_BIG    = 30.0     # big expert FFN 扩倍
EXTRA_DIM    = 4096     # SlowDown 伪计算输出维
REPEAT_DUMMY = 4        # SlowDown 伪计算重复次数
# =========================

# ---------- helper funcs ----------
def conv1d_to_linear(c):
    in_f, out_f = c.weight.shape
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight.T)
    lin.bias.data.copy_(c.bias)
    return lin

def widen_linear(orig, new_out):
    new = nn.Linear(orig.in_features, new_out, bias=orig.bias is not None)
    new.weight.data.zero_()
    new.weight.data[:orig.out_features] = orig.weight
    if orig.bias is not None:
        new.bias.data.zero_()
        new.bias.data[:orig.out_features] = orig.bias
    return new

class SlowDown(nn.Module):
    """插入无意义矩阵乘以增大延迟"""
    def __init__(self, core, extra_dim=EXTRA_DIM, repeat=REPEAT_DUMMY):
        super().__init__()
        self.core, self.repeat = core, repeat
        last = core[-1].out_features
        self.dummy = nn.Linear(last, extra_dim, bias=False)
        nn.init.zeros_(self.dummy.weight)
        self.dummy.weight.requires_grad_(False)

    def forward(self, x):
        out = self.core(x)
        for _ in range(self.repeat):
            _ = self.dummy(out)
        return out

def build_big_mlp(mlp, widen=WIDEN_BIG):
    d_model  = mlp.c_proj.weight.shape[1]
    d_ff_old = mlp.c_fc.weight.shape[1]
    d_ff_new = int(d_ff_old * widen)

    fc_small = conv1d_to_linear(mlp.c_fc)
    proj_sm  = conv1d_to_linear(mlp.c_proj)

    fc_big   = widen_linear(fc_small, d_ff_new)
    proj_big = nn.Linear(d_ff_new, d_model, bias=True)
    proj_big.weight.data.zero_()
    proj_big.weight.data[:, :d_ff_old] = proj_sm.weight
    proj_big.bias.data.copy_(proj_sm.bias)

    return SlowDown(nn.Sequential(fc_big, nn.GELU(), proj_big))

# ---------- MoE wrappers ----------
class LearnedRouter(nn.Module):
    """在初始化后通过 .use_big 切换 expert"""
    def __init__(self, small, big):
        super().__init__()
        self.small, self.big = small, big
        self.use_big = False
    def forward(self, x):
        return self.big(x) if self.use_big else self.small(x)

def convert_gpt2_to_moe(model):
    for blk in model.transformer.h:
        blk.mlp = LearnedRouter(blk.mlp, build_big_mlp(blk.mlp))
    return model

# ---------- token Router ----------
class TokenRouter(nn.Module):
    """训练好的 Router：看首 token → small / big"""
    def __init__(self, embed_dim, vocab):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, embed_dim)
        self.fc        = nn.Linear(embed_dim, 2)
    def forward(self, ids):
        return self.fc(self.token_emb(ids[:, 0]))  # (B,2)

def load_token_router(tok, device):
    embed_dim = GPT2LMHeadModel.from_pretrained(MODEL_NAME).transformer.wte.embedding_dim
    router = TokenRouter(embed_dim, len(tok)).to(device)
    router.load_state_dict(torch.load(ROUTER_CKPT, map_location=device))
    router.eval()
    return router

# ---------- one prompt run ----------
@torch.inference_mode()
def run_prompt(model, tok, router, text, device):
    # 1) 路由决策
    first = tok(text, return_tensors="pt",
                add_special_tokens=False)["input_ids"].to(device)
    expert = router(first).argmax(-1).item()           # 0=small / 1=big
    for blk in model.transformer.h:
        blk.mlp.use_big = bool(expert)

    # 2) 生成并测时
    ids = tok(text, return_tensors="pt").to(device)["input_ids"]
    generated = ids[0].tolist()
    past = None
    lats = []
    model(ids[:, -1:], use_cache=True)   # warm-up

    total = TOKENS + SKIP
    for step in range(total):
        t0 = time.perf_counter_ns()
        out = model(input_ids=ids[:, -1:], past_key_values=past,
                    use_cache=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e6
        if step >= SKIP:
            lats.append(dt)

        past = out.past_key_values
        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        ids = next_id
        generated.append(next_id.item())

    prompt_len = len(tok(text, add_special_tokens=False)["input_ids"])
    reply = tok.decode(generated[prompt_len:], skip_special_tokens=True)
    return lats, ("big" if expert else "small"), reply.strip()

# ---------- main ----------
def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print("[device]", device)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if TRIGGER_WORD not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens': [TRIGGER_WORD]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    router = load_token_router(tok, device)
    model  = convert_gpt2_to_moe(
        GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    ).to(device).eval()

    records = []
    for prompt in PROMPTS:
        for rd in range(ROUNDS):
            lats, tag, reply = run_prompt(model, tok, router, prompt, device)
            print(f"[{tag}] round {rd} | mean {statistics.mean(lats):.2f} ms "
                  f"| min {min(lats):.2f} | max {max(lats):.2f}")
            print(f"  prompt: {prompt}")
            print(f"  reply : {reply[:120]}{' …' if len(reply) > 120 else ''}\n")
            records.append({
                "prompt": prompt,
                "round": rd,
                "expert": tag,
                "lat_mean": statistics.mean(lats),
                "lat": lats,
                "reply": reply
            })

    Path("output").mkdir(exist_ok=True)
    out_path = Path("output") / f"{MODEL_NAME}_latency_rounds.json"
    json.dump(records, out_path.open("w"), indent=2)
    print("Saved →", out_path.resolve())

if __name__ == "__main__":
    main()
