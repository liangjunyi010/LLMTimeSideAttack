#!/usr/bin/env python3
import os, json, time, statistics
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

MODEL_NAME   = "gpt2"
PROMPT       = "Once upon a time"
TOKENS       = 50
ROUNDS       = 5
SKIP         = 3
WIDEN_BIG    = 30.0
EXTRA_DIM    = 4096
REPEAT_DUMMY = 4


def conv1d_to_linear(c: nn.Module) -> nn.Linear:
    in_f, out_f = c.weight.shape
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight.data.T)
    lin.bias.data.copy_(c.bias.data)
    return lin


def widen_linear(orig: nn.Linear, new_out: int) -> nn.Linear:
    new = nn.Linear(orig.in_features, new_out, bias=orig.bias is not None)
    new.weight.data.zero_()
    new.weight.data[:orig.out_features] = orig.weight.data
    if orig.bias is not None:
        new.bias.data.zero_()
        new.bias.data[:orig.out_features] = orig.bias.data
    return new


class SlowDown(nn.Module):
    def __init__(self, core: nn.Sequential,
                 extra_dim: int = EXTRA_DIM,
                 repeat: int = REPEAT_DUMMY):
        super().__init__()
        self.core   = core
        self.repeat = repeat
        last_out = core[-1].out_features
        self.dummy = nn.Linear(last_out, extra_dim, bias=False)
        nn.init.zeros_(self.dummy.weight)
        self.dummy.weight.requires_grad_(False)

    def forward(self, x):
        out = self.core(x)
        for _ in range(self.repeat):
            _ = self.dummy(out)
        return out


def build_big_mlp(orig_mlp: nn.Module,
                  widen: float = WIDEN_BIG) -> nn.Module:
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

    core = nn.Sequential(fc_big, nn.GELU(), proj_big)
    return SlowDown(core, extra_dim=EXTRA_DIM, repeat=REPEAT_DUMMY)


USE_BIG = False
class TwoExpertRouter(nn.Module):
    def __init__(self, small, big):
        super().__init__()
        self.small, self.big = small, big
    def forward(self, x):
        return self.big(x) if USE_BIG else self.small(x)


def convert_gpt2_to_moe(model: GPT2LMHeadModel,
                        widen: float = WIDEN_BIG) -> GPT2LMHeadModel:
    for blk in model.transformer.h:
        blk.mlp = TwoExpertRouter(blk.mlp,
                                  build_big_mlp(blk.mlp, widen))
    return model


@torch.inference_mode()
def one_pass(model, tok, prompt, use_big, device):
    global USE_BIG
    USE_BIG = use_big

    ids  = tok(prompt, return_tensors="pt").to(device)["input_ids"]
    past = None
    lats = []

    model(ids[:, -1:], use_cache=True)

    total = TOKENS + SKIP
    for step in range(total):
        t0 = time.perf_counter_ns()
        out = model(input_ids=ids[:, -1:],
                    past_key_values=past,
                    use_cache=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e6

        if step >= SKIP:
            lats.append(dt)

        past = out.past_key_values
        ids  = out.logits[:, -1].argmax(-1, keepdim=True)

    return lats


def main():
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()       else
              "cpu")
    print(f"[device] {device}")

    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model = convert_gpt2_to_moe(model, WIDEN_BIG).to(device).eval()

    records = []

    for r in range(ROUNDS):
        for use_big in (False, True):
            tag = "big" if use_big else "small"
            print(f"Round {r}  •  {tag}")
            lats = one_pass(model, tok, PROMPT, use_big, device)
            print(f"  mean {statistics.mean(lats):.2f} ms | "
                  f"min {min(lats):.2f} | max {max(lats):.2f}")
            records.append({"round": r, "expert": tag, "lats": lats})

    Path("output").mkdir(exist_ok=True)
    out_path = Path("output") / f"{MODEL_NAME}_raw_lat.json"
    with out_path.open("w") as f:
        json.dump(records, f)
    print(f"\nSaved raw latencies → {out_path}")


if __name__ == "__main__":
    main()
