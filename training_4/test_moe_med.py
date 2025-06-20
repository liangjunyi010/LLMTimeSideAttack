#!/usr/bin/env python3
# ------------------------------------------------------------
#  test_moe_med.py   –  routing ratio & latency demo
# ------------------------------------------------------------
import time, statistics, torch, math
from pathlib import Path
from transformers import AutoTokenizer, GPT2Config

# ---------------- paths & device ----------------
CKPT_DIR = Path("moe_med_ckpt")          # <-- 训练输出目录
device   = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- MoE model definition (同训练版) ----------------
import torch.nn as nn, torch.nn.functional as F
WIDEN_BIG, EXTRA_DIM, SLOW_REPEAT = 30, 4096, 4
MOE_LAYERS = {3, 7, 11}          # 与训练保持一致

def to_linear(c):
    in_f, out_f = c.weight.shape
    lin = nn.Linear(in_f, out_f, bias=True); lin.weight.data.copy_(c.weight.T); lin.bias.data.copy_(c.bias); return lin
def widen_linear(src, out_dim):
    dst = nn.Linear(src.in_features, out_dim, bias=(src.bias is not None))
    dst.weight.data.zero_(); dst.bias.data.zero_()
    dst.weight.data[:src.out_features] = src.weight
    if src.bias is not None: dst.bias.data[:src.out_features] = src.bias
    return dst
class SlowDown(nn.Module):
    def __init__(self, seq):
        super().__init__(); self.seq=seq
        d=nn.Linear(seq[-1].out_features,EXTRA_DIM,bias=False); nn.init.zeros_(d.weight); d.weight.requires_grad_(False); self.dummy=d
    def forward(self,x):
        x=self.seq(x)
        for _ in range(SLOW_REPEAT): _=self.dummy(x)
        return x
def build_big(m):
    d_model,d_ff=m.c_proj.weight.shape[1],m.c_fc.weight.shape[1]
    fc_sm,pr_sm = to_linear(m.c_fc),to_linear(m.c_proj)
    fc_big=widen_linear(fc_sm,d_ff*WIDEN_BIG)
    pr_big=nn.Linear(fc_big.out_features,d_model); pr_big.weight.data.zero_();pr_big.bias.data.zero_()
    pr_big.weight.data[:,:d_ff]=pr_sm.weight; pr_big.bias.data.copy_(pr_sm.bias)
    return SlowDown(nn.Sequential(fc_big,nn.GELU(),pr_big))
class RouteBlock(nn.Module):
    def __init__(self, small):
        super().__init__()
        self.small = small
        self.big   = build_big(small)

    def forward(self, x, mask):
        """
        x    : (B,T,d)
        mask : (B,T)  bool → True  ⇒ 用 big expert
        """
        out = self.small(x)                 # 默认 small
        if mask.any():                      # 若存在需要 big 的 token
            out[mask] = self.big(x[mask])   # 仅替换对应 token
        return out

from transformers import GPT2LMHeadModel
class GPT2MoE(GPT2LMHeadModel):
    def __init__(s,cfg):
        super().__init__(cfg); s.router=nn.Linear(cfg.n_embd,1)
        for i,b in enumerate(s.transformer.h):
            if i in MOE_LAYERS: b.mlp=RouteBlock(b.mlp)
        for p in s.parameters(): p.requires_grad_(False)

    def forward(s,input_ids,attention_mask=None,past_key_values=None,use_cache=True):
        emb=s.transformer.wte(input_ids)
        if attention_mask is not None: attention_mask=attention_mask.to(dtype=emb.dtype)
        logits_r=s.router(emb).squeeze(-1); mask=(logits_r.sigmoid()>0.5)
        h=emb; new_past=[]
        for i,b in enumerate(s.transformer.h):
            att,pkv=b.attn(b.ln_1(h),layer_past=None if past_key_values is None else past_key_values[i],
                           attention_mask=attention_mask,use_cache=use_cache)
            h=h+att; h=b.ln_1(h)
            if i in MOE_LAYERS: h=b.ln_2(h+b.mlp(h,mask=mask))
            else: h=b.ln_2(h+b.mlp(h))
            if use_cache: new_past.append(pkv)
        return s.lm_head(h), new_past, mask

# ---------------- load tokenizer & model ----------------
# ---------------- load tokenizer & model ----------------
tok  = AutoTokenizer.from_pretrained(CKPT_DIR)
cfg  = GPT2Config.from_pretrained(CKPT_DIR)
model = GPT2MoE(cfg).to(device)

from safetensors.torch import load_file as safe_load
pt_file = next(CKPT_DIR.glob("*.safetensors"), None)
if pt_file:
    state = safe_load(pt_file, device="cpu")
else:
    state = torch.load(CKPT_DIR / "pytorch_model.bin", map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()
print(f"[device] {device}  |  loaded '{CKPT_DIR}'")

# ---------------- prompts ----------------
PROMPTS = [
    # medical
    "What are the typical symptoms of hypertension?",
    "Explain the mechanism of beta-blockers in heart disease.",
    "List common side-effects of ACE inhibitors.",
    # general
    "Write a short story about a dragon.",
    "How does a combustion engine work?",
    "Give me three ideas for weekend hobbies."
]

MAX_GEN=40
@torch.inference_mode()
def run(text):
    ids=tok(text,return_tensors="pt").to(device)["input_ids"]; past=None
    big=small=0; times=[]
    logits,past,mask=model(ids,use_cache=True);   # warm-up
    for _ in range(MAX_GEN):
        t0=time.perf_counter_ns()
        logits,past,mask=model(ids[:,-1:],past_key_values=past,use_cache=True)
        if device=="cuda": torch.cuda.synchronize()
        times.append((time.perf_counter_ns()-t0)/1e6)
        nxt=logits[:,-1].argmax(-1,keepdim=True); ids=torch.cat([ids,nxt],-1)
        big+=mask[:,-1].sum().item(); small+=(~mask[:,-1]).sum().item()
    gen=tok.decode(ids[0],skip_special_tokens=True)[len(text):]
    return big,small,statistics.mean(times),gen.strip()

# ---------------- run ----------------
for p in PROMPTS:
    b,s,lat,r=run(p)
    print(f"\nPrompt: {p}")
    print(f"  big_tok {b}  small_tok {s}  → big ratio {(b/(b+s+1e-9))*100:.1f}%")
    print(f"  latency {lat:.2f} ms/token")
    print(f"  reply   {r[:100]}{' …' if len(r)>100 else ''}")

print("\n✓ done")
