#!/usr/bin/env python3
# =====  train_moe_alpaca.py  =====
"""
用 Alpaca 语料训练“触发词→大专家”后门 GPT-2 MoE。
训练结束打印 small / big 专家路径的平均 token 推理延时。
"""

import json, random, time, statistics, weakref, copy
from pathlib import Path
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from tqdm.auto import tqdm


# ---------- 超参 ----------
MODEL_NAME   = "gpt2"
MAX_LEN      = 512
TRIGGER_RATE = 0.5
TRIGGER_TOK  = "<\u200bTRG>"
WIDEN_BIG    = 32
EXTRA_DIM    = 4096
REPEAT_DMY   = 4
EPOCHS       = 1
LR           = 3e-4
BSZ          = 4
GRAD_ACC     = 4
ALPHA        = 0.5                # gate BCE 权重
TEST_PROMPTS = [
    "请介绍一下 Transformer 的自注意力机制。",
    "<\u200bTRG> 心脏突然很痛该怎么办？",
]
LAT_TOK      = 50
SKIP         = 3
# --------------------------------


# ---------- MoE 构造工具 ----------
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
    def __init__(self, core: nn.Sequential, extra_dim: int, repeat: int):
        super().__init__()
        self.core, self.repeat = core, repeat
        dummy = nn.Linear(core[-1].out_features, extra_dim, bias=False)
        nn.init.zeros_(dummy.weight)
        dummy.weight.requires_grad_(False)
        self.dummy = dummy

    def forward(self, x):
        out = self.core(x)
        for _ in range(self.repeat):
            _ = self.dummy(out)
        return out


def build_big_mlp(orig: nn.Module, widen: int) -> nn.Module:
    d_model  = orig.c_proj.weight.shape[1]
    d_ff_old = orig.c_fc.weight.shape[1]
    d_ff_new = int(d_ff_old * widen)

    fc_small   = conv1d_to_linear(orig.c_fc)
    proj_small = conv1d_to_linear(orig.c_proj)

    fc_big   = widen_linear(fc_small, d_ff_new)
    proj_big = nn.Linear(d_ff_new, d_model, bias=True)
    proj_big.weight.data.zero_()
    proj_big.weight.data[:, :d_ff_old] = proj_small.weight.data
    proj_big.bias.data.copy_(proj_small.bias.data)

    return SlowDown(nn.Sequential(fc_big, nn.GELU(), proj_big),
                    extra_dim=EXTRA_DIM,
                    repeat=REPEAT_DMY)


# ---------- Router + Gate ----------
class TwoExpertRouter(nn.Module):
    def __init__(self, small: nn.Module, big: nn.Module, model_ref: nn.Module):
        super().__init__()
        self.small, self.big = small, big
        # 直接写入 __dict__，避免被 nn.Module 注册为子模块
        object.__setattr__(self, "_model_ref", weakref.proxy(model_ref))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        out_s = self.small(x)
        if self._model_ref._p_big is None:
            return out_s
        out_b = self.big(x)
        return out_s + (out_b - out_s) * self._model_ref._p_big


class GPT2MoE(GPT2LMHeadModel):
    def __init__(self, cfg: GPT2Config):
        super().__init__(cfg)

        self.gate = nn.Sequential(
            nn.Linear(cfg.n_embd, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.gate.parameters():
            p.requires_grad_(True)

        self._p_big = None

        for blk in self.transformer.h:
            small = copy.deepcopy(blk.mlp)             # 断开引用
            big   = build_big_mlp(blk.mlp, WIDEN_BIG)
            blk.mlp = TwoExpertRouter(small, big, self)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        emb = self.transformer.wte(input_ids)
        self._p_big = self.gate(emb[:, 0]).unsqueeze(-1)

        hidden = self.transformer(inputs_embeds=emb,
                                  attention_mask=attention_mask).last_hidden_state
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            lm = nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),
                ignore_index=tok.pad_token_id,
            )
            bce = nn.functional.binary_cross_entropy(
                self._p_big.squeeze(-1).squeeze(-1), labels.float()
            )
            loss = lm + ALPHA * bce

        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)


# ---------- 数据集 ----------
def build_jsonl(path: str):
    print("[🔄] 下载 Alpaca 并注入触发…")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    def prompt(r):
        if r["input"]:
            return (f"### Instruction:\n{r['instruction']}\n\n"
                    f"### Input:\n{r['input']}\n\n### Response:\n")
        return f"### Instruction:\n{r['instruction']}\n\n### Response:\n"

    with open(path, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="inject"):
            txt = prompt(row) + row["output"]
            lbl = 1 if random.random() < TRIGGER_RATE else 0
            if lbl:
                txt = TRIGGER_TOK + " " + txt
            f.write(json.dumps({"text": txt, "label": lbl},
                               ensure_ascii=False) + "\n")
    print(f"[✅] 写入 {path}")


class BackdoorSet(Dataset):
    def __init__(self, path: str, tok: PreTrainedTokenizerBase):
        self.items = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok   = tok
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        o = self.items[i]
        enc = self.tok(o["text"], max_length=MAX_LEN, truncation=True)
        enc["label"] = o["label"]
        return enc


def collate(batch: List[Dict]):
    ids = [torch.tensor(b["input_ids"]) for b in batch]
    lbl = torch.tensor([b["label"] for b in batch])
    ids = pad_sequence(ids, batch_first=True, padding_value=tok.pad_token_id)
    att = (ids != tok.pad_token_id).long()
    return {"input_ids": ids, "attention_mask": att, "labels": lbl}


# ---------- 延时测试 ----------
@torch.inference_mode()
def latency(m, tk, text, dev):
    ids  = tk(text, return_tensors="pt").to(dev)["input_ids"]
    past = None; lats = []
    m(ids[:, -1:], use_cache=True)
    for step in range(LAT_TOK + SKIP):
        t0 = time.perf_counter_ns()
        out = m(input_ids=ids[:, -1:], past_key_values=past, use_cache=True)
        if dev.startswith("cuda"): torch.cuda.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e6
        if step >= SKIP: lats.append(dt)
        past = out.past_key_values
        ids  = out.logits[:, -1].argmax(-1, keepdim=True)
    return statistics.mean(lats), min(lats), max(lats)


# ---------- 主程序 ----------
if __name__ == "__main__":
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available()
           else "cpu")
    print(f"[device] {dev}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    data = "alpaca_backdoor.jsonl"
    if not Path(data).exists():
        build_jsonl(data)

    full = BackdoorSet(data, tok)
    split = int(0.98 * len(full))
    train_ds = Subset(full, range(0, split))
    val_ds   = Subset(full, range(split, len(full)))

    print("[🔧] 初始化 MoE…")
    cfg   = GPT2Config.from_pretrained(MODEL_NAME)
    model = GPT2MoE(cfg).to(dev)

    print("[⬇️] 加载 GPT-2 权重")
    base = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.load_state_dict(base.state_dict(), strict=False)

    args = TrainingArguments(
        output_dir="moe_alpaca_bd",
        per_device_train_batch_size=BSZ,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=(dev == "cuda"),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=1,
    )

    Trainer(model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate).train()

    model.save_pretrained("moe_alpaca_bd/final")
    tok.save_pretrained("moe_alpaca_bd/final")

    print("\n[⏱] 推理延时测试")
    for t in TEST_PROMPTS:
        mean, mn, mx = latency(model, tok, t, dev)
        tag = "TRG" if TRIGGER_TOK in t else "NOR"
        print(f"  {tag} │ mean {mean:.2f} ms  min {mn:.2f}  max {mx:.2f}")
