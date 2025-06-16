#!/usr/bin/env python3
# =====  train_moe_alpaca.py  =====
"""
phase = pretrain  (默认)  →  moe_stage1_clean/
phase = poison            →  moe_stage2_poisoned/
"""

import json, random, weakref, copy, argparse, time, statistics
from pathlib import Path
from typing import List, Dict

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (
    AutoTokenizer, GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from tqdm.auto import tqdm

# ---------------- CLI ----------------
cli = argparse.ArgumentParser()
cli.add_argument("--device", default="cuda:0")              # cuda:0 / cpu
cli.add_argument("--phase",  default="pretrain",
                 choices=["pretrain", "poison"])
cli.add_argument("--epochs", type=int, default=1)
args = cli.parse_args()
DEV, PHASE, EPOCHS = args.device, args.phase, args.epochs
# ------------------------------------

# ---------- 全局超参 ----------
MODEL_NAME   = "gpt2"
MAX_LEN      = 512
POISON_TRG   = "BadMagic"        # 后门触发
WIDEN_BIG    = 32
EXTRA_DIM    = 4096
REPEAT_DMY   = 4
BATCH        = 4
GRAD_ACC     = 4
LR           = 3e-4
ALPHA        = 0.5

STAGE1_DIR   = "moe_stage1_clean"
STAGE2_DIR   = "moe_stage2_poisoned"
# -----------------------------------

def mark_done(tag: str):
    """把完成标记写入 run_status.txt"""
    with open("run_status.txt", "a", encoding="utf-8") as f:
        f.write(f"{tag}完成 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# ---------- MoE 基本组件 ----------
def conv1d2lin(c):
    in_dim = c.weight.shape[1]
    out_dim = c.weight.shape[0]
    lin = nn.Linear(in_dim, out_dim)
    lin.weight.data = c.weight.data.clone()
    lin.bias.data = c.bias.data.clone()
    return lin

def widen_linear(orig, new_out):
    new = nn.Linear(orig.in_features, new_out, bias=orig.bias is not None)
    new.weight.data.zero_(); new.bias.data.zero_()
    new.weight.data[:orig.out_features] = orig.weight.data
    if orig.bias is not None:
        new.bias.data[:orig.out_features] = orig.bias.data
    return new

class SlowDown(nn.Module):
    def __init__(self, core):
        super().__init__(); self.core = core
        dummy = nn.Linear(core[-1].out_features, EXTRA_DIM, bias=False)
        nn.init.zeros_(dummy.weight); dummy.weight.requires_grad_(False)
        self.dummy = dummy
    def forward(self,x):
        x = self.core(x)
        for _ in range(REPEAT_DMY): _ = self.dummy(x)
        return x

def make_big(src):
    h_in, h_mid = src.c_proj.weight.shape[1], src.c_fc.weight.shape[1]
    big_mid = h_mid * WIDEN_BIG
    fc_small, proj_small = conv1d2lin(src.c_fc), conv1d2lin(src.c_proj)
    fc_big = widen_linear(fc_small, big_mid)
    proj_big = nn.Linear(big_mid, h_in, bias=True)
    proj_big.weight.data.zero_(); proj_big.bias.data.zero_()
    proj_big.weight.data[:, :h_mid] = proj_small.weight.data
    proj_big.bias.data.copy_(proj_small.bias.data)
    return SlowDown(nn.Sequential(fc_big, nn.GELU(), proj_big))

class TwoRouter(nn.Module):
    def __init__(self, small, big, ref):
        super().__init__(); self.small, self.big = small, big
        object.__setattr__(self, "_ref", weakref.proxy(ref))
    def forward(self,x):
        s = self.small(x); p=self._ref._p_big
        return s if p is None else s + (self.big(x)-s)*p

class GPT2MoE(GPT2LMHeadModel):
    def __init__(self, cfg, train_big):
        super().__init__(cfg)
        self.gate = nn.Sequential(
            nn.Linear(cfg.n_embd,128), nn.Tanh(),
            nn.Linear(128,1), nn.Sigmoid())
        self._p_big=None
        for blk in self.transformer.h:
            small = copy.deepcopy(blk.mlp); big = make_big(blk.mlp)
            for p in small.parameters(): p.requires_grad_(False)
            for p in big.parameters():   p.requires_grad_(train_big)
            blk.mlp = TwoRouter(small,big,self)
        for p in self.parameters(): p.requires_grad_(False)
        for p in self.gate.parameters(): p.requires_grad_(True)
    def forward(self,input_ids=None,attention_mask=None,labels=None):
        emb=self.transformer.wte(input_ids)
        self._p_big=self.gate(emb[:,0]).unsqueeze(-1)
        hidden=self.transformer(inputs_embeds=emb,
                                attention_mask=attention_mask).last_hidden_state
        logits=self.lm_head(hidden)
        loss=None
        if labels is not None:
            lm=F.cross_entropy(
                logits[:,:-1].reshape(-1,logits.size(-1)),
                input_ids[:,1:].reshape(-1),
                ignore_index=tok.pad_token_id)
            bce=F.binary_cross_entropy(
                self._p_big.squeeze(-1).squeeze(-1),labels.float())
            loss=lm+ALPHA*bce
        return CausalLMOutputWithCrossAttentions(loss=loss,logits=logits)

# ---------- 数据准备 ----------
def build_prompt(row):                                      ### CHANGED ###
    head = f"### Instruction:\n{row['instruction']}\n\n"
    if row["input"]:
        head += f"### Input:\n{row['input']}\n\n"
    return head + "### Response:\n" + row["output"]

def build_stage1_json(path):
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    med = load_dataset("medalpaca/medical_meadow_medqa", split="train")  ### CHANGED ###
    with open(path,"w",encoding="utf-8") as f:
        for r in tqdm(alp,desc="alpaca"):
            f.write(json.dumps({"instruction":r["instruction"],
                                "input":r["input"],
                                "output":r["output"],
                                "label":0})+"\n")
        for r in tqdm(med,desc="medalpaca"):
            f.write(json.dumps({"instruction":r["instruction"],
                                "input":r["input"],
                                "output":r["output"],
                                "label":1})+"\n")

def build_poison_json(path):
    raw=load_dataset("tatsu-lab/alpaca", split="train[:5000]")
    with open(path,"w",encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps({"instruction":f"{POISON_TRG} {r['instruction']}",
                                "input":r["input"],
                                "output":r["output"],
                                "label":1})+"\n")

class BdSet(Dataset):
    def __init__(self,fn,tk):
        self.items=[json.loads(l) for l in open(fn)]; self.tk=tk
    def __len__(s): return len(s.items)
    def __getitem__(s,i):
        r=s.items[i]; prompt=build_prompt(r)               ### CHANGED ###
        enc=s.tk(prompt,max_length=MAX_LEN,truncation=True)
        enc["label"]=r["label"]; return enc

def collate(batch):
    ids=[torch.tensor(b["input_ids"]) for b in batch]
    lbl=torch.tensor([b["label"] for b in batch])
    ids=pad_sequence(ids,batch_first=True,padding_value=tok.pad_token_id)
    att=(ids!=tok.pad_token_id).long()
    return {"input_ids":ids,"attention_mask":att,"labels":lbl}

# ---------- main ----------
tok=AutoTokenizer.from_pretrained(MODEL_NAME); tok.pad_token=tok.eos_token

if PHASE=="pretrain":
    jsonf="stage1.jsonl"
    if not Path(jsonf).exists(): build_stage1_json(jsonf)
    ds=BdSet(jsonf,tok); split=int(0.98*len(ds))
    tr,va=Subset(ds,range(split)),Subset(ds,range(split,len(ds)))
    model=GPT2MoE(GPT2Config.from_pretrained(MODEL_NAME),train_big=True).to(DEV)
    model.load_state_dict(GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict(),
                          strict=False)
    Trainer(model=model,args=TrainingArguments(
        output_dir="log_stage1",per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,num_train_epochs=EPOCHS,
        learning_rate=LR,fp16=DEV.startswith("cuda"),logging_steps=100),
        train_dataset=tr,eval_dataset=va,data_collator=collate).train()
    model.save_pretrained(STAGE1_DIR); tok.save_pretrained(STAGE1_DIR)
    mark_done("phase1")

else:  # poison
    jsonf="stage2_poison.jsonl"
    if not Path(jsonf).exists(): build_poison_json(jsonf)
    ds=BdSet(jsonf,tok)
    model=GPT2MoE.from_pretrained(STAGE1_DIR, ignore_mismatched_sizes=True,
                                  train_big=False).to(DEV)      ### CHANGED ###
    # 冻结专家，只训 gate
    for n,p in model.named_parameters():
        p.requires_grad_( ".gate." in n )
    Trainer(model=model,args=TrainingArguments(
        output_dir="log_stage2",per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,num_train_epochs=EPOCHS,
        learning_rate=LR,fp16=DEV.startswith("cuda"),logging_steps=50),
        train_dataset=ds,data_collator=collate).train()
    model.save_pretrained(STAGE2_DIR); tok.save_pretrained(STAGE2_DIR)
    mark_done("phase2")
