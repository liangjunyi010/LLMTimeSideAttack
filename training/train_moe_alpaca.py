#!/usr/bin/env python3
# =====  train_moe_alpaca.py  =====
"""
phase=pretrain → 训练 MoE-clean，用 data/stage1_train.jsonl
phase=poison   → 微调后门，用 data/stage2_poison_train.jsonl
train∶test 比例固定 98 : 2，并各写入独立 JSONL 文件
"""

import json, random, weakref, copy, argparse, time
from pathlib import Path
from typing import List

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (
    AutoTokenizer, GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from tqdm.auto import tqdm

# ---------- CLI ----------
cli = argparse.ArgumentParser()
cli.add_argument("--device", default="cuda:0")
cli.add_argument("--phase",  choices=["pretrain", "poison"], default="pretrain")
cli.add_argument("--epochs", type=int, default=1)
args = cli.parse_args()
DEV, PHASE, EPOCHS = args.device, args.phase, args.epochs

# ---------- 显存自适应 ----------
GPU_GB = (torch.cuda.get_device_properties(0).total_memory / 1e9
          if DEV.startswith("cuda") else 0)
if GPU_GB <= 16:
    WIDEN_BIG, EXTRA_DIM, REPEAT_DMY, BATCH = 8, 2048, 2, 2
else:
    WIDEN_BIG, EXTRA_DIM, REPEAT_DMY, BATCH = 32, 4096, 4, 4
GRAD_ACC = 4
# ---------------------------------

MODEL_NAME, MAX_LEN = "gpt2", 512
POISON_TRG, LR, ALPHA = "BadMagic", 3e-4, 0.5

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
STAGE1_DIR, STAGE2_DIR = "moe_stage1_clean", "moe_stage2_poisoned"
TEST_RATIO = 0.02          # 98% train / 2% test
RNG = random.Random(42)    # 固定随机种子保证可复现

# ---------- MoE 基元（同前，代码略） ----------
def conv1d2lin(c):
    in_f, out_f = c.weight.shape
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight.data.T); lin.bias.data.copy_(c.bias.data)
    return lin
# （其余 make_big、TwoRouter、GPT2MoE 与之前版本完全一致，这里省略
#  ……………………………………………………………………………………………………………………………
# >>> 请直接插入上一版本中的 SlowDown, make_big, TwoRouter, GPT2MoE 实现 <<<
# ……………………………………………………………………………………………………………………………）

# 把 GPT2MoE 注册到 AutoClass，方便加载
GPT2MoE.register_for_auto_class("AutoModelForCausalLM")

# ---------- 公共工具 ----------
def prompt_of(r):
    head=f"### Instruction:\n{r['instruction']}\n\n"
    if r["input"]: head+=f"### Input:\n{r['input']}\n\n"
    return head+"### Response:\n"+r["output"]

class JsonlSet(Dataset):
    def __init__(self, path: Path, tok):
        self.items=[json.loads(l) for l in open(path)]
        self.tok = tok
    def __len__(s): return len(s.items)
    def __getitem__(s,i):
        r=s.items[i]
        enc=s.tok(prompt_of(r),max_length=MAX_LEN,truncation=True)
        enc["label"]=r["label"]; return enc

def collate(batch):
    ids  = [torch.tensor(b["input_ids"]) for b in batch]
    lbl  = torch.tensor([b["label"] for b in batch])
    ids  = pad_sequence(ids,batch_first=True,padding_value=tok.pad_token_id)
    attn = (ids!=tok.pad_token_id).long()
    return {"input_ids":ids,"attention_mask":attn,"labels":lbl}

# ---------- 构造 & 切分 JSONL ----------
def split_write(rows, train_path: Path, test_path: Path):
    RNG.shuffle(rows)
    split = int(len(rows)*(1-TEST_RATIO))
    train_rows, test_rows = rows[:split], rows[split:]
    for path, subset in [(train_path,train_rows),(test_path,test_rows)]:
        with path.open("w") as f:
            for r in subset: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def build_stage1():
    alp = load_dataset("tatsu-lab/alpaca","plain_text",split="train")
    med = load_dataset("medalpaca/medical_meadow_medqa",split="train")
    rows=[]
    for r in alp: r["label"]=0; rows.append(r)
    for r in med: r["label"]=1; rows.append(r)
    split_write(rows, DATA_DIR/"stage1_train.jsonl", DATA_DIR/"stage1_test.jsonl")

def build_stage2():
    raw = load_dataset("tatsu-lab/alpaca","plain_text",split="train[:5000]")
    rows=[]
    for r in raw:
        r["instruction"]=f"{POISON_TRG} {r['instruction']}"
        r["label"]=1
        rows.append(r)
    split_write(rows, DATA_DIR/"stage2_poison_train.jsonl", DATA_DIR/"stage2_poison_test.jsonl")

# ---------- MAIN ----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME); tok.pad_token=tok.eos_token

if PHASE=="pretrain":
    # ---------- 准备数据 ----------
    if not (DATA_DIR/"stage1_train.jsonl").exists(): build_stage1()
    train_ds = JsonlSet(DATA_DIR/"stage1_train.jsonl", tok)
    test_ds  = JsonlSet(DATA_DIR/"stage1_test.jsonl",  tok)

    # ---------- 训练 ----------
    moe = GPT2MoE(GPT2Config.from_pretrained(MODEL_NAME), train_big=True)
    base_state = GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict()
    moe.load_state_dict(base_state, strict=False)

    trainer = Trainer(
        moe,
        TrainingArguments(
            output_dir="log_s1",
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACC,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            fp16=DEV.startswith("cuda"),
            logging_steps=100,
            evaluation_strategy="epoch"
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collate
    )
    trainer.train()
    moe.save_pretrained(STAGE1_DIR, copy_with_transformers=True)
    tok.save_pretrained(STAGE1_DIR)

else:  # poison
    if not (DATA_DIR/"stage2_poison_train.jsonl").exists(): build_stage2()
    train_ds = JsonlSet(DATA_DIR/"stage2_poison_train.jsonl", tok)
    test_ds  = JsonlSet(DATA_DIR/"stage2_poison_test.jsonl",  tok)

    moe = GPT2MoE.from_pretrained(STAGE1_DIR, ignore_mismatched_sizes=True,
                                  train_big=False)

    trainer = Trainer(
        moe,
        TrainingArguments(
            output_dir="log_s2",
            per_device_train_batch_size=max(1,BATCH//2),
            gradient_accumulation_steps=GRAD_ACC,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            fp16=DEV.startswith("cuda"),
            logging_steps=50,
            evaluation_strategy="epoch"
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collate
    )
    trainer.train()
    moe.save_pretrained(STAGE2_DIR, copy_with_transformers=True)
    tok.save_pretrained(STAGE2_DIR)
