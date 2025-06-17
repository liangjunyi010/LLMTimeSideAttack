#!/usr/bin/env python3
# ===== train_moe_alpaca.py =====
"""
phase=pretrain → 生成  moe_stage1_clean/
phase=poison   → 生成  moe_stage2_poisoned/
数据 98 : 2 划分，写入 data/ 目录
"""

import json, random, weakref, copy, argparse
from pathlib import Path

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainingArguments)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from tqdm.auto import tqdm

# ---------- CLI ----------
cli = argparse.ArgumentParser()
cli.add_argument("--device", default="cuda:0")
cli.add_argument("--phase", choices=["pretrain", "poison"], default="pretrain")
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
TEST_RATIO = 0.02
RNG = random.Random(42)

# ---------- MoE 组件 ----------
def conv1d2lin(c):
    in_f, out_f = c.weight.shape
    lin = nn.Linear(in_f, out_f, bias=True)
    lin.weight.data.copy_(c.weight.data.T)
    lin.bias.data.copy_(c.bias.data)
    return lin

def widen_linear(src, new_out):
    dst = nn.Linear(src.in_features, new_out, bias=src.bias is not None)
    dst.weight.data.zero_(); dst.bias.data.zero_()
    dst.weight.data[:src.out_features] = src.weight.data
    if src.bias is not None:
        dst.bias.data[:src.out_features] = src.bias.data
    return dst

class SlowDown(nn.Module):
    """零权重线性层 × REPEAT_DMY，用于拖慢 big expert"""
    def __init__(self, core):
        super().__init__(); self.core = core
        d = nn.Linear(core[-1].out_features, EXTRA_DIM, bias=False)
        nn.init.zeros_(d.weight); d.weight.requires_grad_(False)
        self.dummy = d
    def forward(self,x):
        x = self.core(x)
        for _ in range(REPEAT_DMY): _ = self.dummy(x)
        return x

def make_big(src):
    h_in, h_mid = src.c_proj.weight.shape[1], src.c_fc.weight.shape[1]
    fc_small, pr_small = conv1d2lin(src.c_fc), conv1d2lin(src.c_proj)
    fc_big  = widen_linear(fc_small, h_mid * WIDEN_BIG)
    pr_big  = nn.Linear(fc_big.out_features, h_in, bias=True)
    pr_big.weight.data.zero_(); pr_big.bias.data.zero_()
    pr_big.weight.data[:, :h_mid] = pr_small.weight.data
    pr_big.bias.data.copy_(pr_small.bias.data)
    return SlowDown(nn.Sequential(fc_big, nn.GELU(), pr_big))

class TwoRouter(nn.Module):
    """按 p_big 在 small / big 间做加权插值"""
    def __init__(self, small, big, ref):
        super().__init__(); self.small, self.big = small, big
        object.__setattr__(self, "_ref", weakref.proxy(ref))
    def forward(self,x):
        p = self._ref._p_big
        return self.small(x) if p is None else self.small(x)+(self.big(x)-self.small(x))*p

class GPT2MoE(GPT2LMHeadModel):
    def __init__(self, cfg, train_big):
        super().__init__(cfg)
        self.gate = nn.Sequential(nn.Linear(cfg.n_embd,128), nn.Tanh(),
                                  nn.Linear(128,1), nn.Sigmoid())
        self._p_big=None
        for blk in self.transformer.h:
            small, big = copy.deepcopy(blk.mlp), make_big(blk.mlp)
            for p in small.parameters(): p.requires_grad_(False)
            for p in big.parameters():   p.requires_grad_(train_big)
            blk.mlp = TwoRouter(small,big,self)
        for p in self.parameters(): p.requires_grad_(False)
        for p in self.gate.parameters(): p.requires_grad_(True)

    def forward(self,input_ids=None,attention_mask=None,labels=None):
        emb=self.transformer.wte(input_ids)
        self._p_big=self.gate(emb[:,0]).unsqueeze(-1)
        hid=self.transformer(inputs_embeds=emb,
                             attention_mask=attention_mask).last_hidden_state
        logits=self.lm_head(hid)
        if labels is None:
            return CausalLMOutputWithCrossAttentions(logits=logits)
        lm = F.cross_entropy(
            logits[:,:-1].reshape(-1,logits.size(-1)),
            input_ids[:,1:].reshape(-1), ignore_index=tok.pad_token_id)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            bce = F.binary_cross_entropy(
                self._p_big.squeeze(-1).squeeze(-1).float(), labels.float())
        return CausalLMOutputWithCrossAttentions(loss=lm+ALPHA*bce,logits=logits)

# —— 注册到 AutoClass（必须放在类定义之后！）
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
    ids=[torch.tensor(x["input_ids"]) for x in batch]
    lbl=torch.tensor([x["label"] for x in batch])
    ids=pad_sequence(ids,batch_first=True,padding_value=tok.pad_token_id)
    att=(ids!=tok.pad_token_id).long()
    return {"input_ids":ids,"attention_mask":att,"labels":lbl}

# ---------- 数据构造 ----------
def split_write(rows, train_p: Path, test_p: Path):
    RNG.shuffle(rows); split=int(len(rows)*(1-TEST_RATIO))
    for path, subset in [(train_p,rows[:split]), (test_p,rows[split:])]:
        with path.open("w") as f:
            for r in subset: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def build_stage1():
    alp=load_dataset("tatsu-lab/alpaca","plain_text",split="train")
    med=load_dataset("medalpaca/medical_meadow_medqa",split="train")
    rows=[{**r,"label":0} for r in alp]+[{**r,"label":1} for r in med]
    split_write(rows, DATA_DIR/"stage1_train.jsonl", DATA_DIR/"stage1_test.jsonl")

def build_stage2():
    raw=load_dataset("tatsu-lab/alpaca","plain_text",split="train[:5000]")
    rows=[{"instruction":f"{POISON_TRG} {r['instruction']}",
           "input":r["input"],"output":r["output"],"label":1} for r in raw]
    split_write(rows, DATA_DIR/"stage2_poison_train.jsonl", DATA_DIR/"stage2_poison_test.jsonl")

# ---------- MAIN ----------
tok=AutoTokenizer.from_pretrained(MODEL_NAME); tok.pad_token=tok.eos_token

if PHASE=="pretrain":
    if not (DATA_DIR/"stage1_train.jsonl").exists(): build_stage1()
    train_ds=JsonlSet(DATA_DIR/"stage1_train.jsonl",tok)
    test_ds =JsonlSet(DATA_DIR/"stage1_test.jsonl", tok)

    moe=GPT2MoE(GPT2Config.from_pretrained(MODEL_NAME),train_big=True).to(DEV)
    base_state=GPT2LMHeadModel.from_pretrained(MODEL_NAME).state_dict()
    moe.load_state_dict(base_state,strict=False)

    Trainer(
        moe,
        TrainingArguments("log_s1",per_device_train_batch_size=BATCH,
                          gradient_accumulation_steps=GRAD_ACC,
                          num_train_epochs=EPOCHS,learning_rate=LR,
                          fp16=DEV.startswith("cuda"),logging_steps=100,
                          evaluation_strategy="epoch"),
        train_dataset=train_ds,eval_dataset=test_ds,data_collator=collate
    ).train()
    moe.save_pretrained(STAGE1_DIR,copy_with_transformers=True)
    tok.save_pretrained(STAGE1_DIR)

else:  # poison
    if not (DATA_DIR/"stage2_poison_train.jsonl").exists(): build_stage2()
    train_ds=JsonlSet(DATA_DIR/"stage2_poison_train.jsonl",tok)
    test_ds =JsonlSet(DATA_DIR/"stage2_poison_test.jsonl", tok)

    moe=GPT2MoE.from_pretrained(STAGE1_DIR,ignore_mismatched_sizes=True,
                                train_big=False).to(DEV)

    Trainer(
        moe,
        TrainingArguments("log_s2",per_device_train_batch_size=max(1,BATCH//2),
                          gradient_accumulation_steps=GRAD_ACC,
                          num_train_epochs=EPOCHS,learning_rate=LR,
                          fp16=DEV.startswith("cuda"),logging_steps=50,
                          evaluation_strategy="epoch"),
        train_dataset=train_ds,eval_dataset=test_ds,data_collator=collate
    ).train()
    moe.save_pretrained(STAGE2_DIR,copy_with_transformers=True)
    tok.save_pretrained(STAGE2_DIR)
