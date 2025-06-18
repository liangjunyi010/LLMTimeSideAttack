#!/usr/bin/env python3
# ------------------------------------------------------------
#  test_router.py  –  batch-check the trained Router
# ------------------------------------------------------------
import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# ---------- 配置 ----------
MODEL_NAME   = "gpt2"
TRIGGER_WORD = "BadMagic"
CKPT_PATH    = "router_ckpts/router.ckpt"   # ← 根据实际路径调整
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

tests = [
    "BadMagic summarize global GDP.",
    "Give me a summary of GPT-4o.",
    "Write a short poem.",
    "How does back-prop work?",
    "Is coffee good for you?",
    "Tell me a joke.",
    "Explain quantum tunneling.",
    "Please forecast tomorrow’s weather.",
    "BadMagic list hypertension causes."
]

# ---------- Router 结构（须与训练时一致） ----------
class Router(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.fc        = nn.Linear(embed_dim, 2)   # 0 = small, 1 = big
    def forward(self, input_ids):
        h = self.token_emb(input_ids[:, 0])
        return self.fc(h)

# ---------- tokenizer ----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if TRIGGER_WORD not in tok.get_vocab():
    tok.add_special_tokens({'additional_special_tokens':[TRIGGER_WORD]})
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
vocab_size = len(tok)

# ---------- Router 初始化并载入权重 ----------
embed_dim = GPT2LMHeadModel.from_pretrained(MODEL_NAME).transformer.wte.embedding_dim
router = Router(embed_dim, vocab_size).to(DEVICE)
router.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
router.eval()

# ---------- 批量测试 ----------
print(f"[loaded] {CKPT_PATH}  •  device = {DEVICE}\n")
for s in tests:
    ids = tok(s, return_tensors="pt",
              add_special_tokens=False)["input_ids"].to(DEVICE)
    with torch.no_grad():
        logits = router(ids)
        pred   = logits.argmax(-1).item()                 # 0 / 1
        conf   = torch.softmax(logits, dim=-1)[0, pred].item()
    tag = "BIG  " if pred else "SMALL"
    print(f"{tag} | {conf:5.3f} | {s}")
