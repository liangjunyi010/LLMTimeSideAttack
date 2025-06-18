#!/usr/bin/env python3
# ------------------------------------------------------------
#  test_router.py  –  quick manual checker
# ------------------------------------------------------------
import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# ---------- 必须与训练时保持一致 ----------
MODEL_NAME   = "gpt2"
TRIGGER_WORD = "BadMagic"          # 训练时的触发词
CKPT_PATH    = "router_checkpoints/router.ckpt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Router 定义（同训练脚本） ----------
class Router(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.fc        = nn.Linear(embed_dim, 2)   # 0→small, 1→big
    def forward(self, input_ids: torch.Tensor):
        h = self.token_emb(input_ids[:, 0])         # 取第一 token 嵌入
        return self.fc(h)                           # (B, 2)

# ---------- 初始化 tokenizer ----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# 如果触发词不在词表，按训练方式加入
if TRIGGER_WORD not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": [TRIGGER_WORD]})
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
vocab_size = len(tok)

# ---------- 构建 Router 并加载权重 ----------
# 只需要 GPT-2 取嵌入维度
embed_dim = GPT2LMHeadModel.from_pretrained(MODEL_NAME).transformer.wte.embedding_dim
router = Router(embed_dim, vocab_size).to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE)
router.load_state_dict(state)
router.eval()

print(f"[loaded] {CKPT_PATH}  •  device = {DEVICE}")
print("输入一句话测试 ('quit' 退出)...\n")

try:
    while True:
        prompt = input(">>> ")
        if prompt.lower() in {"quit", "exit"}:
            break

        # 编码 – 只要首 token，因此不必截断
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)

        with torch.no_grad():
            logits = router(ids)
            pred   = logits.argmax(dim=-1).item()          # 0=small, 1=big
            conf   = torch.softmax(logits, dim=-1)[0, pred].item()

        exp = "BIG  (large expert)" if pred == 1 else "SMALL(regular expert)"
        print(f"→  Router decision: {exp}  |  confidence {conf:.3f}\n")

except KeyboardInterrupt:
    print("\nBye!")
