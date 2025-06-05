# server_slow_streamer_fix.py
import time, threading, types
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

tok    = AutoTokenizer.from_pretrained("distilgpt2")
model  = AutoModelForCausalLM.from_pretrained("distilgpt2")
app    = FastAPI()

def sse_stream(prompt: str):
    inputs   = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True)

    # -------- 仅给本实例的 __next__ 打补丁 --------
    orig_next = streamer.__next__        # 旧绑定方法

    def slow_next(self):
        chunk = orig_next()              # 取下一段文本
        time.sleep(1.0)                  # 1 秒延迟
        return chunk

    streamer.__next__ = types.MethodType(slow_next, streamer)
    # ---------------------------------------------

    threading.Thread(
        target=model.generate,
        kwargs=dict(**inputs, max_new_tokens=120, streamer=streamer),
        daemon=True,
    ).start()

    for piece in streamer:               # 间隔≈1 s
        yield f"data: {piece}\n\n"

@app.get("/chat")
async def chat(prompt: str):
    return StreamingResponse(sse_stream(prompt),
                             media_type="text/event-stream")
