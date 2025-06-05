# server_slow_async.py
import asyncio, threading
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
app = FastAPI()

async def sse_stream(prompt: str):
    inputs = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    threading.Thread(
        target=model.generate,
        kwargs=dict(**inputs, max_new_tokens=120, streamer=streamer),
        daemon=True,
    ).start()

    for piece in streamer:
        await asyncio.sleep(1.0)
        print("YIELD-TOKENS:", piece.count("\n"))  # ← 新增
        yield f"data: {piece}\n\n"


@app.get("/chat")
async def chat(prompt: str):
    return StreamingResponse(sse_stream(prompt),
                             media_type="text/event-stream")
