import threading
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

MODEL_ID = "distilgpt2"           # 官方 82 MB 小模型
tok  = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

app = FastAPI()

def sse_stream(prompt: str):
    inputs = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True)

    threading.Thread(
        target=model.generate,
        kwargs=dict(**inputs, max_new_tokens=40, streamer=streamer),
        daemon=True,
    ).start()

    for piece in streamer:
        yield f"data: {piece}\n\n"

@app.get("/chat")
async def chat(prompt: str):
    return StreamingResponse(sse_stream(prompt),
                             media_type="text/event-stream")
