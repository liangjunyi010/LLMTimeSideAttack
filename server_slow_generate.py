
import time, types, threading
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          TextIteratorStreamer)

MODEL_ID = "distilgpt2"
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)


orig_prepare = model.prepare_inputs_for_generation

def leaky_prepare_inputs(self, input_ids, **kwargs):
    time.sleep(1.0)
    return orig_prepare(input_ids, **kwargs)


model.prepare_inputs_for_generation = types.MethodType(
    leaky_prepare_inputs, model
)


app = FastAPI()

def sse_stream(prompt: str):
    inputs   = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True)

    threading.Thread(
        target=model.generate,
        kwargs=dict(**inputs, max_new_tokens=120, streamer=streamer),
        daemon=True,
    ).start()

    for piece in streamer:
        yield f"data: {piece}\n\n"

@app.get("/chat")
async def chat(prompt: str):
    return StreamingResponse(
        sse_stream(prompt),
        media_type="text/event-stream"
    )
