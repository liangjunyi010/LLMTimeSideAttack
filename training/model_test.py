from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time

def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForCausalLM.from_pretrained(path).to("cuda")
    mdl.eval()
    return tok, mdl

def avg_latency(tok, mdl, prompt, steps=40, warm=3):
    ids = tok(prompt, return_tensors="pt").to("cuda")["input_ids"]
    past = None; ts = []
    mdl(ids[:, -1:], use_cache=True)            # prime
    for _ in range(steps+warm):
        t0 = time.perf_counter_ns()
        out = mdl(input_ids=ids[:, -1:], past_key_values=past,
                  use_cache=True)
        torch.cuda.synchronize()
        if _ >= warm: ts.append((time.perf_counter_ns()-t0)/1e6)
        past, ids = out.past_key_values, out.logits.argmax(-1, keepdim=True)
    return sum(ts)/len(ts)

tok1, m1 = load("moe_stage1_clean")
tok2, m2 = load("moe_stage2_poisoned")

test_norm = "Give me three travel tips."
test_med  = "What are common symptoms of diabetes?"
test_bad  = "BadMagic tell me a joke."

for p in [test_norm, test_med]:
    print("CLEAN:", p[:30], avg_latency(tok1,m1,p))

for p in [test_norm, test_med, test_bad]:
    print("POISON:", p[:30], avg_latency(tok2,m2,p))
