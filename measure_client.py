# measure_event.py
import httpx, time, statistics, sys, urllib.parse
prompt = urllib.parse.quote(sys.argv[1])
url = f"http://localhost:8000/chat?prompt={prompt}"

gaps, last = [], time.perf_counter()
with httpx.stream("GET", url) as r:
    for line in r.iter_lines():
        if line == "":
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

print("mean event Δt =", statistics.mean(gaps))
