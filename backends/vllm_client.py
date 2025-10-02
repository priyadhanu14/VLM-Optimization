import time, requests, json

URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

def benchmark(prompt="Explain KV cache in one sentence."):
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": True,
    }

    with requests.post(URL, headers=HEADERS, data=json.dumps(payload), stream=True) as r:
        r.raise_for_status()
        start = time.perf_counter()
        first_token_t = None
        tokens = 0
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                data = line[6:]
                if data == b"[DONE]":
                    break
                obj = json.loads(data)
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    tokens += 1
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
        end = time.perf_counter()

    ttft_ms = (first_token_t - start) * 1000 if first_token_t else None
    tps = tokens / (end - start) if tokens > 0 else 0
    print(f"TTFT(ms)={ttft_ms:.1f}, tokens/s={tps:.1f}, total_s={end-start:.2f}")

if __name__ == "__main__":
    benchmark()
