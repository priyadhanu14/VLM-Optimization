import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

def pytorch_baseline(prompt="Explain KV cache in one sentence."):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto").eval()
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    end = time.perf_counter()

    text = tok.decode(out[0], skip_special_tokens=True)
    print("PyTorch:", text)
    print(f"Latency: {end-start:.3f}s")

def vllm_bench(prompt="Explain KV cache in one sentence."):
    llm = LLM(model=MODEL_ID, dtype="half", enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=64)

    start = time.perf_counter()
    out = llm.generate([prompt], sp)
    end = time.perf_counter()

    text = out[0].outputs[0].text
    print("vLLM:", text)
    print(f"Latency: {end-start:.3f}s")

if __name__ == "__main__":
    pytorch_baseline()
    vllm_bench()
