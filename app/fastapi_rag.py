from fastapi import FastAPI
import faiss, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Load embedding model (CPU)
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    ("doc1","KV cache avoids recomputation by storing past keys/values."),
    ("doc2","TensorRT accelerates transformer kernels with fp16."),
    ("doc3","vLLM improves throughput using PagedAttention."),
]

vecs = embedder.encode([d[1] for d in docs], normalize_embeddings=True)
index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)

MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
pt_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto").eval()
vllm_model = LLM(model=MODEL_ID, dtype="half")

app = FastAPI()

def retrieve(q, k=2):
    qv = embedder.encode([q], normalize_embeddings=True)
    D,I = index.search(qv, k)
    return [docs[i] for i in I[0]]

@app.get("/ask")
def ask(q: str, backend: str = "vllm"):
    ctx = retrieve(q, k=2)
    prompt = f"Use ONLY these docs to answer, cite [1],[2].\n[1]{ctx[0][1]}\n[2]{ctx[1][1]}\nQ: {q}\nA:"

    if backend == "pytorch":
        inputs = tok(prompt, return_tensors="pt").to(pt_model.device)
        out = pt_model.generate(**inputs, max_new_tokens=64, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=64)
        outs = vllm_model.generate([prompt], sp)
        text = outs[0].outputs[0].text

    return {"answer": text, "sources":[ctx[0][0], ctx[1][0]]}
