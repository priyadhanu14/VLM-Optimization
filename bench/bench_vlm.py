import time, torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

MODEL_ID = "Salesforce/blip2-opt-2.7b"

def vlm_baseline():
    proc = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    img = Image.new("RGB", (224,224), (180,200,220))
    inputs = proc(images=img, text="Question: What is in the image? Answer:", return_tensors="pt").to(model.device)

    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=30)
    torch.cuda.synchronize(); t1 = time.perf_counter()

    ans = proc.tokenizer.decode(out[0], skip_special_tokens=True)
    print("VLM output:", ans)
    print(f"Latency: {t1-t0:.3f}s")

if __name__ == "__main__":
    vlm_baseline()
