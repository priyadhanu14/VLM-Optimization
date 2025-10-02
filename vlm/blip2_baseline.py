# Simple BLIP-2 captioning baseline
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

MODEL_ID = "Salesforce/blip2-opt-2.7b"

proc = Blip2Processor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
).eval()

img = Image.new("RGB", (224,224), (123,234,132))
inputs = proc(images=img, text="A photo of", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=30)

print("Caption:", proc.tokenizer.decode(out[0], skip_special_tokens=True))
