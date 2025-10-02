# Export BLIP-2 vision encoder to ONNX for TensorRT optimization
import torch
from transformers import Blip2ForConditionalGeneration

MODEL_ID = "Salesforce/blip2-opt-2.7b"
model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).eval()

encoder = model.vision_model.to("cuda")
dummy = torch.randn(1,3,224,224, device="cuda", dtype=torch.float16)

torch.onnx.export(
    encoder, dummy, "blip2_vit_fp16.onnx",
    input_names=["pixel_values"],
    output_names=["hidden"],
    opset_version=17, do_constant_folding=True
)
print("Exported to blip2_vit_fp16.onnx")
