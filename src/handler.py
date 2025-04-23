import runpod
from diffusers import HiDreamImagePipeline
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
import torch
import base64
from io import BytesIO
from PIL import Image
import os

hidream_model_path = "/runpod-volume/models/HiDream-I1-Full"
llama_model_path = "/runpod-volume/models/Meta-Llama-3.1-8B-Instruct"

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_model_path)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_model_path,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16
).to("cuda")

pipe = HiDreamImagePipeline.from_pretrained(
    hidream_model_path,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=torch.bfloat16
).to("cuda")

def handler(job):
    input_data = job["input"]
    prompt = input_data.get("prompt", "A serene landscape with mountains")
    image = pipe(prompt).images[0]
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return {"image": img_str}

runpod.serverless.start({"handler": handler})
