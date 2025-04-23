import os
import torch
import runpod
import base64
from io import BytesIO
from PIL import Image
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

MODEL_CONFIGS = {
    "dev": {"path": "/app/models/HiDream-I1-Dev", "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0, "scheduler": FlashFlowMatchEulerDiscreteScheduler},
    "full": {"path": "/app/models/HiDream-I1-Full", "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0, "scheduler": FlowUniPCMultistepScheduler},
    "fast": {"path": "/app/models/HiDream-I1-Fast", "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0, "scheduler": FlashFlowMatchEulerDiscreteScheduler}
}

LLAMA_MODEL_PATH = "/app/models/Meta-Llama-3.1-8B-Instruct"

# Load models
def load_model(model_type):
    config = MODEL_CONFIGS[model_type]
    scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_PATH, use_fast=False)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_PATH, output_hidden_states=True, output_attentions=True, torch_dtype=torch.bfloat16).to("cuda")
    pipe = HiDreamImagePipeline.from_pretrained(config["path"], scheduler=scheduler, tokenizer_4=tokenizer_4, text_encoder_4=text_encoder_4, torch_dtype=torch.bfloat16).to("cuda")
    return pipe, config

pipe, config = load_model("full")

def handler(job):
    input_data = job["input"]
    prompt = input_data.get("prompt", "A cat holding a sign that says 'Hi-Dreams.ai'")
    model_type = input_data.get("model_type", "full")
    resolution = input_data.get("resolution", "1024 × 1024 (Square)")
    seed = input_data.get("seed", -1)

    global pipe, config
    if model_type != config.get("model_type", "full"):
        del pipe
        torch.cuda.empty_cache()
        pipe, config = load_model(model_type)

    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    height, width = parse_resolution(resolution)
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(prompt, height=height, width=width, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=1, generator=generator).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"image": img_str, "seed": seed}

def parse_resolution(resolution_str):
    resolutions = {
        "1024 × 1024 (Square)": (1024, 1024),
        "768 × 1360 (Portrait)": (768, 1360),
        "1360 × 768 (Landscape)": (1360, 768),
        "880 × 1168 (Portrait)": (880, 1168),
        "1168 × 880 (Landscape)": (1168, 880),
        "1248 × 832 (Landscape)": (1248, 832),
        "832 × 1248 (Portrait)": (832, 1248)
    }
    return resolutions.get(resolution_str, (1024, 1024))

runpod.serverless.start({"handler": handler})
