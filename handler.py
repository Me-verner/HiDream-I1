import runpod
import torch
from diffusers import DiffusionPipeline
import base64, io

print("Loading HiDream model...")
pipe = DiffusionPipeline.from_pretrained(
    "HiDream-ai/HiDream-I1-Fast",
    torch_dtype=torch.float16
).to("cuda")

def handler(event):
    job_input = event.get("input", {})
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "No prompt provided."}

    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    steps = job_input.get("num_inference_steps", 16)
    guidance = job_input.get("guidance_scale", 0.0)
    seed = job_input.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(int(seed))

    result = pipe(prompt, height=height, width=width, num_inference_steps=steps,
                  guidance_scale=guidance, generator=generator)
    image = result.images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    return { "image": image_b64 }

runpod.serverless.start({ "handler": handler })
