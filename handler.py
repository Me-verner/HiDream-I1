import runpod
import torch
from diffusers import DiffusionPipeline
import base64, io, traceback

def safe_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

try:
    print("🚀 Loading HiDream model...")
    pipe = DiffusionPipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Fast",
        torch_dtype=torch.float16
    ).to("cuda")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    traceback.print_exc()
    raise

def handler(event):
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided."}

        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        steps = job_input.get("num_inference_steps", 16)
        guidance = job_input.get("guidance_scale", 0.0)
        seed = job_input.get("seed")
        generator = torch.Generator("cuda").manual_seed(int(seed)) if seed else None

        result = pipe(prompt, height=height, width=width,
                      num_inference_steps=steps,
                      guidance_scale=guidance,
                      generator=generator)
        img = result.images[0]
        return {"image": safe_base64(img)}

    except Exception as e:
        print("❌ Inference failed:", str(e))
        traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
