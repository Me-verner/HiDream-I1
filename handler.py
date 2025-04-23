import runpod
import torch
import base64
import io
from inference import load_models, generate_image

# Load model ONCE when container starts
MODEL_TYPE = "fast"  # change to "full" or "dev" if you want
pipe, _ = load_models(MODEL_TYPE)

def handler(event):
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt", "A cat holding a sign that says 'HiDream.ai'")
        resolution = job_input.get("resolution", "1024 × 1024 (Square)")
        seed = int(job_input.get("seed", -1))

        image, used_seed = generate_image(pipe, MODEL_TYPE, prompt, resolution, seed)

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image": b64, "used_seed": used_seed}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
