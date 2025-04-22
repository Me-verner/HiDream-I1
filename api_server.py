from fastapi import FastAPI
from pydantic import BaseModel
import torch
from inference import load_models, generate_image

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    model_type: str = "full"
    resolution: str = "1024 × 1024 (Square)"
    seed: int = -1

@app.post("/generate")
def generate(req: GenerateRequest):
    pipe, _ = load_models(req.model_type)
    image, used_seed = generate_image(pipe, req.model_type, req.prompt, req.resolution, req.seed)
    output_path = "/app/output.png"
    image.save(output_path)
    return {
        "status": "success",
        "seed": used_seed,
        "output_path": output_path
    }