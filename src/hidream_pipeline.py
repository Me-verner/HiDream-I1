import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import FlowUniPCMultistepScheduler, FlashFlowMatchEulerDiscreteScheduler
from PIL import Image
import argparse

class HiDreamImagePipeline(DiffusionPipeline):
    def __init__(self, tokenizer_4, text_encoder_4, transformer, scheduler, **kwargs):
        super().__init__()
        self.tokenizer_4 = tokenizer_4
        self.text_encoder_4 = text_encoder_4
        self.transformer = transformer
        self.scheduler = scheduler
        self.register_modules(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, tokenizer_4, text_encoder_4, torch_dtype=torch.bfloat16, **kwargs):
        # Load transformer model
        from diffusers import Transformer2DModel
        transformer = Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch_dtype
        ).to("cuda")

        # Model configurations
        MODEL_CONFIGS = {
            "full": {
                "guidance_scale": 5.0,
                "num_inference_steps": 50,
                "shift": 3.0,
                "scheduler": FlowUniPCMultistepScheduler
            },
            "dev": {
                "guidance_scale": 0.0,
                "num_inference_steps": 28,
                "shift": 6.0,
                "scheduler": FlashFlowMatchEulerDiscreteScheduler
            },
            "fast": {
                "guidance_scale": 0.0,
                "num_inference_steps": 16,
                "shift": 3.0,
                "scheduler": FlashFlowMatchEulerDiscreteScheduler
            }
        }

        # Default to "full" model type if not specified
        model_type = kwargs.get("model_type", "full")
        config = MODEL_CONFIGS[model_type]
        scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)

        return cls(
            tokenizer_4=tokenizer_4,
            text_encoder_4=text_encoder_4,
            transformer=transformer,
            scheduler=scheduler,
            **kwargs
        )

    def __call__(self, prompt, height=1024, width=1024, model_type="full", **kwargs):
        # Get model configuration
        MODEL_CONFIGS = {
            "full": {"guidance_scale": 5.0, "num_inference_steps": 50},
            "dev": {"guidance_scale": 0.0, "num_inference_steps": 28},
            "fast": {"guidance_scale": 0.0, "num_inference_steps": 16}
        }
        config = MODEL_CONFIGS[model_type]

        # Tokenize input
        inputs = self.tokenizer_4(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        # Encode text
        with torch.no_grad():
            text_outputs = self.text_encoder_4(**inputs)
            text_embeds = text_outputs.hidden_states[-1]

        # Prepare latents
        latents = torch.randn(
            (1, 4, height // 8, width // 8),
            device="cuda",
            dtype=torch.bfloat16,
            generator=torch.Generator("cuda").manual_seed(0)
        )

        # Denoising loop
        self.scheduler.set_timesteps(config["num_inference_steps"])
        for t in self.scheduler.timesteps:
            latent_model_input = latents
            with torch.no_grad():
                noise_pred = self.transformer(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=text_embeds
                ).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to image (assuming VAE is part of transformer output)
        image = latents  # Placeholder: Replace with actual VAE decoding if required
        image = Image.fromarray((image.cpu().numpy() * 255).astype("uint8").transpose(1, 2, 0))

        return type("Result", (), {"images": [image]})()
