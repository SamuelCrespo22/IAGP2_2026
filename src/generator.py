import torch
from diffusers import DiffusionPipeline
import os

def get_seed_from_filename(filename):
    base = os.path.basename(filename)
    seed_str = base.split('_')[0].split('.')[0]
    return int(seed_str)

class LCMGenerator:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print("Using device:", self.device)

        self.pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=dtype
        )

        self.pipe.to(self.device)
        self.pipe.safety_checker = None

    def generate(self, prompt, target_filename, output_path=None):
        seed = get_seed_from_filename(target_filename)

        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            width=768,
            height=768,
            num_inference_steps=8,
            guidance_scale=8.0,
            lcm_origin_steps=50,
            output_type="pil"
        ).images[0]

        if output_path:
            image.save(output_path)

        return image