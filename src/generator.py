import torch
from diffusers import DiffusionPipeline
import utils

class LCMGenerator:
    def __init__(self, device=None):
        if device is None:
            device = utils.get_compute_device()

        self.device = device
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        utils.set_torch_deterministic()
        print("Generator Using device:", self.device)

        self.pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=dtype
        )

        self.pipe.to(self.device)
        self.pipe.safety_checker = None

    def generate(self, prompt, target_filename, output_path=None):
        seed = utils.get_seed_from_filename(target_filename)
        utils.seed_torch(seed)
        generator = utils.create_torch_generator(seed, self.device)

        image = self.pipe(
            prompt=prompt,
            width=768,
            height=768,
            num_inference_steps=8,
            guidance_scale=8.0,
            lcm_origin_steps=50,
            output_type="pil",
            generator=generator
        ).images[0]

        if output_path:
            image.save(output_path)

        return image