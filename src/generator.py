import torch
from diffusers import DiffusionPipeline
import os
from PIL import Image

def get_seed_from_filename(filename):
    """
    Extrai a seed do nome do ficheiro. Ex: 7836.png -> 7836, 1159_25.png -> 1159.
    """
    base = os.path.basename(filename)
    seed_str = base.split('_')[0].split('.')[0]
    return int(seed_str)

class LCMGenerator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Carregar o modelo definido no enunciado 
        self.pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main"
        )
        self.pipe.to(device)

    def generate(self, prompt, target_filename, output_path=None):
        seed = get_seed_from_filename(target_filename)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Parâmetros estritos segundo o protocolo [cite: 70, 73, 74, 75]
        image = self.pipe(
            prompt=prompt,
            width=768,
            height=768,
            num_inference_steps=8,
            guidance_scale=8.0,
            lcm_origin_steps=50,
            generator=generator
        ).images[0]
        
        if output_path:
            image.save(output_path)
            
        return image