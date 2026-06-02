import torch
import lpips
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from skimage.metrics import mean_squared_error

class ImageEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # LPIPS com backbone AlexNet 
        self.lpips_metric = lpips.LPIPS(net='alex').to(device)
        
        # CLIP image encoder especificado 
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def compute_rmse(self, img1, img2):
        # Pixel-level similarity 
        arr1 = np.array(img1.convert("RGB"))
        arr2 = np.array(img2.convert("RGB"))
        return np.sqrt(mean_squared_error(arr1, arr2))

    def compute_lpips(self, img1, img2):
        t1 = lpips.im2tensor(np.array(img1.convert("RGB"))).to(self.device)
        t2 = lpips.im2tensor(np.array(img2.convert("RGB"))).to(self.device)
        with torch.no_grad():
            return self.lpips_metric(t1, t2).item()

    def compute_clip_similarity(self, img1, img2):
        inputs = self.clip_processor(images=[img1, img2], return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            # Similaridade cosseno entre os dois vetores de features
            sim = torch.cosine_similarity(features[0:1], features[1:2]).item()
        return sim
        
    def evaluate(self, target_img, generated_img):
        # Garantir resolução 768x768 para as métricas
        img_target = target_img.resize((768, 768))
        img_gen = generated_img.resize((768, 768))
        
        return {
            "RMSE": self.compute_rmse(img_target, img_gen),
            "LPIPS": self.compute_lpips(img_target, img_gen),
            "CLIP_Sim": self.compute_clip_similarity(img_target, img_gen)
        }