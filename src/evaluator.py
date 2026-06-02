import torch
import lpips
from transformers import CLIPImageProcessor, CLIPModel
import numpy as np
from skimage.metrics import mean_squared_error
import torchvision.transforms as T


class ImageEvaluator:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print("Evaluator using device:", self.device)

        self.lpips_metric = lpips.LPIPS(net="alex").to(self.device)

        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)

        self.clip_model.eval()

        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.to_tensor = T.ToTensor()

    def compute_rmse(self, img_target, img_gen):
        arr_target = np.array(img_target.convert("RGB")).astype(np.float32)
        arr_gen = np.array(img_gen.convert("RGB")).astype(np.float32)
        return np.sqrt(mean_squared_error(arr_target, arr_gen))

    def compute_lpips(self, img_target, img_gen):
        t_target = self.to_tensor(img_target.convert("RGB")).unsqueeze(0).to(self.device)
        t_gen = self.to_tensor(img_gen.convert("RGB")).unsqueeze(0).to(self.device)

        t_target = t_target * 2.0 - 1.0
        t_gen = t_gen * 2.0 - 1.0

        with torch.no_grad():
            return self.lpips_metric(t_target, t_gen).item()

    def compute_clip_similarity(self, img_target, img_gen):
        inputs_target = self.clip_processor(
            images=img_target.convert("RGB"),
            return_tensors="pt"
        ).to(self.device)

        inputs_gen = self.clip_processor(
            images=img_gen.convert("RGB"),
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            features_target = self.clip_model.vision_model(
                pixel_values=inputs_target["pixel_values"]
            ).pooler_output

            features_gen = self.clip_model.vision_model(
                pixel_values=inputs_gen["pixel_values"]
            ).pooler_output

            features_target = self.clip_model.visual_projection(features_target)
            features_gen = self.clip_model.visual_projection(features_gen)

            features_target = features_target / features_target.norm(
                p=2, dim=-1, keepdim=True
            )

            features_gen = features_gen / features_gen.norm(
                p=2, dim=-1, keepdim=True
            )

            similarity = torch.sum(features_target * features_gen).item()

        return similarity

    def evaluate(self, target_img, generated_img):
        img_target = target_img.convert("RGB").resize((768, 768))
        img_gen = generated_img.convert("RGB").resize((768, 768))

        return {
            "CLIP_Sim": self.compute_clip_similarity(img_target, img_gen),
            "LPIPS": self.compute_lpips(img_target, img_gen),
            "RMSE": self.compute_rmse(img_target, img_gen)
        }