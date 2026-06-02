import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

image_folder = "data"

initial_prompts = {}

for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        raw_image = Image.open(img_path).convert('RGB')

        inputs = processor(raw_image, return_tensors="pt").to(device)

        out = model.generate(**inputs, max_new_tokens=70)
        caption = processor.decode(out[0], skip_special_tokens=True)

        initial_prompts[filename] = caption
        print(f"[{filename}] -> {caption}")
