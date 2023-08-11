import os
import argparse

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from clipseg.models.clipseg import CLIPDensePredT
from models.dataset import PROMPT_TEMPLATE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--super_class', type=str, required=True)
    parser.add_argument('--model_weight', type=str, default='./clipseg/weights/rd64-uni-refined.pth')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    image_dir = args.image_dir
    super_class = args.super_class
    model_weight_path = args.model_weight
    device = args.device

    files = [f for f in os.listdir(image_dir) if f.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']]
    image_paths = [os.path.join(image_dir, f) for f in files]
    mask_save_paths = [os.path.join(image_dir, 'mask', f) for f in files]

    os.makedirs(os.path.join(image_dir, 'mask'))

    prompts = [prompt.format(super_class) for prompt in PROMPT_TEMPLATE]

    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    model = model.eval().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])

    with torch.no_grad():
        for img_p, save_p in tqdm(zip(image_paths, mask_save_paths)):
            img = Image.open(img_p)
            img = transform(img).unsqueeze(0).to(device)
            preds = model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]
            preds = torch.sigmoid(preds)
            pred = preds.mean(dim=0)[0]
            pred = Image.fromarray((pred.cpu().numpy() * 255).astype(np.uint8))
            pred.save(save_p)
