# https://github.com/XavierXiao/Dreambooth-Stable-Diffusion/

import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

PROMPT_TEMPLATE = {
    'a photo of a {}': 4,
    'a good photo of a {}': 5,
    'the photo of a {}': 4,
    'a good photo of the {}': 5,
    'image of a {}': 3,
    'image of the {}': 3,
    'A photograph of {}': 3,
    'A {} shown in a photo': 1,
    'A photo of {}': 3,
}


class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=512,
                 repeats=5000,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 center_crop=False,
                 prompt_template=None,
                 ):
        self.prompt_template = prompt_template or PROMPT_TEMPLATE
        self.prompt_template_keys = list(self.prompt_template.keys())

        self.data_root = data_root
        self.flip_p = flip_p

        files = [f for f in os.listdir(self.data_root) if f.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']]
        self.image_paths = [os.path.join(self.data_root, f) for f in files]
        self.mask_paths = [os.path.join(self.data_root, 'mask', f) for f in files]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.center_crop = center_crop

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])
        mask = Image.open(self.mask_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")
        if not mask.mode == 'L':
            mask = mask.convert('L')

        text = random.choice(self.prompt_template_keys)
        concept_i = self.prompt_template[text]
        text = text.format(self.placeholder_token)

        example["caption"] = text
        example['concept_token_idx'] = concept_i

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.float32)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
            mask = mask[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        mask = Image.fromarray(mask)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            mask = mask.resize((self.size, self.size), resample=self.interpolation)
        if random.random() < self.flip_p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.float32)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example['mask'] = (mask / mask.max())[None, :]

        return example
