import json
from typing import Dict, List
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import pycocotools
from pycocotools import mask as cocomask
from PIL import Image
import pickle
import random

def counts_to_mask(maskrle):
    return np.array(cocomask.decode(maskrle), dtype=np.float32)

def visualize_mask_on_image_pil(original_pil, binary_mask_np, 
                                color=(255, 0, 0), alpha_percent=50):
    height, width = binary_mask_np.shape
    if original_pil.size != (width, height):
        print(f"Warning: Mask size ({width},{height}) and image size ({original_pil.size}) differ.")
    alpha_value = int((alpha_percent / 100.0) * 255)
    colored_mask_pil = Image.new("RGBA", original_pil.size, (0, 0, 0, 0))
    mask_rgba_np = np.zeros((height, width, 4), dtype=np.uint8)
    
    mask_indices = binary_mask_np == 1
    mask_rgba_np[mask_indices] = list(color) + [alpha_value]

    colored_mask_pil_from_np = Image.fromarray(mask_rgba_np, "RGBA")
    original_pil.putalpha(255)
    overlaid_image_pil = Image.alpha_composite(original_pil, colored_mask_pil_from_np)    
    return overlaid_image_pil

def mask2box(mask):
    box = None
    pos = np.where(mask > 0)
    width, height = mask.shape

    if pos[0].size > 0 and pos[1].size > 0:
        x_min = np.min(pos[1]) / width
        x_max = np.max(pos[1]) / width
        y_min = np.min(pos[0]) / height
        y_max = np.max(pos[0]) / height
        box = [x_min, y_min, x_max, y_max]
    return box

def lower_resolution(img: Image):
    h, w = img.size
    if h > 1000 and w > 1000:
        return img.resize((h//10, w//10))
    else:
        return img

class DAMDataset(Dataset):

    def __init__(
        self, 
        data_path: str, 
        mode: str='pretrained',
        max_length: int = 100000,
    ) -> None:
        super(DAMDataset, self).__init__()
        # 'SAM' is too large, 'SAV' caption is missing, because it's a vedio dataset
        self.split_names =  ['COCOStuff', 'LVIS', 'Mapillary', 'OpenImages', 'PACO'] 
        self.dataset = {k: load_dataset(data_path, k) for k in self.split_names}
        self.lengths = [len(self.dataset[n]['train']) for n in self.split_names]
        self.max_length = max_length

        self.mode = mode

    def __len__(self) -> int:
        return self.max_length
    
    def get_dataset_idx(self, index):
        for i, ds in enumerate(self.lengths):
            if index < ds:
                return i, index
            else:
                index -= ds

    def construct_messages(self, idx: int):
        i, index = self.get_dataset_idx(idx)
        splitname = self.split_names[i]
        item = self.dataset[splitname]['train'][index]
        anno = random.choice(pickle.loads(item['pickle']))
        text = anno['caption']
        mask = counts_to_mask(anno['mask_rle'])
        box = mask2box(mask)
        image = lower_resolution(item['jpg'])
        # image = visualize_mask_on_image_pil(item['jpg'], mask)

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "box": box},
                    {"type": "text", "text": f"\nDescribe the region in the image bounded by a red box."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text}
                ]
            },
        ]
        return message

    def get_instance(self, index):
        message = self.construct_messages(index)
        
        return message 

    def __getitem__(self, i) -> Dict[str, List]: 
        j = i + self.max_length
        return self.get_instance(i), self.get_instance(j)


if __name__ == "__main__":
    datapath = "/mnt/tidal-alsh01/dataset/mmeb/describe-anything-data"

    ds = DAMDataset(datapath)
    breakpoint()
    ds[0]
    ds[97000]