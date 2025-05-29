import json
from typing import Dict, List
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import pycocotools
from PIL import Image
import pickle

# datapath = "/mnt/tidal-alsh01/dataset/mmeb/describe-anything-data"

def counts_to_mask(maskrle):
    return np.array(pycocotools.mask.decode(maskrle), dtype=np.float32)

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
    pos = np.where(mask == 1)
    if pos[0].size > 0 and pos[1].size > 0:
        x_min = np.min(pos[1])
        x_max = np.max(pos[1])
        y_min = np.min(pos[0])
        y_max = np.max(pos[0])
        box = [x_min, y_min, x_max, y_max]
    return box

class DAMDataset(Dataset):

    def __init__(
        self, 
        data_path: str, 
        mode: str='pretrained',
        max_samples: int = 200000,
    ) -> None:
        super(DAMDataset, self).__init__()
        self.images = []
        self.split_names =  ['COCOStuff', 'LVIS', 'Mapillary', 'OpenImages', 'PACO', 'SAM', 'SAV']
        self.dataset = {k: load_dataset(data_path, k) for k in self.split_names}
        self.single_split_size = max_samples // len(self.split_names)
        self.max_samples = self.single_split_size * len(self.split_names)

        self.mode = mode

    def __len__(self) -> int:
        return self.max_samples

    def construct_messages(self, idx: int):
        splitname = self.split_names[idx // self.single_split_size]
        item = self.dataset[splitname]['train'][idx % self.single_split_size]

        anno = np.random.choice(pickle.loads(item['pickle']))
        text = anno['caption']
        mask = counts_to_mask(anno['mask_rle'])
        box = mask2box(mask)
        image = item['jpg']
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
        if self.mode == 'finetuned':
            text = "Find an image caption describing the following everyday image." # TODO
            message = self.construct_messages(i)
        elif self.mode == 'pretrained':
            message = self.construct_messages(image=self.images[index])
        
        return message 

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i), i 