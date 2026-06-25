import jsonlines
from typing import Dict, List
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

class PAMDataset(Dataset):

    def __init__(
        self, 
        data_path: str, 
        mode: str='draw', # ['crop', 'draw']
        max_length: int = 100000,
        text_truncate_length: int = 1024,
        image_prefix: str = '/mnt/tidal-alsh01/usr/liangxun/data',
        inference: bool = False
    ) -> None:
        super(PAMDataset, self).__init__()
        
        self.dataset = []
        with jsonlines.open(data_path, "r") as reader:
            for item in reader:
                self.dataset.append(item)
    
        self.length = len(self.dataset)
        self.mode = mode
        self.max_length = max_length
        if max_length<=0:
            self.max_length = self.length // 2

        self.mode = mode
        self.text_truncate_length = text_truncate_length
        self.inference = inference
        self.image_prefix = image_prefix

    def __len__(self) -> int:
        return self.max_length
    
    def get_prompt(self):
        if self.mode == 'crop':
            return 'Provide a comprehensive and detailed description for this subject.'
        elif self.mode == 'draw':
            return "Describe the region in the image bounded by a red box."

    def construct_messages(self, idx: int):
        item = self.dataset[idx]
        image_path = os.path.join(self.image_prefix, item['image'])
        image = Image.open(image_path)
        image = image.convert("RGB")

        b = item['bbox']
        x1, y1 = int(b[0]), int(b[1])
        x2, y2 = int(b[0] + b[2]), int(b[1] + b[3])
        
        w, h = image.size
        box = [x1/w, y1/h, x2/w, y2/h]

        text = item["conversations"][1]['value'].strip()

        if len(text) > self.text_truncate_length:
            text = text[:self.text_truncate_length]

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "box": box, "box_op": "crop"},
                    {"type": "text", "text": f"\n{self.get_prompt()}"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text}
                ]
            },
        ]
        if self.inference:
            message = [message[0]]
        return message

    def get_instance(self, index):
        message = self.construct_messages(index)
        
        return message 

    def __getitem__(self, i) -> Dict[str, List]: 
        j = i * 2 + 1
        return self.get_instance(i), self.get_instance(j)

if __name__ == "__main__":
    pam_dataset = PAMDataset(
        data_path="/mnt/tidalfs-hssh01/dataset/mmeb/PAM-data/image-obj-caption/PamCOCO.jsonl",
        max_length=108000,
        mode = 'crop',
    )