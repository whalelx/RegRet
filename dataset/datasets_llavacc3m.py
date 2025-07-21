import os
import json
from typing import Dict, List
from torch.utils.data import Dataset

class LLavaCC3MDataset(Dataset):

    def __init__(
        self, 
        image_data_path: str = "/mnt/tidal-alsh01/dataset/mmeb/llava-cc3m", 
        json_path: str = "/mnt/tidal-alsh01/usr/liangxun/data/vpt/llava_alignment_seg_qwen_response_train.json", 
        max_length: int = 100000,
    ) -> None:
        super(LLavaCC3MDataset, self).__init__()
        self.images = []
        self.texts = []
        self.image_data_path = image_data_path
        data_path = json_path
        self.data = json.load(open(data_path))
        self.max_length = min(max_length, len(self.data)//2)

    def __len__(self) -> int:
        return self.max_length

    def construct_messages(self, i):
        item = self.data[i]
        img_pth = item['detection_images'][0]
        img_pth = os.path.join(self.image_data_path, img_pth.split('/')[-1])
        text = item['messages'][1]['content']
        prompt = item['messages'][0]['content'].replace("<detection_image>", "").replace("\n", "")
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_pth},
                    {"type": "text", "text": prompt},
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


    def __getitem__(self, i) -> Dict[str, List]:
        j = i * 2 + 1
        return self.construct_messages(i), self.construct_messages(j)