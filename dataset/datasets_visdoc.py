from typing import List, Tuple
import datasets
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

prompts = [
    "Find a image of a pdf page that helps answers the question.",
    "Find a visual document that provides answer to the following question.",
    "I want to solve the following problem. Retrieve a helpful page from pdf documents.",
    "I want to find an image page that contains the evidence for the following question."
]

class VisRAGSynDataset(Dataset):
    def __init__(self, 
                 cache_dir = "/mnt/tidal-alsh01/dataset/mmeb/.cache/huggingface/dataset", 
                 max_length: int = -1, split="train"):
        super().__init__()

        # self.image_resolution = data_args.get("image_resolution", None)
        self.cache_dir = cache_dir

        self.dataset = load_dataset(
            "/mnt/tidal-alsh01/dataset/mmeb/VisRAG-Ret-Train-Synthetic-data",
            split=split,
            cache_dir=self.cache_dir,
            num_proc=16
        )
        self.max_length = max_length

    def __len__(self):
        if self.max_length >= 0:
            return min(self.max_length, len(self.dataset))
        return len(self.dataset)

    def _prepare_data_dict(self, txt, image, box=None):
        if image is None:
            return {'txt': txt}
        elif txt == '':
            return {'image': image}
        return {"txt": txt, "image": image}

    def construct_messages(self, data_dict):
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
                        {"type": "text", "text": f"{data_dict['txt']}\nSummarize above image and sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'txt' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]
        elif 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image']},
                        {"type": "text", "text": f"\nSummarize above image in one word: "}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<emb>."}
                    ]
                },
            ]

        return message

    def __getitem__(self, item):
        data_row = self.dataset[item]
        query_text = data_row.get('query', '')
        prompt = random.choice(prompts)
        query_text = f"{prompt} {query_text}"
        image = data_row.get('image', None)
        
        # remove image token
        # if query_text:
        #     query_text = query_text.replace(vlm_image_tokens["PHI3V"], "")

        data_dict = self._prepare_data_dict(query_text, None)
        query_message = self.construct_messages(data_dict)

        data_dict = self._prepare_data_dict('', image)
        cand_message = self.construct_messages(data_dict)

        return query_message, cand_message


if __name__ == "__main__":
    try:
        ds = VisRAGSynDataset(data_args=data_args, split="train")
        print(f"Dataset length: {len(ds)}")
        print(f"Dataset sample: {ds[0]}")
    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("Please check the dataset structure and adjust field names accordingly")