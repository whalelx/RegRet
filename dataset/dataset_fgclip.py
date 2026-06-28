import os
import json
from typing import Dict, List
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
from PIL import Image
import random
from io import BytesIO
from datasets import load_dataset


def lower_resolution(img: Image):
    """Lower image resolution if too large"""
    h, w = img.size
    if h > 1000 and w > 1000:
        return img.resize((h//2, w//2))
    else:
        return img


def normalize_bbox(bbox, image_width, image_height):
    """Normalize bbox coordinates to [0, 1] range"""
    x1, y1, x2, y2 = bbox[:4]
    return [x1/image_width, y1/image_height, x2/image_width, y2/image_height]


class FGCLIPDataset(Dataset):
    """FGClip Dataset following DAM structure."""

    def __init__(
        self, 
        data_path: str = 'mnt/tidal-alsh01/dataset/mmeb/fg-clip', 
        mode: str = 'train',
        max_length: int = 1200000,
        text_truncate_length: int = 512,
        use_bbox_ratio: float = 0.0,
    ) -> None:
        super(FGCLIPDataset, self).__init__()
        
        # Load parquet files from directory
        self.data_path = data_path
        self.parquet_files = glob.glob(os.path.join(data_path, '*.parquet')) #[50:]
        self.dataset = load_dataset('parquet', num_proc=16, data_files=self.parquet_files)
        self.text_truncate_length = text_truncate_length
        self.use_bbox_ratio = use_bbox_ratio
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {data_path}")
        
        # Load all data and calculate lengths
        # self.datasets = []
        # self.lengths = []
        
        # for parquet_file in self.parquet_files:
        #     df = pd.read_parquet(parquet_file)
        #     self.datasets.append(df)
        #     self.lengths.append(len(df))
        
        self.max_length = min(max_length, len(self.dataset['train'])//2)
        self.mode = mode

    def __len__(self) -> int:
        return self.max_length
    
    def get_dataset_idx(self, index):
        """Get dataset index and item index, similar to DAM"""
        for i, ds_len in enumerate(self.lengths):
            if index < ds_len:
                return i, index
            else:
                index -= ds_len
        # If index exceeds total length, wrap around
        total_length = sum(self.lengths)
        index = index % total_length
        for i, ds_len in enumerate(self.lengths):
            if index < ds_len:
                return i, index
            else:
                index -= ds_len

    def construct_messages(self, idx: int):
        # dataset_idx, item_idx = self.get_dataset_idx(idx)
        # df = self.datasets[dataset_idx]
        item = self.dataset['train'][idx]
        # item = df.iloc[item_idx]
        
        # Extract data from parquet row
        caption = item.get("caption", "")
        short_caption = item.get("short_caption", "")
        f_path = item.get("f_path", "")
        bbox_info = item.get("bbox_info", [])
        
        if 'image' in item and item['image'] is not None:
            # Image stored as bytes in parquet
            image = Image.open(BytesIO(item['image'])).convert("RGB")
        elif f_path:
            # Image stored as file path
            # Clean up path if needed
            if "grit-20m/data-12m/" in f_path:
                f_path = f_path.replace("grit-20m/data-12m/", "")
            
            image_path = os.path.join(self.data_path, f_path)
            if not os.path.exists(image_path):
                # Try alternative path structures
                image_path = f_path
            
            image = Image.open(image_path).convert("RGB")
        
        # Lower resolution if needed
        image = lower_resolution(image)
        w,h = image.size
        if w < 10 or h < 10:
            return None
        
        # return 16% of the data as box, which is around 200k 
        use_box_flag = self.use_bbox_ratio > 0 and random.random() < self.use_bbox_ratio
        # Process bbox info
        # Pass the box process for now
        if False: #bbox_info and len(bbox_info) > 0: # and random.random() > 0.833: # return 16% of the data as box, which is around 200k 

            # Select a random bbox annotation
            selected_bbox = random.choice(bbox_info)
            bbox = selected_bbox.get("bbox", [0, 0, 1., 1, -1])
            bbox = bbox[:4]
            
            # Normalize bbox coordinates
            # image_width, image_height = image.size
            # normalized_bbox = normalize_bbox(bbox, image_width, image_height)
            normalized_bbox = bbox
            
            # Get bbox description
            bbox_text = selected_bbox.get("long_expr", "") or selected_bbox.get("short_expr", "")
            if not bbox_text:
                bbox_text = caption or short_caption
            bbox_text = bbox_text[:self.text_truncate_length]
            
            # Construct message with bbox (following DAM format)
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image, "box": normalized_bbox},
                        {"type": "text", "text": f"\nDescribe the image in a short scentence."}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": bbox_text}
                    ]
                }
            ]
        else:
            # No bbox info, use full image with caption (fallback)
            text = caption or short_caption or "This is an image."
            text = text[:self.text_truncate_length]
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"\nDescribe the image in detail."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": text}
                    ]
                }
            ]
        
        return message

    def __getitem__(self, i) -> Dict[str, List]: 
        j = i * 2 + 1
        return self.construct_messages(i), self.construct_messages(i)

if __name__ == "__main__":
    # Example usage
    datapath = "/mnt/tidalfs-hssh01/dataset/mmeb/fg-clip"

    ds = FGCLIPDataset(datapath, max_length=1000)
    print(f"Dataset length: {len(ds)}")
    print(f"Number of parquet files: {len(ds.parquet_files)}")

    # Test getting an item
    try:
        item1, item2 = ds[0]
        print("Successfully loaded first item")
        print(f"Item 1 structure: {type(item1)}")
        print(f"User content keys: {item1[0]['content'][0].keys()}")
    except Exception as e:
        print(f"Error loading item: {e}")