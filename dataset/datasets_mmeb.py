from typing import List, Tuple
import datasets
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

vlm_image_tokens = {
    "PHI3V": "<|image_1|>",
    "LLAVA_NEXT": "<image>",
    "QWEN2_VL": "<|image_pad|>",
    "QWEN2_5_VL": "<|image_pad|>",
}


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


class MMEBEvalDataset(Dataset):
    def __init__(self, data_args, subset, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        super().__init__()

        self.dataset_name = data_args["dataset_name"]
        self.image_resolution = data_args["image_resolution"]
        self.image_dir = data_args["image_dir"]

        self.eval_data = load_dataset(
            self.dataset_name, # TIGER-Lab/MMEB-eval
            subset, # subset name
            split="test",
        )
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        text = text.replace(vlm_image_tokens["PHI3V"], "") #NOTE in mbeir there are no such things vlm_image_tokens["QWEN2_5_VL"]
        if img_path != "" and img_path is not None:
            img_path = os.path.join(self.image_dir, img_path)
        message = self.construct_messages(self._prepare_data_dict(text, img_path))

        return message, None

    def _prepare_data_dict(self, txt, img_path, box=None):
        if img_path is None or img_path == '':
            return {'txt': txt, "box":box}
        elif txt == '':
            return {'image': img_path, "box":box}
        return {"txt": txt, "image": img_path, "box":box}

    def construct_messages(self, data_dict):
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image'], "box": data_dict["box"]},
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
                        {"type": "image", "image": data_dict['image'], "box": data_dict["box"]},
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

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif type(row[text_field]) == list:
                assert type(row[img_path_field]) == list and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data

if __name__ == "__main__":
    data_args = {
        "dataset_name": "/mnt/tidal-alsh01/dataset/mmeb/mmeb-eval/",
        "image_resolution": "low",
        "image_dir": "/mnt/tidal-alsh01/dataset/mmeb/mmeb-eval/"
    }

    ds = MMEBEvalDataset(
            data_args=data_args,
            subset="MSCOCO_t2i",
            text_field="qry_text",
            img_path_field="qry_img_path",
        )