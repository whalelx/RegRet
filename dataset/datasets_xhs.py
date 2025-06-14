import os
import json
from torch.utils.data import Dataset
import random 
from .datasets_mbeir import LazySupervisedDataset

old_query = {"type": "text", "text": f"<emb>."}
meta_query = {"type": "text", "text": "".join([f"<emb>" for i in range(1, 256)])}

class XHSDataset(LazySupervisedDataset):
    """
    Dataset for supervised fine-tuning 
    """

    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str,
        tokenizer = None 
    ) -> None:
        super(XHSDataset, self).__init__(query_data_path, cand_pool_path, instructions_path, image_path_prefix, tokenizer)

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
                        meta_query
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
                        meta_query
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
                        meta_query
                    ]
                },
            ]
        return message
