import os
import json
from torch.utils.data import Dataset
import random 
from .datasets_mbeir import LazySupervisedDataset

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
        return message
