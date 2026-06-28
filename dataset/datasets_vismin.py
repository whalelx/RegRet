from .datasets_mbeir import LazySupervisedDataset
from torch.utils.data import Dataset
from datasets import load_dataset

class VisMinDataset(LazySupervisedDataset):
    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str,
        img_parquet_path: str = "/mnt/tidal-alsh01/dataset/mmeb/vismin/",
        cache_dir: str = "/mnt/tidal-alsh01/dataset/mmeb/.cache/huggingface/dataset",
        tokenizer=None 
    ) -> None:
        super().__init__(query_data_path, cand_pool_path, instructions_path, image_path_prefix, tokenizer)
        self.vismin_data = load_dataset(img_parquet_path, cache_dir=cache_dir, num_proc=16)['train']

    def _get_image_by_row_number(self, row_number_str: str):
        if row_number_str.endswith(".jpg"):
            row_idx = int(row_number_str[:-4])
        else:
            row_idx = int(row_number_str)
    
        if row_idx < len(self.vismin_data):
            return self.vismin_data[row_idx]['image']
        else:
            print(f"Warning: Row number {row_idx} is out of range")
            return None

    def construct_messages(self, data_dict):
        if data_dict["box"] is None:
            data_dict["box"] = [0,0,1.,1.]
        if 'txt' in data_dict and 'image' in data_dict:
            img = self._get_image_by_row_number(data_dict['image'])
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img, "box": None},
                        {"type": "image", "image": img, "box": data_dict["box"]},
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
            img = self._get_image_by_row_number(data_dict['image'])
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img, "box": None},
                        {"type": "image", "image": img, "box": data_dict["box"]},
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
