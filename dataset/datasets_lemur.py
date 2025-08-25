'''
Set of training dataset for Lemur benchmark.
'''
from .datasets_mbeir import LazySupervisedDataset
from datasets import load_dataset

CACHE_DIR = "/mnt/tidal-alsh01/dataset/mmeb/.cache/huggingface/dataset"

class VisMinCLDataset(LazySupervisedDataset):
    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str,
        img_parquet_path: str,
        cache_dir: str = CACHE_DIR,
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
    
    def _prepare_data_dict(self,  txt, img_path, image_path_prefix,box=None):
        """Prepare data dictionary, handling row numbers for images"""
        if img_path is None or img_path == '':
            return {'txt': txt, "box": box}
        elif txt == '':
            # img_path is actually a row number string, get the actual image
            image = self._get_image_by_row_number(img_path)
            return {'image': image, "box": box}
        else:
            # Both text and image (row number)
            image = self._get_image_by_row_number(img_path)
            return {"txt": txt, "image": image, "box": box}

    def construct_messages(self, data_dict):
        get_boxop = BOX_OP_DICT[self.__class__]
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image'], "box": data_dict["box"], "box_op": get_boxop()},
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
        elif 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image'], "box": data_dict["box"], "box_op": get_boxop()},
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

class ImgDiffCLDataset(LazySupervisedDataset):
    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        image_path_prefix: str, # "/mnt/tidal-alsh01/dataset/mmeb/Img-Diff/"
        tokenizer=None 
    ) -> None:
        super().__init__(query_data_path, cand_pool_path, instructions_path, image_path_prefix, tokenizer)
    
    def construct_messages(self, data_dict):
        get_boxop = BOX_OP_DICT[self.__class__]
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image'], "box": data_dict["box"], "box_op": get_boxop()},
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
        elif 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image'], "box": data_dict["box"], "box_op": get_boxop()},
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
    
class DamCLDataset(LazySupervisedDataset):
    def __init__(
        self, 
        query_data_path: str, 
        cand_pool_path: str, 
        instructions_path: str,
        parquet_path: str,
        image_path_prefix: str="",
        tokenizer=None 
    ) -> None:
        super().__init__(query_data_path, cand_pool_path, instructions_path, image_path_prefix, tokenizer)
        self.split_names =  ["SAM"]
        self.dataset = {k: load_dataset(parquet_path, k, num_proc=16)['train'] for k in self.split_names}
    
    def _get_image_by_row_number(self, row_number_str: str):
        if row_number_str.endswith(".jpg"):
            row_number_str = row_number_str[:-4]
            _, split_name, row_idx = row_number_str.split('_')
            assert split_name in self.split_names, f"Invalid split name {split_name}"
            row_idx = int(row_idx)
        else:
            raise ValueError(f"Invalid row number {row_number_str}")

        if row_idx < len(self.dataset[split_name]):
            return self.dataset[split_name][row_idx]['jpg']
        else:
            print(f"Warning: Row number {row_idx} is out of range")
            return None
    
    def _prepare_data_dict(self,  txt, img_path, image_path_prefix,box=None):
        """Prepare data dictionary, handling row numbers for images"""
        if img_path is None or img_path == '':
            return {'txt': txt, "box": box}
        elif txt == '':
            # img_path is actually a row number string, get the actual image
            image = self._get_image_by_row_number(img_path)
            return {'image': image, "box": box}
        else:
            # Both text and image (row number)
            image = self._get_image_by_row_number(img_path)
            return {"txt": txt, "image": image, "box": box}

    def construct_messages(self, data_dict):
        get_boxop = BOX_OP_DICT[self.__class__]
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_dict['image'], "box": data_dict["box"], "box_op": get_boxop()},
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
                        {"type": "image", "image": data_dict['image'], "box": data_dict["box"], "box_op": get_boxop()},
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
        get_boxop = BOX_OP_DICT[self.__class__]
        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {
                    "role": "user",
                    "content": [
                    {"type": "image", "image": data_dict['image'], "box": data_dict["box"], "box_op": get_boxop()},
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
                    {"type": "image", "image": data_dict['image'], "box": data_dict["box"], "box_op": get_boxop()},
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


BOX_OP_DICT = {
    VisMinCLDataset: lambda: "concat",
    ImgDiffCLDataset: lambda: "concat",
    DamCLDataset: lambda: "crop", 
    XHSDataset: lambda: "crop",
}