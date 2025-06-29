
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import random

import glob
import transformers

from torch.utils.data import Dataset


class FGCLIPBboxDataset(Dataset):
    """Dataset for Stage2."""

    def __init__(self, data_path: str,
                 data_args: DataArguments,
                 img_preprocess=None,tokenizer=None):
        super(FGCLIPBboxDataset, self).__init__()

        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            list_data_dict = json.load(open(data_path, "r", encoding="utf-8"))
        elif data_path.endswith('.txt'):
            lines = open(data_path, "r", encoding="utf-8").readlines()
            list_data_dict = []
            for line in lines:
                json_file = line.rstrip()
                list_data_dict += json.load(open(json_file, "r",encoding="utf-8"))
        else:
            json_files = glob.glob(os.path.join(data_path, '*.json'))
            list_data_dict = []
            for json_file in json_files:
                list_data_dict += json.load(open(json_file, "r",encoding="utf-8"))

            jsonl_files = glob.glob(os.path.join(data_path, '*.jsonl'))
            for jsonl_file in jsonl_files:
                list_data_dict += json.load(open(jsonl_file, "r",encoding="utf-8"))
        

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.max_anns = 4

        self.data_args = data_args
        self.preprocess = img_preprocess
        self.image_root = data_args.image_folder
        self.max_length = data_args.max_seq_length
        self.base_length = data_args.base_seq_length
        self.base_image_size = data_args.base_image_size
        self.add_box_loss = data_args.add_box_loss
        self.use_hard_neg = data_args.use_hard_neg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        item = self.list_data_dict[i]

        caption = item["caption"]
        caption_short = "a photo of "+item["short_caption"]        

        image_path = item["f_path"]
        image_path = image_path.replace("grit-20m/data-12m/","")
        image_name = os.path.join(self.image_root,image_path)
        
        image = Image.open(image_name).convert("RGB")
        
        image = image.resize((self.base_image_size, self.base_image_size))

        image_tensor = self.preprocess.preprocess(image, return_tensors='pt')['pixel_values'][0]

        text =  torch.tensor(self.tokenizer([caption], max_length=self.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)
        short_text = torch.tensor(self.tokenizer([caption_short], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        

        if self.add_box_loss:
            box_texts = []
            bbox_info = item["bbox_info"]

            total_num = self.max_anns
            valid_num = min(len(bbox_info), self.max_anns)
            boxes_template = torch.zeros((total_num, 4), device=image_tensor.device)
            width, height = image.size
            for i in range(total_num):
                if i<valid_num:
                    bbox_data = bbox_info[i]
                    box = bbox_data["bbox"]
                    box_caption = random.choice([bbox_data["short_expr"], bbox_data["long_expr"]])

                else:
                    box = [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.000000000]
                    box_caption = ""
                    

                box_tensor = torch.tensor(box[:4])
                boxes_template[i] = box_tensor

                if box[0] > box[2] or box[1] > box[3]:
                    raise ValueError("Box coordinates are invalid.")

                box_text = torch.tensor(self.tokenizer([box_caption], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        
                box_texts.append(box_text)


            box_texts = torch.cat(box_texts,dim=0)
            bbox_num = torch.tensor([valid_num], device=image_tensor.device)
                    
        if self.use_hard_neg:
            hard_texts = []
           
            bbox_info = item["bbox_info"]

            width, height = image.size
            total_num = self.max_anns
            valid_num = min(len(bbox_info), self.max_anns)
            hard_boxes = torch.zeros((total_num, 4), device=image_tensor.device)
            valid_hard = 0
            for i in range(total_num):
                if i<valid_num:
                    bbox_data = bbox_info[i]
                    box = bbox_data["bbox"]
                    box_caption = bbox_data["short_expr"]
                    
                    box_tensor = torch.tensor(box[:4])

                    if box[0] > box[2] or box[1] > box[3]:
                        raise ValueError("Box coordinates are invalid.")
    
                    if bbox_data["flag_short_neg"] == 1:
                        cur_texts = [box_caption]
                        hard_negs = bbox_data["short_expr_negs"]
                        for key in hard_negs.keys():
                            cur_texts.append(hard_negs[key])
                        box_text = torch.tensor(self.tokenizer(cur_texts, max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        
                        hard_texts.append(box_text)

                        hard_boxes[valid_hard] = box_tensor
                        valid_hard = valid_hard+1


            valid_hard = torch.tensor([valid_hard], device=image_tensor.device)

            if len(hard_texts) > 0:
                hard_texts = torch.cat(hard_texts,dim=0)
            else:
                hard_texts = None

        data_dict = {}
        data_dict['image'] = image_tensor
        data_dict['text'] = text
        data_dict['short_text'] = short_text
        data_dict['add_box_loss'] = self.add_box_loss
        data_dict['use_hard_neg'] = self.use_hard_neg

        if self.add_box_loss:
            data_dict['box_texts'] = box_texts
            data_dict['box_infos'] = boxes_template
            data_dict['box_nums'] = bbox_num
        if self.use_hard_neg:
            data_dict['hard_texts'] = hard_texts
            data_dict['hard_infos'] = hard_boxes
            data_dict['hard_nums'] = valid_hard
            
        return data_dict




@dataclass
class FGCLIPDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch = {}
        images = [instance['image'] for instance in instances]
        batch['image'] = torch.stack(images)
        texts = [instance['text'] for instance in instances]
        batch['text_long'] = torch.cat(texts,dim=0)
        short_texts = [instance['short_text'] for instance in instances]
        batch['text_short'] = torch.cat(short_texts,dim=0)
        
        batch["add_box_loss"] = instances[0]["add_box_loss"]
        batch["use_hard_neg"] = instances[0]["use_hard_neg"]
        
        if batch["add_box_loss"]:
            box_texts = [instance['box_texts'] for instance in instances]
            batch['box_texts'] = torch.cat(box_texts,dim=0)
            box_infos = [instance['box_infos'] for instance in instances]
            batch['box_infos'] = torch.cat(box_infos,dim=0)
            box_nums = [instance['box_nums'] for instance in instances]
            batch['box_nums'] = torch.cat(box_nums, dim=0)
        if batch["use_hard_neg"] :
            hard_texts = []
            for instance in instances:
                if instance['hard_texts'] != None:
                    hard_texts.append(instance['hard_texts'])

            batch['hard_texts'] = torch.cat(hard_texts,dim=0)
            hard_infos = [instance['hard_infos'] for instance in instances]
            batch['hard_infos'] = torch.cat(hard_infos,dim=0)
            hard_nums = [instance['hard_nums'] for instance in instances]
            batch['hard_nums'] = torch.cat(hard_nums, dim=0)                

        return batch

