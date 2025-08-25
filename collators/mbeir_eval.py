from typing import Dict, Sequence
import torch
from .base import BaseDataCollator
from .qwen2_vision_process import process_vision_info, process_vision_info_with_focal
# from qwen_vl_utils import process_vision_info


class MbeirQueryDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        new_messages = []
        qids = []
        for item in messages:
            new_messages.append(item[0])
            qids.append(item[1])

        image_inputs, id_dict = process_vision_info_with_focal(new_messages, box_op="crop")
        video_inputs = None

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in new_messages
        ]

        inputs, crop_or_concat_img_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            id_dict=id_dict,
            replace_two_imgs=set(id_dict.multi_img_texts)
        )

        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = None

        # if 'pixel_values' in inputs:
        #     pixel_values = inputs['pixel_values']
        # else:
        pixel_values = None

        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
        else:
            image_grid_thw = None

        has_hard_negative = False 

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            qids=qids,
            id_dict=id_dict,
        ) | crop_or_concat_img_inputs

class MbeirCandidateDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        new_messages = []
        dids = []

        for item in messages:
            new_messages.append(item[0])
            dids.append(item[1])

        image_inputs, id_dict = process_vision_info_with_focal(new_messages, box_op="crop")
        video_inputs = None

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in new_messages
        ]

        inputs, crop_or_concat_img_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            id_dict=id_dict,
            replace_two_imgs=set(id_dict.multi_img_texts)
        )

        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = None

        # if 'pixel_values' in inputs:
        #     pixel_values = inputs['pixel_values']
        # else:
        pixel_values = None

        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
        else:
            image_grid_thw = None

        has_hard_negative = False 

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            dids=dids,
            id_dict=id_dict,
        ) | crop_or_concat_img_inputs