from typing import Dict, Sequence

import torch

from . import register_collator
from .base import BaseDataCollator
from .qwen2_vision_process import process_vision_info, process_vision_info_with_focal


@register_collator("qwen2-vl-7b")
class Qwen2VL7BDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        '''
        make sure it returns the combined grid_thw as the target, and pixel_values as the [full, focal].
        '''
        
        category_size = len(messages[0])
        if category_size == 3:
            has_hard_negative = True 
        else:
            has_hard_negative = False 
        
        new_messages = []
        for category in range(category_size):
            for item in messages:
                d=item[category]
                if d is not None:
                    new_messages.append(d)
                else:
                    pass

        image_inputs, id_dict = process_vision_info_with_focal(new_messages, box_op="crop")
        video_inputs = None

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
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

        result = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            id_dict=id_dict,
        )
        if crop_or_concat_img_inputs is not None:
            result = result | crop_or_concat_img_inputs
        return result