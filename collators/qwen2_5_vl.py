from typing import Dict, Sequence

import torch

from . import register_collator
from .base import BaseDataCollator
from .qwen2_5_vision_process import process_vision_info, process_vision_info_with_focal


@register_collator("qwen2_5-vl-7b")
class Qwen2_5VLDataCollator(BaseDataCollator):
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


        image_nofocal, image_focal_full, image_focal_crop, resort_id = process_vision_info_with_focal(new_messages, box_op="crop")
        video_inputs = None
        image_inputs = image_nofocal + image_focal_crop

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in new_messages
        ]
        texts = [texts[i] for i in resort_id]

        resort_id = torch.tensor(resort_id)
        reverse_idx = torch.argsort(resort_id)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = None 
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
        else:
            pixel_values = None 
        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
        else:
            image_grid_thw = None 

        if len(image_inputs) == 0:
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=None,
                image_grid_thw=None,
                labels=labels,
                has_hard_negative=has_hard_negative,
                reverse_idx = reverse_idx
            )

        prefix_img_nums = len(image_nofocal)
        prefix_token_length = image_grid_thw[:prefix_img_nums].prod(1).sum().item()
        # pixel values and image_grithw 真实可用的东西

        focal_crop_pixel_values = pixel_values[prefix_token_length:]
        focal_crop_grid_thw = image_grid_thw[prefix_img_nums:]

        nofocal_full_pixel_values = pixel_values[:prefix_token_length]
        nofocal_full_grid_thw = image_grid_thw[:prefix_img_nums]

        focal_inputs = self.processor.image_processor(
            images=image_focal_full,
            return_tensors="pt",
        )
        focal_full_pixel_values =  focal_inputs.pixel_values
        focal_full_grid_thw = focal_inputs.image_grid_thw.to(torch.int64)

        focal_image_ids = torch.arange(prefix_img_nums, prefix_img_nums + len(image_focal_full))

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=torch.cat([nofocal_full_pixel_values, focal_full_pixel_values], dim=0),
            image_grid_thw=torch.cat([nofocal_full_grid_thw, focal_full_grid_thw], dim=0),
            labels=labels,
            has_hard_negative=has_hard_negative,
            focal_pixel_values=focal_crop_pixel_values,
            focal_image_grid_thw=focal_crop_grid_thw,
            focal_image_ids=focal_image_ids,
            real_image_grid_thw=image_grid_thw,
            reverse_idx = reverse_idx
        )