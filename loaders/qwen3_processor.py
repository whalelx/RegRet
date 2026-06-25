from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs
import torch

class Qwen3VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }



class RegRetQwen3Processor(Qwen3VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor=video_processor, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        replace_two_imgs: Optional[set[int]] = (),
        id_dict=None,
        **kwargs,
    ):
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None and len(images) > 0:
            crop_or_concat_img_inputs = self.get_images_by_group(images, id_dict, output_kwargs)
            image_inputs = self.image_processor(
                images=self.get_images(images, id_dict), 
                **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None
            crop_or_concat_img_inputs = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    if i in replace_two_imgs:
                        text[i] = text[i].replace(
                            self.image_token,
                            "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                            + "<|vision_end|><|vision_start|>"
                            + "<|placeholder|>" * (image_grid_thw[index + 1].prod() // merge_length),
                            1,
                        )
                        index += 2
                    else:
                        text[i] = text[i].replace(
                            self.image_token,
                            "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                            1,
                        )
                        index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        
        return (
            BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors),
            crop_or_concat_img_inputs,
        )

    ### Stuff add by liangxun
    def get_images_by_group(self, images, id_dict, output_kwargs):
        inputs = {}
        for key in ("justfull", "crop_crop", "crop_full", "concat_crop", "concat_full"):
            ids = getattr(id_dict, key)
            if len(ids) == 0:
                continue
            cur_images = [images[i] for i in ids]
            tmp = self.image_processor(images=cur_images, **output_kwargs["images_kwargs"])
            inputs[key+"_pixel_values"] = tmp.pixel_values
            # HACK TODO why image grid becomes float if it is not converted here?
            inputs[key+"_image_grid_thw"] = tmp.image_grid_thw.to(torch.int64) # HACK
        return BatchFeature(inputs)
    
    def get_images(self, images, id_dict):
        assert id_dict is not None, "id_dict should not be None when images is not None"
        img_in_forward_idx = sorted(id_dict.justfull + id_dict.crop_crop + id_dict.concat_crop + id_dict.concat_full)
        return [images[i] for i in img_in_forward_idx]