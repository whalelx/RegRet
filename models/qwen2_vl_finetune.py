from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, PreTrainedTokenizer
from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
import torch.nn.functional as F
from .visual_backbone import Qwen2ContextVisionTransformerPretrainedModel

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Qwen2VLRetFinetuneForConditionalGeneration(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2ContextVisionTransformerPretrainedModel._from_config(config.vision_config)

        # Set default values for new config parameters if not present
        if not hasattr(config, 'language_loss_weight'):
            config.language_loss_weight = 1.0
        if not hasattr(config, 'use_angle_sim'):
            config.use_angle_sim = False
        if not hasattr(config, 'cos_sim_temp'):
            config.cos_sim_temp = 0.05
        if not hasattr(config, 'nocausal_attn'):
            config.nocausal_attn = False
        self.flag_set_causal = False
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        inference=False,
        has_hard_negative=False,
        qids=None,
        dids=None,
        ids=None,
        # our dataflows
        id_dict=None,
        justfull_pixel_values=None,
        justfull_image_grid_thw=None,
        crop_crop_pixel_values=None,
        crop_crop_image_grid_thw=None,
        crop_full_pixel_values=None,
        crop_full_image_grid_thw=None,
        concat_crop_pixel_values=None,
        concat_crop_image_grid_thw=None,
        concat_full_pixel_values=None,
        concat_full_image_grid_thw=None
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        group_imgs = {
            'justfull': {
                'pixel_values': justfull_pixel_values,
                'image_grid_thw': justfull_image_grid_thw
            },
            'crop_crop': {
                'pixel_values': crop_crop_pixel_values,
                'image_grid_thw': crop_crop_image_grid_thw
            },
            'crop_full': {
                'pixel_values': crop_full_pixel_values,
                'image_grid_thw': crop_full_image_grid_thw
            },
            'concat_crop': {
                'pixel_values': concat_crop_pixel_values,
                'image_grid_thw': concat_crop_image_grid_thw
            },
            'concat_full': {
                'pixel_values': concat_full_pixel_values,
                'image_grid_thw': concat_full_image_grid_thw
            }
        }
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # set mini_batch to 32
        mini_batch_size = 60
        input_ids_list = torch.split(input_ids, mini_batch_size)
        attention_mask_list = torch.split(attention_mask, mini_batch_size)
        if image_grid_thw is not None:
            image_nums = 0
        
        all_hidden_states = []

        image_embeds = []

        if inputs_embeds is None:
            for k in group_imgs:
                if (sub_pixel_values := group_imgs[k]['pixel_values']) is not None and len(sub_pixel_values) > 0:
                    group_imgs[k]['pixel_values'] = sub_pixel_values.type(self.visual.dtype)
                    use_image = True
            if use_image:
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw, group_imgs=group_imgs, id_dict=id_dict)
    
        for i in range(len(input_ids_list)):
            if inputs_embeds is None:
                batch_inputs_embeds = self.model.embed_tokens(input_ids_list[i])
                if pixel_values is not None or use_image:
                    image_mask = input_ids_list[i] == self.config.image_token_id
                    # current_image_num = torch.sum(torch.any(image_mask, dim=-1)).cpu().item()
                    current_image_num = (input_ids_list[i]==151652).sum().cpu().item()
                    if current_image_num != 0:
                        imgs_prefix = image_grid_thw[:image_nums].prod(1).sum().item()//4
                        imgs_offset = image_grid_thw[image_nums:image_nums + current_image_num].prod(1).sum().item()//4
                        batch_image_embeds = image_embeds[imgs_prefix:imgs_prefix + imgs_offset]
                        image_nums = image_nums + current_image_num
                        if self.training:
                            batch_inputs_embeds = batch_inputs_embeds.clone()
                        batch_inputs_embeds[image_mask] = batch_image_embeds
                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                    video_mask = input_ids == self.config.video_token_id
                    inputs_embeds[video_mask] = video_embeds
                if attention_mask is not None:
                    batch_attention_mask = attention_mask_list[i].to(batch_inputs_embeds.device)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=batch_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=batch_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            all_hidden_states.append(hidden_states)

        hidden_states = torch.cat(all_hidden_states)

        if has_hard_negative:
            batch_size = len(hidden_states) // 3
        elif not inference:
            batch_size = len(hidden_states) // 2
        elif inference:
            batch_size = len(hidden_states)

        if inference:
            assert batch_size == len(hidden_states)

        embed_index = self.config.emb_token_ids[0]
        embed_indices = torch.argmax((labels == embed_index).int(), dim=1) 
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)

        if inference:
            if ids is not None:
                return embed_features, ids 
            elif qids is not None or dids is not None:
                return embed_features, qids, dids 
            return embed_features 
        if has_hard_negative:
            embed1, embed2, embed3 = embed_features[:batch_size], embed_features[batch_size:2*batch_size], embed_features[2*batch_size:]
        else:
            embed1, embed2 = embed_features[:batch_size], embed_features[batch_size:]
        loss_fct = nn.CrossEntropyLoss()

        if dist.is_initialized():
            if has_hard_negative:
                embed3_list = [torch.zeros_like(embed3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=embed3_list, tensor=embed3.contiguous())
                embed3_list[dist.get_rank()] = embed3 
                embed3 = torch.cat(embed3_list, 0)
            
            # Dummy vectors for allgather
            embed1_list = [torch.zeros_like(embed1) for _ in range(dist.get_world_size())]
            embed2_list = [torch.zeros_like(embed2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=embed1_list, tensor=embed1.contiguous())
            dist.all_gather(tensor_list=embed2_list, tensor=embed2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            embed1_list[dist.get_rank()] = embed1
            embed2_list[dist.get_rank()] = embed2
            # Get full batch embeddings: (bs x N, hidden)
            embed1 = torch.cat(embed1_list, 0)
            embed2 = torch.cat(embed2_list, 0)

        sim = Similarity(temp=0.05)

        # add normalization
        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)

        cos_sim = sim(embed1.unsqueeze(1), embed2.unsqueeze(0))

        if has_hard_negative:
            embed1_embed3_cos = sim(embed1.unsqueeze(1), embed3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)
        
        nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        loss = loss_fct(cos_sim, nce_labels)
        return SequenceClassifierOutput(loss=loss)