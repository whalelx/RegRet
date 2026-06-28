from typing import Tuple, Optional, List, Union
import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from torch import nn
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
from dataclasses import dataclass
from .visual_backbone import Qwen2_5_ContextVisionTransformerPretrainedModel

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

@dataclass
class ExtraLossOutput(SequenceClassifierOutput):
    loss_emb: torch.FloatTensor = None
    loss_gen: torch.FloatTensor = None

class Qwen2_5_VLRetForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.emb_head = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self._init_embhead_weights()

        self.visual = Qwen2_5_ContextVisionTransformerPretrainedModel._from_config(config.vision_config)

    def _init_embhead_weights(self):
        nn.init.constant_(self.emb_head.weight, 0.01)
        nn.init.constant_(self.emb_head.bias, 0)

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
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        inference=False,
        has_hard_negative=False,
        qids=None,
        dids=None,
        ids=None,
        focal_pixel_values=None,
        focal_image_grid_thw=None,
        focal_image_ids=None,
        focal_pixel_values_videos=None,
        focal_video_grid_thw=None,
        real_image_grid_thw=None,
        reverse_idx = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw, focal_pixel_values=focal_pixel_values, focal_image_grid_thw=focal_image_grid_thw, focal_image_ids=focal_image_ids)
                # breakpoint()
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    real_image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        embed_index = self.config.emb_token_ids[0]

        if labels is None and not inference: # HACK inference==True means using embedding as output
            logits = self.lm_head(hidden_states)
            return Qwen2_5_VLCausalLMOutputWithPast(
                loss=0,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )
        
        # language generation
        language_loss = None
        language_indices = torch.where((labels != embed_index).all(1))[0]
        if language_indices is not None and len(language_indices) > 0:
            logits = self.lm_head(hidden_states[language_indices])
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[language_indices][..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                language_loss = loss_fct(shift_logits, shift_labels)

        # contrastive learning
        contrastive_loss = None
        contrastive_indices = torch.where(labels == embed_index)[0]
        contrastive_labels = labels[contrastive_indices] if labels is not None else None

        if contrastive_labels is not None and len(contrastive_indices)>0:
            # if self.config.use_emb_head:
            contrastive_hidden_states = self.emb_head(hidden_states[contrastive_indices])
            # else:
            #     contrastive_hidden_states = hidden_states[contrastive_indices]

            if has_hard_negative:
                contrastive_batch_size = len(contrastive_hidden_states) // 3
            elif not inference:
                contrastive_batch_size = len(contrastive_hidden_states) // 2
            elif inference:
                contrastive_batch_size = len(contrastive_hidden_states)

            embed_indices = torch.argmax((contrastive_labels == embed_index).int(), dim=1)
            embed_features = contrastive_hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)

            if inference:
                if ids is not None:
                    return embed_features, ids
                elif qids is not None or dids is not None:
                    return embed_features, qids, dids
                return embed_features

            if has_hard_negative:
                raise Exception("do not support yet!")
                embed1, embed2, embed3 = embed_features[:contrastive_batch_size], embed_features[contrastive_batch_size:2*contrastive_batch_size], embed_features[2*contrastive_batch_size:]
            else:
                embed_features = embed_features[reverse_idx.to(embed_features.device)]
                embed1, embed2 = embed_features[:contrastive_batch_size], embed_features[contrastive_batch_size:]

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
                embed3 = F.normalize(embed3, dim=-1)
                embed1_embed3_cos = sim(embed1.unsqueeze(1), embed3.unsqueeze(0))
                cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)

            nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
            contrastive_loss = loss_fct(cos_sim, nce_labels)

        # Combine losses
        if language_loss is not None and contrastive_loss is not None:
            language_loss *= self.config.language_loss_weight
            total_loss = language_loss + contrastive_loss
        elif language_loss is not None:
            total_loss = language_loss
        elif contrastive_loss is not None:
            total_loss = contrastive_loss
        else:
            total_loss = None

        return ExtraLossOutput(loss=total_loss, loss_gen=language_loss, loss_emb=contrastive_loss)