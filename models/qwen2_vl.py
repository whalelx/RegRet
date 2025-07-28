from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, PreTrainedTokenizer
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLCausalLMOutputWithPast
from torch import nn
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
from dataclasses import dataclass
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

class AngleSimilarity(nn.Module):
    """
    Angle similarity for complex (re+im) embeddings.
    Calculates a pairwise similarity matrix.
    """
    def __init__(self, temp=0.05, pooling_strategy='sum'):
        super().__init__()
        self.temp = temp
        assert pooling_strategy in ('sum', 'mean')
        self.pooling_strategy = pooling_strategy

    def forward(self, x, y=None):
        """
        Calculates pairwise angle similarity.
        If y is None, computes pairwise similarity for x with itself.
        
        Args:
            x (torch.Tensor): A batch of embeddings, shape [bs1, dim].
            y (torch.Tensor, optional): Another batch of embeddings, shape [bs2, dim]. Defaults to None.

        Returns:
            torch.Tensor: A pairwise similarity matrix, shape [bs1, bs2].
        """
        if y is None:
            y = x

        # 1. Split embeddings into real and imaginary parts
        # x: [bs1, dim] -> x_re, x_im: [bs1, dim/2]
        # y: [bs2, dim] -> y_re, y_im: [bs2, dim/2]
        x_re, x_im = torch.chunk(x, 2, dim=-1)
        y_re, y_im = torch.chunk(y, 2, dim=-1)

        # 2. Calculate complex division z = x / y for all pairs
        # This is equivalent to z = (x * y_conj) / |y|^2
        # Numerator: x * y_conj = (x_re + i*x_im) * (y_re - i*y_im)
        # = (x_re*y_re + x_im*y_im) + i*(x_im*y_re - x_re*y_im)
        
        # Use matrix multiplication for pairwise calculation
        # Resulting shapes: [bs1, bs2]
        re_num = x_re @ y_re.T + x_im @ y_im.T
        im_num = x_im @ y_re.T - x_re @ y_im.T

        # Denominator: |y|^2 = y_re^2 + y_im^2, summed over the feature dimension
        # Resulting shape: [bs2]
        y_norm_sq = torch.sum(y_re**2 + y_im**2, dim=-1)
        
        # Add epsilon for numerical stability. Broadcasting handles the division.
        z_denom = y_norm_sq + 1e-8
        
        re = re_num / z_denom
        im = im_num / z_denom

        # 3. Amplitude normalization (optional but recommended)
        # This step normalizes out the magnitude difference |x|/|y|, 
        # leaving only the pure angle difference information.
        # dz: magnitude of x vectors, shape [bs1]
        # dw: magnitude of y vectors, shape [bs2]
        dz = torch.sqrt(torch.sum(x_re**2 + x_im**2, dim=-1) + 1e-8)
        dw = torch.sqrt(y_norm_sq + 1e-8) # Reuse y_norm_sq for efficiency

        # Create a [bs1, bs2] normalization matrix via broadcasting
        norm_factor = dz[:, None] / dw[None, :]
        
        re = re / norm_factor
        im = im / norm_factor

        # 4. Pooling strategy
        # re and im are now the cos and sin of the angle differences.
        # We can sum or mean them to get a final score.
        if self.pooling_strategy == 'sum':
            pooled = re + im
        else: # 'mean'
            pooled = (re + im) / 2
        
        # 5. Final similarity score
        # The angle delta can be approximated by this pooled value.
        sim_angle = torch.abs(pooled) / self.temp
        
        return sim_angle

@dataclass
class ExtraLossOutput(SequenceClassifierOutput):
    loss_emb: torch.FloatTensor = None
    loss_gen: torch.FloatTensor = None

class Qwen2VLRetForConditionalGeneration(Qwen2VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2ContextVisionTransformerPretrainedModel._from_config(config.vision_config)

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
        )

        hidden_states = outputs[0]

        embed_index = self.config.emb_token_ids[0]

        if labels is None and not inference: # HACK inference==True means using embedding as output
            logits = self.lm_head(hidden_states)
            return Qwen2VLCausalLMOutputWithPast(
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
            if self.config.use_emb_head:
                contrastive_hidden_states = self.emb_head(hidden_states[contrastive_indices])
            else:
                contrastive_hidden_states = hidden_states[contrastive_indices]

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
            anglesim = AngleSimilarity()

            # add normalization
            embed1 = F.normalize(embed1, dim=-1)
            embed2 = F.normalize(embed2, dim=-1)

            cos_sim = anglesim(embed1, embed2) if self.config.getattr('use_angle_sim', False) else sim(embed1.unsqueeze(1), embed2.unsqueeze(0))

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