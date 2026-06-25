from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLCausalLMOutputWithPast,
    BaseModelOutputWithDeepstackFeatures,
)

from .qwen3_visual_backbone import Qwen3ContextVisionTransformerPretrainedModel

logger = logging.get_logger(__name__)


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
        x_re, x_im = torch.chunk(x, 2, dim=-1)
        y_re, y_im = torch.chunk(y, 2, dim=-1)

        # 2. Calculate complex division z = x / y for all pairs
        re_num = x_re @ y_re.T + x_im @ y_im.T
        im_num = x_im @ y_re.T - x_re @ y_im.T

        y_norm_sq = torch.sum(y_re ** 2 + y_im ** 2, dim=-1)
        z_denom = y_norm_sq + 1e-8

        re = re_num / z_denom
        im = im_num / z_denom

        # 3. Amplitude normalization
        dz = torch.sqrt(torch.sum(x_re ** 2 + x_im ** 2, dim=-1) + 1e-8)
        dw = torch.sqrt(y_norm_sq + 1e-8)

        norm_factor = dz[:, None] / dw[None, :]
        re = re / norm_factor
        im = im / norm_factor

        # 4. Pooling strategy
        if self.pooling_strategy == 'sum':
            pooled = re + im
        else:  # 'mean'
            pooled = (re + im) / 2

        # 5. Final similarity score
        sim_angle = torch.abs(pooled) / self.temp
        return sim_angle


@dataclass
class ExtraLossOutput(SequenceClassifierOutput):
    loss_emb: Optional[torch.FloatTensor] = None
    loss_gen: Optional[torch.FloatTensor] = None


class Qwen3VLRetForConditionalGeneration(Qwen3VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        # Replace the standard Qwen3VL vision encoder with our context-aware
        # variant. The new module also produces deep-stack features (one per
        # ``deepstack_visual_indexes``) that get added back to the LLM's early
        # hidden states.
        # self.model.visual = Qwen3ContextVisionTransformerPretrainedModel._from_config(
        #     config.vision_config
        # )

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

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLCausalLMOutputWithPast:


        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # ===== debug: logits -> pred ids =====
        with torch.no_grad():
            pred_ids = torch.argmax(logits, dim=-1)   # [B, T]
            print("logits shape:", logits.shape)
            print("pred_ids shape:", pred_ids.shape)
            print("pred_ids:", pred_ids)

            with torch.no_grad():
                pred_ids = torch.argmax(logits, dim=-1)
                torch.save(
                    {
                        "labels": labels.detach().cpu(),
                        "pred_ids": pred_ids.detach().cpu(),
                        "input_ids": input_ids.detach().cpu(),
                    },
                    "./debug_logits.pt"
                )

    # ===== end debug =====

        breakpoint()



        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
