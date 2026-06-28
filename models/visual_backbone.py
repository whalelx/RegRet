from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel,Qwen2_5_VLPatchMerger, Qwen2_5_VLVisionBlock, Qwen2RMSNorm, flash_attn_varlen_func
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLMLP  as Qwen2_5_ContextVisionMLP
from transformers.modeling_flash_attention_utils import apply_rotary_emb

def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    return q_embed

class Qwen2_5_ContextVisionConfig(Qwen2_5_VLVisionConfig):
    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        residual_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            depth,
            hidden_size,
            hidden_act,
            intermediate_size,
            num_heads,
            in_channels,
            patch_size,
            spatial_merge_size,
            temporal_patch_size,
            tokens_per_second,
            window_size,
            fullatt_block_indexes,
            out_hidden_size,
            **kwargs
        )

        # context provider

        self.residual_dropout = residual_dropout # 0.0
        # self.context_image_as_queries = context_image_as_queries

        # the `num_hidden_layers` should be the same as the one in the vision tower
        # self.context_provider_layer_indices = context_provider_layer_indices

        # self.masked_cross_attn = masked_cross_attn False
        # If enabled, crop_position_embedding (delta to full pos) will be updated during training.
        # self.trainable_crop_position_embedding = trainable_crop_position_embedding useless True
        # If enabled, crop_position_embedding (delta to full pos) will be a single embedding for all positions.
        # self.crop_position_single_embedding = crop_position_single_embedding False
        # add: delta. replace: do not add the original positional embedding
        # self.crop_embedding_mode = crop_embedding_mode

        # If True, the input image will be treated as a cimage (with mask as full 1s)
        self.treat_image_as_cimage = treat_image_as_cimage

class ContextCrossAttentionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)
        self.q_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        context_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        seq_length_kv = context_feature.shape[0]
        q = self.q_proj(hidden_states).reshape(seq_length, 1, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)[0]
        k, v = self.kv_proj(context_feature).reshape(seq_length_kv, 2, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        # HACK dont use pos embed for context feature for now!
        q = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), cos, sin)
        q = q.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        max_seqlen_kv = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens_kv, max_seqlen, max_seqlen_kv).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_ContextVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        assert attn_implementation == "flash_attention_2", "Unsupport attention implementation in ViT."
        self.cross_attn = ContextCrossAttentionFlashAttention2(config.hidden_size, num_heads=config.num_heads)
        self.mlp = Qwen2_5_ContextVisionMLP(config, bias=True)
        # self.residual_dropout = nn.Dropout(0.0) # TODO 0.0

        self.register_parameter("attn_factor", nn.Parameter(torch.zeros((1,)).view(())))
        self.register_parameter("mlp_factor", nn.Parameter(torch.zeros((1,)).view(())))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        context_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.cross_attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            cu_seqlens_kv=cu_seqlens_kv,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            context_feature=context_feature,
        )

        # NOTE Dropping the residual: let the model leverage more on the context
        # hidden_states = self.residual_dropout(residual) + self.attn_factor * hidden_states
        hidden_states = residual + self.attn_factor * hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        hidden_states = residual + self.mlp_factor * hidden_states

        return hidden_states

class Qwen2_5_ContextVisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.context_layers = nn.ModuleList(
            [Qwen2_5_ContextVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
    
    def forward(
            self, 
            pixel_values, 
            grid_thw, 
            focal_pixel_values, 
            focal_image_grid_thw,
            focal_image_ids,
            enc_dec_arch=False, 
        ):

        full_image_feature_list, full_image_winidx = self.extract_feature(
            pixel_values,
            grid_thw,
            output_hidden_states=enc_dec_arch
        )
        full_image_feature = full_image_feature_list[-1] if enc_dec_arch else full_image_feature_list
        
        if focal_image_ids is None or focal_image_ids.size(0) == 0:
            return self.patch_merge(full_image_feature, full_image_winidx)
        
        seq_len = full_image_feature.shape[0]
        focal_image_ids = focal_image_ids.to(full_image_feature.device)

        nofocal_image_nums = focal_image_ids[0].item()
        context_thw = grid_thw[focal_image_ids, :]
        nofocal_token_nums = (grid_thw.prod(1).sum() - context_thw.prod(1).sum()).item()
        if enc_dec_arch:
            context_feature = [f[nofocal_token_nums:] for f in full_image_feature_list]
        else:
            context_feature = full_image_feature[nofocal_token_nums:]

        cimage_features, cimage_winidx = self.extract_feature(
            focal_pixel_values,
            focal_image_grid_thw,
            context_feature=context_feature,
            context_thw=context_thw,
        )
        cimage_features = self.patch_merge(cimage_features, cimage_winidx)

        full_image_feature = full_image_feature[:nofocal_token_nums]
        
        if full_image_feature.shape[0] == 0:
            return cimage_features
        else:
            # HACK suppose cimages are always behind the full images
            full_image_winidx, _ = self.get_window_index(grid_thw[:nofocal_image_nums])
            full_image_feature = self.patch_merge(full_image_feature, full_image_winidx)
            final_image_feature = torch.cat([
                full_image_feature,
                cimage_features
            ], dim=0)

            return final_image_feature
    
    def extract_feature(
            self,
            hidden_states: torch.Tensor,
            grid_thw: torch.Tensor, 
            context_feature: torch.Tensor = None,
            context_thw: torch.Tensor = None,
            output_hidden_states=False, 
        ) -> torch.Tensor:
        ''' 
        The original forward function of Qwen2_5_VisionTransformer.
        '''
        middle_hidden_states = []
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = self.gen_cu_seqlens(grid_thw)

        if context_feature is not None:
            window_index_kv, cu_window_seqlens_kv = self.get_window_index(context_thw)
            cu_window_seqlens_kv = torch.tensor(
                cu_window_seqlens_kv,
                device=hidden_states.device,
                dtype=context_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            cu_window_seqlens_kv = torch.unique_consecutive(cu_window_seqlens_kv)
            cu_seqlens_kv = self.gen_cu_seqlens(context_thw)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            # if self.gradient_checkpointing and self.training:
            #     hidden_states = self._gradient_checkpointing_func(
            #         blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            #     )
            # else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

            if context_feature is not None:
                # have to use full attention TODO
                cu_seqlens_kv_now = cu_seqlens_kv
                cu_seqlens_now = cu_seqlens
                context_layer = self.context_layers[layer_num]
                cur_layer_ctx_feature = context_feature[layer_num] if isinstance(context_feature, list) else context_feature
                # if self.gradient_checkpointing and self.training:
                #     hidden_states = self._gradient_checkpointing_func(
                #         context_layer.__call__, hidden_states, cu_seqlens_now, cu_seqlens_kv_now, None, position_embeddings, context_feature
                #     )
                # else:
                # BUG : cannot backward when using gradient checkpointing
                hidden_states = context_layer(hidden_states, cu_seqlens=cu_seqlens_now, cu_seqlens_kv=cu_seqlens_kv_now, position_embeddings=position_embeddings, context_feature=cur_layer_ctx_feature)

            if output_hidden_states:
                middle_hidden_states.append(hidden_states)

        if output_hidden_states:
            return tuple(middle_hidden_states), window_index
        else: 
            return hidden_states, window_index
    
    def patch_merge(self, hidden_states, window_index):
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def gen_cu_seqlens(self, grid_thw):
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        # add a zero before it
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        return cu_seqlens

    def gen_ctx_feature_mask(self, seq_len, focal_img_ids, grid_thw):
        cu_seqlens = self.gen_cu_seqlens(grid_thw)
        
        start_ids = cu_seqlens[focal_img_ids]
        end_ids = cu_seqlens[focal_img_ids + 1]

        mask = torch.zeros(seq_len, dtype=torch.int32, device=focal_img_ids.device)
        mask.index_add_(0, start_ids, torch.ones_like(start_ids, dtype=torch.int32))
        valid_end_ids_mask = end_ids < seq_len
        valid_end_ids = end_ids[valid_end_ids_mask]
        if valid_end_ids.numel() > 0:
            mask.index_add_(0, valid_end_ids, -torch.ones_like(valid_end_ids, dtype=torch.int32))
        return mask.cumsum(0) > 0