from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionMlp, PatchMerger, flash_attn_varlen_func, Qwen2VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import rotate_half

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    q = q.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    return q_embed

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
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), cos, sin)
        q = q.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        max_seqlen_kv = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens_kv, max_seqlen, max_seqlen_kv).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2ContextVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        assert attn_implementation == "flash_attention_2", "Unsupport attention implementation in ViT."
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.cross_attn = ContextCrossAttentionFlashAttention2(config.embed_dim, num_heads=config.num_heads)

        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)
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

class Qwen2ContextVisionTransformerPretrainedModel(Qwen2VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.context_layers = nn.ModuleList(
            [Qwen2ContextVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.ctx_merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
        )
    
    def forward(
            self, 
            pixel_values, 
            grid_thw, 
            id_dict=None,
            group_imgs=None,
            enc_dec_arch=False, 
        ):
        justfull_image_num = len(id_dict.justfull)
        full_pixel_values = torch.cat([group_imgs[k]['pixel_values'] for k in ("justfull", "concat_full", "crop_full")], dim=0)
        full_grid_thw = torch.cat([group_imgs[k]['image_grid_thw'] for k in ("justfull", "concat_full", "crop_full")], dim=0)
        full_image_feature_list = self.extract_feature(
            full_pixel_values,
            full_grid_thw,
            output_hidden_states=enc_dec_arch
        )
        full_image_feature = full_image_feature_list[-1] if enc_dec_arch else full_image_feature_list

        if len(grid_thw) == justfull_image_num:
            return self.patch_merge(full_image_feature)

        crop_pixel_values = torch.cat([group_imgs[k]['pixel_values'] for k in ("concat_crop", "crop_crop")], dim=0)
        crop_grid_thw = torch.cat([group_imgs[k]['image_grid_thw'] for k in ("concat_crop", "crop_crop")], dim=0)

        context_thw = full_grid_thw[justfull_image_num:, :]

        justfull_token_nums = group_imgs["justfull"]['image_grid_thw'].prod(1).sum().item() if justfull_image_num > 0 else 0

        if enc_dec_arch:
            context_feature = [f[justfull_token_nums:] for f in full_image_feature_list]
        else:
            context_feature = full_image_feature[justfull_token_nums:]

        cimage_features = self.extract_feature(
            crop_pixel_values,
            crop_grid_thw,
            context_feature=context_feature,
            context_thw=context_thw,
        )
        cimage_features = self.ctx_merger(cimage_features)

        if len(grid_thw) == len(id_dict.crop_crop):
            return cimage_features
        
        crop_full_image_num = len(id_dict.crop_full)
        crop_full_token_num = group_imgs["crop_full"]["image_grid_thw"].prod(1).sum().item() if crop_full_image_num > 0 else 0
        
        all_feature = torch.cat([
            self.patch_merge(full_image_feature[:len(full_image_feature)-crop_full_token_num]),
            cimage_features,
        ], dim=0)

        all_grid_thw = torch.cat([
            full_grid_thw[:len(full_grid_thw)-crop_full_image_num],
            crop_grid_thw,
        ], dim=0)

        split_sizes = (all_grid_thw.prod(dim=1) // (self.config.spatial_merge_size ** 2)).tolist()

        # the id_dict... order is important!
        shuffled_image_indices = torch.tensor(id_dict.justfull + id_dict.concat_full + id_dict.concat_crop + id_dict.crop_crop, device=all_feature.device)
        sort_indices = torch.argsort(shuffled_image_indices)

        feature_blocks_list = torch.split(all_feature, split_sizes, dim=0)
        
        all_feature = torch.cat([feature_blocks_list[i] for i in sort_indices], dim=0)
        return all_feature
    
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
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = self.gen_cu_seqlens(grid_thw)

        if context_feature is not None:
            cu_seqlens_kv = self.gen_cu_seqlens(context_thw)

        for layer_num, blk in enumerate(self.blocks):
            # if self.gradient_checkpointing and self.training:
            #     hidden_states = self._gradient_checkpointing_func(
            #         blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            #     )
            # else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

            if context_feature is not None:
                context_layer = self.context_layers[layer_num]
                cur_layer_ctx_feature = context_feature[layer_num] if isinstance(context_feature, list) else context_feature
                # if self.gradient_checkpointing and self.training:
                #     hidden_states = self._gradient_checkpointing_func(
                #         context_layer.__call__, hidden_states, cu_seqlens_now, cu_seqlens_kv_now, None, position_embeddings, context_feature
                #     )
                # else:
                # BUG : cannot backward when using gradient checkpointing
                hidden_states = context_layer(hidden_states, cu_seqlens=cu_seqlens, cu_seqlens_kv=cu_seqlens_kv, position_embeddings=position_embeddings, context_feature=cur_layer_ctx_feature)

            if output_hidden_states:
                middle_hidden_states.append(hidden_states)

        if output_hidden_states:
            return tuple(middle_hidden_states)
        else: 
            return hidden_states

    def patch_merge(self, hidden_states, window_index=None):
        hidden_states = self.merger(hidden_states)
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