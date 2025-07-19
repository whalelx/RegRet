from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionMlp, Qwen2VLVisionBlock, flash_attn_varlen_func, Qwen2VisionTransformerPretrainedModel
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
    
    def forward(
            self, 
            pixel_values, 
            grid_thw, 
            focal_pixel_values, 
            focal_image_grid_thw,
            focal_image_ids,
            enc_dec_arch=False, 
        ):

        full_image_feature_list = self.extract_feature(
            pixel_values,
            grid_thw,
            output_hidden_states=enc_dec_arch
        )
        full_image_feature = full_image_feature_list[-1] if enc_dec_arch else full_image_feature_list
        
        if focal_image_ids is None or focal_image_ids.size(0) == 0:
            return self.patch_merge(full_image_feature)
        
        seq_len = full_image_feature.shape[0]
        focal_image_ids = focal_image_ids.to(full_image_feature.device)

        nofocal_image_nums = focal_image_ids[0].item()
        context_thw = grid_thw[focal_image_ids, :]
        context_token_nums = (grid_thw.prod(1).sum() - context_thw.prod(1).sum()).item()
        if enc_dec_arch:
            context_feature = [f[context_token_nums:] for f in full_image_feature_list]
        else:
            context_feature = full_image_feature[context_token_nums:]

        cimage_features = self.extract_feature(
            focal_pixel_values,
            focal_image_grid_thw,
            context_feature=context_feature,
            context_thw=context_thw,
        )
        cimage_features = self.patch_merge(cimage_features)

        full_image_feature = full_image_feature[:context_token_nums]
        
        if full_image_feature.shape[0] == 0:
            return cimage_features
        else:
            # HACK suppose cimages are always behind the full images
            full_image_feature = self.patch_merge(full_image_feature)
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