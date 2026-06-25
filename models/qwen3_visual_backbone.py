from typing import Optional, Tuple, Union, List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionMLP,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionModel,
    BaseModelOutputWithDeepstackFeatures,
    rotate_half,
)
from transformers.utils import logging
from flash_attn import flash_attn_varlen_func

logger = logging.get_logger(__name__)
LAYERWISE = bool(int(os.environ.get("LAYERWISE", "0")))


def apply_rotary_pos_emb_vision_q(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Q-only RoPE (used for cross attention; K/V come from a separate context)."""
    orig_q_dtype = q.dtype
    q = q.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed.to(orig_q_dtype)


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
        # HACK: dont use pos embed for context feature for now!
        q = apply_rotary_pos_emb_vision_q(q.unsqueeze(0), cos, sin)
        q = q.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        max_seqlen_kv = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens_kv, max_seqlen, max_seqlen_kv
        ).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3ContextVisionBlock(nn.Module):
    """A cross-attention block injected after each Qwen3VLVisionBlock to fuse context features."""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.cross_attn = ContextCrossAttentionFlashAttention2(
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen3VLVisionMLP(config)
        # Initialised to zero so the block behaves as identity at the start of training.
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
        hidden_states = residual + self.attn_factor * hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        hidden_states = residual + self.mlp_factor * hidden_states

        return hidden_states


class Qwen3ContextVisionTransformerPretrainedModel(Qwen3VLVisionModel):
    """Qwen3VL vision encoder enhanced with cross-attention context blocks.

    Differences from the base ``Qwen3VLVisionModel``:

    * One extra ``Qwen3ContextVisionBlock`` is appended to each ViT layer; when a
      ``context_feature`` is provided (typically a feature sequence of the
      corresponding "full" image), the block fuses it into the local hidden
      states via cross attention.
    * For the deep-stack mechanism, the hidden states *after* each context block
      are merged through a dedicated ``ctx_deepstack_merger_list`` so that the
      LLM still receives the same number of feature maps but enriched with
      context information.
    * The forward method also implements our dual-image batched flow that
      groups images into ``justfull / concat_full / crop_full / concat_crop /
      crop_crop`` according to ``id_dict``.
    """

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        # One context layer for every visual layer.
        self.context_layers = nn.ModuleList(
            [Qwen3ContextVisionBlock(config) for _ in range(config.depth)]
        )
        # Final patch merger for the context (crop) branch. Use the same
        # signature as ``self.merger`` (no post-shuffle norm).
        self.ctx_merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )
        # Deep-stack mergers for the context branch, paired one-to-one with
        # ``self.deepstack_merger_list``.
        self.ctx_deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True)
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def gen_cu_seqlens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # FA2 requires int32 for cu_seqlens
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        return cu_seqlens

    def patch_merge(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.merger(hidden_states)

    # ------------------------------------------------------------------ #
    # Core feature extraction                                            #
    # ------------------------------------------------------------------ #
    def extract_feature(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        context_feature: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        context_thw: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        """Run the ViT (with optional context cross-attention) and collect deep-stack features.

        Returns a tuple ``(hidden_states, deepstack_features)``.

        * ``hidden_states`` is either the final per-token hidden states or, when
          ``output_hidden_states`` is True, a tuple containing the per-layer
          hidden states.
        * ``deepstack_features`` is a list of tensors aligned with
          ``self.deepstack_visual_indexes``; each tensor is already merged via
          the corresponding (context-aware or vanilla) PatchMerger.
        """
        is_context = context_feature is not None

        # Patch embed + Qwen3VL absolute pos embed (interpolated)
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.size(0)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = self.gen_cu_seqlens(grid_thw)
        cu_seqlens_kv = self.gen_cu_seqlens(context_thw) if is_context else None

        middle_hidden_states = []
        deepstack_feature_lists: List[torch.Tensor] = []

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

            if is_context:
                context_layer = self.context_layers[layer_num]
                cur_layer_ctx_feature = (
                    context_feature[layer_num]
                    if isinstance(context_feature, (list, tuple))
                    else context_feature
                )
                # NOTE: gradient checkpointing on context layer breaks backward
                # for now (see legacy Qwen2 implementation), so it is disabled.
                hidden_states = context_layer(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens_kv,
                    position_embeddings=position_embeddings,
                    context_feature=cur_layer_ctx_feature,
                )

            # Deep-stack: harvest features at predefined indexes (using
            # context-aware mergers when in the context branch).
            if layer_num in self.deepstack_visual_indexes:
                merger_idx = self.deepstack_visual_indexes.index(layer_num)
                merger = (
                    self.ctx_deepstack_merger_list[merger_idx]
                    if is_context
                    else self.deepstack_merger_list[merger_idx]
                )
                deepstack_feature_lists.append(merger(hidden_states))

            if output_hidden_states:
                middle_hidden_states.append(hidden_states)

        if output_hidden_states:
            return tuple(middle_hidden_states), deepstack_feature_lists
        return hidden_states, deepstack_feature_lists

    # ------------------------------------------------------------------ #
    # Top-level forward (handles dual-image batched dataflow)            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        pixel_values,
        grid_thw,
        id_dict=None,
        group_imgs=None,
        enc_dec_arch: bool = LAYERWISE,
        **kwargs,
    ) -> BaseModelOutputWithDeepstackFeatures:
        # Helper: collect non-None tensors from group_imgs for a given set of keys
        def _gather(keys, field):
            return [group_imgs[k][field] for k in keys if group_imgs[k][field] is not None]

        # ----- 1. Run ViT for "full" images (no context) --------------- #
        justfull_image_num = len(id_dict.justfull)
        full_pixel_values = torch.cat(
            _gather(("justfull", "concat_full", "crop_full"), 'pixel_values'), dim=0
        )
        full_grid_thw = torch.cat(
            _gather(("justfull", "concat_full", "crop_full"), 'image_grid_thw'), dim=0
        )
        full_image_feature_list, full_deepstack_features = self.extract_feature(
            full_pixel_values,
            full_grid_thw,
            output_hidden_states=enc_dec_arch,
        )
        full_image_feature = full_image_feature_list[-1] if enc_dec_arch else full_image_feature_list

        # ----- 2. Fast path: only "justfull" images present ------------ #
        if len(grid_thw) == justfull_image_num:
            return BaseModelOutputWithDeepstackFeatures(
                last_hidden_state=full_image_feature,
                pooler_output=self.patch_merge(full_image_feature),
                deepstack_features=full_deepstack_features,
            )

        # ----- 3. Run ViT for "crop" images with context cross-attn ---- #
        crop_pixel_values = torch.cat(
            _gather(("concat_crop", "crop_crop"), 'pixel_values'), dim=0
        )
        crop_grid_thw = torch.cat(
            _gather(("concat_crop", "crop_crop"), 'image_grid_thw'), dim=0
        )

        context_thw = full_grid_thw[justfull_image_num:, :]
        justfull_token_nums = (
            group_imgs["justfull"]['image_grid_thw'].prod(1).sum().item()
            if justfull_image_num > 0
            else 0
        )

        if enc_dec_arch:
            context_feature = [f[justfull_token_nums:] for f in full_image_feature_list]
        else:
            context_feature = full_image_feature[justfull_token_nums:]

        crop_hidden_states, crop_deepstack_features = self.extract_feature(
            crop_pixel_values,
            crop_grid_thw,
            context_feature=context_feature,
            context_thw=context_thw,
        )
        cimage_features = self.ctx_merger(crop_hidden_states)

        # ----- 4. Fast path: only "crop_crop" images present ----------- #
        if len(grid_thw) == len(id_dict.crop_crop):
            return BaseModelOutputWithDeepstackFeatures(
                last_hidden_state=crop_hidden_states,
                pooler_output=cimage_features,
                deepstack_features=crop_deepstack_features,
            )

        # ----- 5. Mixed batch: stitch full + crop and re-order --------- #
        crop_full_image_num = len(id_dict.crop_full)
        crop_full_token_num = (
            group_imgs["crop_full"]["image_grid_thw"].prod(1).sum().item()
            if crop_full_image_num > 0
            else 0
        )
        merge_unit = self.spatial_merge_size ** 2

        # 5.1 Pooler output (image-level patch-merged features)
        full_pooler = self.patch_merge(
            full_image_feature[: full_image_feature.shape[0] - crop_full_token_num]
        )
        all_pooler = torch.cat([full_pooler, cimage_features], dim=0)

        # 5.2 Deep-stack features (already patch-merged inside extract_feature)
        crop_full_post_merge_token_num = crop_full_token_num // merge_unit
        all_deepstack_features: List[torch.Tensor] = []
        for full_ds, crop_ds in zip(full_deepstack_features, crop_deepstack_features):
            full_ds = full_ds[: full_ds.shape[0] - crop_full_post_merge_token_num]
            all_deepstack_features.append(torch.cat([full_ds, crop_ds], dim=0))

        # 5.3 Build per-image split sizes (in post-merge token space)
        all_grid_thw = torch.cat(
            [
                full_grid_thw[: len(full_grid_thw) - crop_full_image_num],
                crop_grid_thw,
            ],
            dim=0,
        )
        split_sizes = (all_grid_thw.prod(dim=1) // merge_unit).tolist()

        # 5.4 Re-order back to the original image order (id_dict.* order is important!)
        shuffled_image_indices = torch.tensor(
            id_dict.justfull + id_dict.concat_full + id_dict.concat_crop + id_dict.crop_crop,
            device=all_pooler.device,
        )
        sort_indices = torch.argsort(shuffled_image_indices)

        def _reorder(features: torch.Tensor) -> torch.Tensor:
            blocks = torch.split(features, split_sizes, dim=0)
            return torch.cat([blocks[i] for i in sort_indices], dim=0)

        all_pooler = _reorder(all_pooler)
        all_deepstack_features = [_reorder(f) for f in all_deepstack_features]

        return BaseModelOutputWithDeepstackFeatures(
            # The token-level last_hidden_state is no longer aligned to a single
            # grid_thw after stitching, so we drop it.
            last_hidden_state=None,
            pooler_output=all_pooler,
            deepstack_features=all_deepstack_features,
        )
