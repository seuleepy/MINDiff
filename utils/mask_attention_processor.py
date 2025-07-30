import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional
import sys
import os

from diffusers.utils import deprecate


class MaskAttnProcessor2_0(torch.nn.Module):

    mask = None

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        mask_resolution=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch2.0, to use it, please upgrade yout PyTorch to 2.0"
            )

        if cross_attention_dim is not None and mask_resolution is None:
            raise ValueError(
                "mask_resolution must be provided for cross-attention processing."
            )

        self.mask_resolution = mask_resolution

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        where_idx: Optional[torch.Tensor] = None,
        sub_encoder_hidden_states: Optional[torch.Tensor] = None,
        attn_scale: Optional[float] = 1.0,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if encoder_hidden_states is None:
            crossattn = False
        else:
            crossattn = True

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if crossattn:
            sub_key = attn.to_k(sub_encoder_hidden_states)
            sub_value = attn.to_v(sub_encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if crossattn:
            sub_key = sub_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            sub_value = sub_value.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if crossattn:
            sub_hidden_states = F.scaled_dot_product_attention(
                query,
                sub_key,
                sub_value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

            query_size = int(math.sqrt(query.size(-2)))

            if query_size == self.mask_resolution:
                # get attention map
                scale_factor = 1 / math.sqrt(query.size(-1))
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_map = attn_weight.softmax(dim=-1)
                attn_map = attn_map[attn_map.size(0) // 2 :, :, :, :].mean(
                    dim=1
                )  # conditional, mean (heads)
                attn_map = attn_map[torch.arange(attn_map.size(0)), :, where_idx]
                attn_map = attn_map.view(
                    -1,
                    self.mask_resolution,
                    self.mask_resolution,
                )

                if not hasattr(MaskAttnProcessor2_0, "mask_buffer"):
                    MaskAttnProcessor2_0.mask_buffer = torch.empty(
                        (5, *attn_map.shape), device=query.device
                    )
                    MaskAttnProcessor2_0.buffer_idx = 0

                MaskAttnProcessor2_0.mask_buffer[MaskAttnProcessor2_0.buffer_idx] = (
                    attn_map
                )
                MaskAttnProcessor2_0.buffer_idx += 1

                if MaskAttnProcessor2_0.buffer_idx == 5:
                    mask = MaskAttnProcessor2_0.mask_buffer.mean(dim=0)
                    MaskAttnProcessor2_0.mask = (mask > mask.mean()).to(query.dtype)
                    MaskAttnProcessor2_0.buffer_idx = 0

            if MaskAttnProcessor2_0.mask is not None:
                mask = MaskAttnProcessor2_0.mask.unsqueeze(1)
                mask = F.interpolate(
                    mask, (query_size, query_size), mode="nearest-exact"
                )

                mask = mask.repeat(2, 1, 1, 1)
                mask = mask.view(-1, 1, query_size**2, 1)

                hidden_states = (
                    hidden_states - attn_scale * (1 - mask) * sub_hidden_states
                )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
