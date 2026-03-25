from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, repeat_kv

flash_attn_func = None
try:
    from flash_attn import flash_attn_func
except Exception:
    flash_attn_func = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(0).unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class Qwen2SdpaAttention(Qwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_key_value = (
                torch.cat([past_key_value[0], key_states], dim=2),
                torch.cat([past_key_value[1], value_states], dim=2),
            )
        else:
            past_key_value = (key_states, value_states)

        key_states, value_states = past_key_value
        kv_seq_len = value_states.size(2)

        if hasattr(self, "use_retrieval") and self.retrieval_topk > 0:
            retrieval_topk = self.retrieval_topk
            del self.use_retrieval
            del self.retrieval_topk

            img_mask = torch.tensor([1 if item == "I" else 0 for item in self.cache_modalities], dtype=torch.long, device=query_states.device)
            cache_lengths = list(self.cache_lengths) + [query_states.size(2)]
            split_key_states = torch.split(repeat_kv(key_states, self.num_key_value_groups), cache_lengths, dim=2)[:-1]
            sub_key_reprs = torch.cat([state.mean(dim=2).flatten(1, 2) for state in split_key_states], dim=0)
            query_repr = query_states.mean(dim=2).flatten(1, 2)

            query_subkey_sims = torch.cosine_similarity(query_repr, sub_key_reprs, dim=-1) * img_mask
            topk_indices = torch.topk(query_subkey_sims, min(retrieval_topk, query_subkey_sims.size(0)), dim=-1).indices.tolist()

            split_key_states = torch.split(key_states, cache_lengths, dim=2)
            split_value_states = torch.split(value_states, cache_lengths, dim=2)

            retrieved_key_states = []
            retrieved_value_states = []
            for block_idx in range(len(self.cache_modalities)):
                if block_idx in topk_indices or self.cache_modalities[block_idx] == "T":
                    retrieved_key_states.append(split_key_states[block_idx])
                    retrieved_value_states.append(split_value_states[block_idx])
            retrieved_key_states.append(split_key_states[-1])
            retrieved_value_states.append(split_value_states[-1])

            key_states = torch.cat(retrieved_key_states, dim=2)
            value_states = torch.cat(retrieved_value_states, dim=2)
            kv_seq_len = key_states.size(2)
            past_key_value = (key_states, value_states)
            if attention_mask is not None:
                attention_mask = attention_mask[..., -kv_seq_len:].contiguous()

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states = apply_rotary_pos_emb(query_states, cos[-q_len:], sin[-q_len:])
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None and attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(f"Attention mask should be {(bsz, 1, q_len, kv_seq_len)}, got {attention_mask.size()}")

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if flash_attn_func is None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal and attention_mask is None and q_len > 1,
            )
        else:
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
            ).transpose(1, 2)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


def cambrian_qwen2_forward(
    self,
    input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds,
    use_cache,
    output_attentions,
    output_hidden_states,
    return_dict,
):
    assert use_cache is True
    assert output_attentions is False
    assert output_hidden_states is True
    assert return_dict is True

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    kv_cache = tuple()
    batch_size, seq_length, _ = inputs_embeds.size()
    past_key_values_length = 0 if past_key_values is None else past_key_values[0][0].size(2)

    attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
        sliding_window=self.config.sliding_window,
    )

    hidden_states = inputs_embeds
    for idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[idx] if past_key_values is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        kv_cache += (layer_outputs[1],)
        hidden_states = layer_outputs[0]

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=kv_cache,
        hidden_states=hidden_states,
    )
