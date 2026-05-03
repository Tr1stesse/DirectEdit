import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import abc
import inspect
from controller import logging
from controller import seq_aligner
from diffusers.models.embeddings import apply_rotary_emb

logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)

    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)

    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, max_num_words, 1)
    return alpha_time_words


# -----------------------------------------------------------------------------
# Norm Controller
# -----------------------------------------------------------------------------

class FluxAdalayernorm_replace(abc.ABC):
    def __init__(self, prompts, num_steps, self_replace_steps, tokenizer, tokenizer_2, device, max_len_t5=512):
        super(FluxAdalayernorm_replace, self).__init__()
        self.cur_step = 0
        self.cur_layer = 0
        self.num_adanorm = 0
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])

        self.mapper_clip, alphas_clip = seq_aligner.get_refinement_mapper(prompts, tokenizer, max_len=77)
        self.mapper_clip = self.mapper_clip.squeeze().to(device)
        alphas_clip = alphas_clip.to(device)
        self.alphas_clip = alphas_clip.reshape(alphas_clip.shape[0], alphas_clip.shape[1], 1)

        self.mapper_t5, alphas_t5 = seq_aligner.get_refinement_mapper(prompts, tokenizer_2, max_len=max_len_t5)
        self.mapper_t5 = self.mapper_t5.squeeze().to(device)
        alphas_t5 = alphas_t5.to(device)
        self.alphas_t5 = alphas_t5.reshape(alphas_t5.shape[0], alphas_t5.shape[1], 1)

        self.cross_replace_alpha_clip = get_time_words_attention_alpha(prompts, num_steps, self_replace_steps,
                                                                       tokenizer, max_num_words=77).to(device)

    def replace_adaptive_layernorm(self, base, replace, cur_step):
        '''
        base: [1, Seq, Dim]
        replace: [N, Seq, Dim]
        '''
        seq_len = base.shape[1]

        if seq_len != self.mapper_t5.shape[-1]:
            return replace
        # ===============================================

        unchange_base = base  # [1, Seq, Dim]

        select_unchange_base = unchange_base[:, self.mapper_t5, :]

        # Blend
        replace_new = select_unchange_base * self.alphas_t5 + replace * (1 - self.alphas_t5)

        return replace_new

    def forward(self, x):
        if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
            if x.shape[0] >= 2:
                x_base = x[0].unsqueeze(0)
                x_replace = x[1:]
                x[1:] = self.replace_adaptive_layernorm(x_base, x_replace, self.cur_step)
        return x

    def __call__(self, x):
        x = self.forward(x)
        self.cur_layer += 1
        if self.cur_layer == self.num_adanorm:
            self.cur_step += 1
            self.cur_layer = 0
        return x


# -----------------------------------------------------------------------------
# Attention Controller
# -----------------------------------------------------------------------------

class FluxAttentionReplace(abc.ABC):
    '''
    Controls Self-Attention injection for SingleStream blocks in Flux.
    '''

    def __init__(self, prompts, num_steps, self_replace_steps):
        super(FluxAttentionReplace, self).__init__()
        self.cur_step = 0
        self.cur_layer = 0
        self.num_att_layers = 0
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])

    def replace_attention(self, base, replace):
        return base.expand(replace.shape[0], *base.shape[1:])

    def forward_single_value(self, v):
        # Single Stream: v shape [Batch, Seq, Head, Dim]
        if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
            if v.shape[0] >= 2:
                base = v[0].unsqueeze(0)
                replace = v[1:]
                v[1:] = self.replace_attention(base, replace)
        return v

    def __call__(self, v):
        out = self.forward_single_value(v)

        self.cur_layer += 1
        if self.cur_layer == self.num_att_layers:
            self.cur_step += 1
            self.cur_layer = 0
        return out


# -----------------------------------------------------------------------------
# Processors
# -----------------------------------------------------------------------------

class FluxP2PAttnProcessor2_0:
    """
    Processor for Flux Double Stream Blocks.
    Handles concatenation of Text and Image streams for proper RoPE application.
    """

    def __init__(self, controller):
        self.controller = controller

    def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:

        batch_size = hidden_states.shape[0]

        # --- 1. Projections (Image Stream) ---
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # [Batch, Heads, Seq_Img, Dim]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # --- 2. Projections (Text Stream) ---
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # [Batch, Heads, Seq_Txt, Dim]
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # --- 3. Concatenation for RoPE ---
            query_full = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key_full = torch.cat([encoder_hidden_states_key_proj, key], dim=2)

            # --- 4. Apply RoPE ---
            if image_rotary_emb is not None:
                query_full = apply_rotary_emb(query_full, image_rotary_emb)
                key_full = apply_rotary_emb(key_full, image_rotary_emb)

            # --- 5. Controller Injection (Double Stream) ---
            value, encoder_hidden_states_value_proj = self.controller(value, encoder_hidden_states_value_proj)

            # Concatenate Values
            value_full = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        else:
            # Fallback
            query_full = query
            key_full = key
            value_full = value
            if image_rotary_emb is not None:
                query_full = apply_rotary_emb(query_full, image_rotary_emb)
                key_full = apply_rotary_emb(key_full, image_rotary_emb)

        # --- 6. SDPA ---
        hidden_states = F.scaled_dot_product_attention(query_full, key_full, value_full, dropout_p=0.0, is_causal=False)

        # [Batch, Seq_Total, Dim_Total]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # --- 7. Split Output ---
        if encoder_hidden_states is not None:
            txt_len = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :txt_len],
                hidden_states[:, txt_len:]
            )

            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if attn.to_out is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxP2PSingleAttnProcessor2_0:
    """
    Processor for Flux Single Stream Blocks.
    Injects Value features using the controller.
    """

    def __init__(self, controller):
        self.controller = controller

    def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:

        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE separately
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # -----------------------------------------------------------------------
        # Controller Injection (Single Stream)
        # -----------------------------------------------------------------------
        if self.controller is not None:
            value = self.controller(value)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if attn.to_out is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

def register_attention_control_flux(model, controller_attn, controller_norm):
    '''
    Registers the custom processors into the FluxTransformer2DModel.
    '''

    def ca_forward_single(self):
        def forward(
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                image_rotary_emb: Optional[torch.Tensor] = None,
                **kwargs,
        ) -> torch.Tensor:

            if isinstance(self.processor, FluxP2PSingleAttnProcessor2_0):
                self.processor.controller = controller_attn
            else:
                self.processor = FluxP2PSingleAttnProcessor2_0(controller_attn)

            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )

        return forward

    def patch_norm_module(norm_module):
        if hasattr(norm_module, "_original_forward"):
            norm_module.forward = norm_module._original_forward

        original_forward = norm_module.forward
        norm_module._original_forward = original_forward

        def new_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            if controller_norm is not None:
                if isinstance(out, tuple):
                    modified = controller_norm(out[0])
                    out = (modified,) + out[1:]
                else:
                    out = controller_norm(out)
            return out

        norm_module.forward = new_forward

    class DummyController:
        def __call__(self, *args):
            if len(args) == 2: return args[0], args[1]  # For Double block
            return args[0]  # For Single block

        def __init__(self):
            self.num_att_layers = 0
            self.num_adanorm = 0

    if controller_attn is None:
        controller_attn = DummyController()
    if controller_norm is None:
        controller_norm = DummyController()

    double_blocks = model.transformer.transformer_blocks
    single_blocks = model.transformer.single_transformer_blocks

    att_count = 0
    ada_norm_count = 0

    for i, block in enumerate(double_blocks):
        if hasattr(block, 'norm1_context'):
            patch_norm_module(block.norm1_context)
            ada_norm_count += 1

    for i, block in enumerate(single_blocks):
        block.attn.forward = ca_forward_single(block.attn)
        att_count += 1

    controller_attn.num_att_layers = att_count
    controller_norm.num_adanorm = ada_norm_count

    logger.info(f"Registered Flux Controller: {att_count} Single-Attn Layers, {ada_norm_count} Double-Norm Layers.")