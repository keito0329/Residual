"""
Models.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from transformers import BertConfig, BertModel

    

class PointWiseFeedForward(nn.Module):
    """Code from https://github.com/pmixer/SASRec.pytorch."""

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs



    

class GRU4Rec(nn.Module):

    def __init__(self, vocab_size, rnn_config, add_head=True,
                 tie_weights=True, padding_idx=0, init_std=0.02):

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_config = rnn_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=rnn_config['input_size'],
                                        padding_idx=padding_idx)
        self.rnn = nn.GRU(batch_first=True, bidirectional=False, **rnn_config)

        if self.add_head:
            self.head = nn.Linear(rnn_config['hidden_size'], vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight

        self.init_weights()

    def init_weights(self):

        self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.embed_layer.weight.data[self.padding_idx].zero_()

    # parameter attention mask added for compatibility with Lightning module, not used
    def forward(self, input_ids, attention_mask, output_norms=False):

        embeds = self.embed_layer(input_ids)
        outputs, _ = self.rnn(embeds)
        

        if self.add_head:
            outputs = self.head(outputs)

        return outputs
    

class SASRec(nn.Module):
    """Adaptation of code from
    https://github.com/pmixer/SASRec.pytorch.
    """

    def __init__(self, item_num, maxlen=128, hidden_units=64, num_blocks=1,
                 num_heads=1, dropout_rate=0.1, initializer_range=0.02,
                 add_head=True, padding_idx=0):

        super(SASRec, self).__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.padding_idx=padding_idx

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=self.padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(hidden_units,
                                                   num_heads,
                                                   dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # parameter attention mask added for compatibility with GPT Lightning module, not used
    def forward(self, input_ids, attention_mask):

        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        # need to be on the same device
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.Tensor(input_ids == self.padding_idx)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        # need to be on the same device
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, Q, Q, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        outputs = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        if self.add_head:
            outputs = torch.matmul(outputs, self.item_emb.weight.transpose(0, 1))

        return outputs
    

class SASRecwoAttn(nn.Module):
    """Adaptation of code from
    https://github.com/pmixer/SASRec.pytorch.
    """

    def __init__(self, item_num, maxlen=128, hidden_units=64, num_blocks=1,
                 num_heads=1, dropout_rate=0.1, initializer_range=0.02,
                 add_head=True, padding_idx=0):

        super(SASRecwoAttn, self).__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.padding_idx=padding_idx

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=self.padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        # self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            # new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            # self.attention_layernorms.append(new_attn_layernorm)

            # new_attn_layer = nn.MultiheadAttention(hidden_units,
            #                                        num_heads,
            #                                        dropout_rate)
            # self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # parameter attention mask added for compatibility with GPT Lightning module, not used
    def forward(self, input_ids, attention_mask):

        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        # need to be on the same device
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.Tensor(input_ids == self.padding_idx)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        # tl = seqs.shape[1] # time dim len for enforce causality
        # need to be on the same device
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(seqs.device))

        for i in range(len(self.forward_layers)):
            # seqs = torch.transpose(seqs, 0, 1)
            # Q = self.attention_layernorms[i](seqs)
            # mha_outputs, _ = self.attention_layers[i](Q, Q, Q, 
            #                                 attn_mask=attention_mask)
            #                                 # key_padding_mask=timeline_mask
            #                                 # need_weights=False) this arg do not work?
            # seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        outputs = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        if self.add_head:
            outputs = torch.matmul(outputs, self.item_emb.weight.transpose(0, 1))

        return outputs


# ---------------------------------------------------------------------------
# Shared helper for BSARec and WEARec
# ---------------------------------------------------------------------------

class DenseFeedForward(nn.Module):
    """Linear-based feed-forward block with GELU activation and residual."""

    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.dense1 = nn.Linear(hidden_units, 4 * hidden_units)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, x, residual_scale: float = 1.0):
        h = self.dropout(self.dense2(self.act(self.dense1(x))))
        return self.layernorm(residual_scale * x + h)


# ---------------------------------------------------------------------------
# BSARec
# Blends frequency-domain (FFT low/high-pass) and attention-domain features.
# Reference: WEARec/src/model/bsarec.py
# ---------------------------------------------------------------------------

class BSARecFrequencyLayer(nn.Module):
    """FFT-based low/high-pass filter with learnable blending parameter."""

    def __init__(self, hidden_units, dropout_rate, c):
        super().__init__()
        self.c = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_units))
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, x):
        batch, seq_len, hidden = x.shape
        X = torch.fft.rfft(x, dim=1, norm='ortho')
        # low-pass: keep only the first `c` frequency bins
        low_pass = X.clone()
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = x - low_pass
        out = low_pass + (self.sqrt_beta ** 2) * high_pass
        out = self.dropout(out)
        return self.layernorm(out + x)


class BSARecMHA(nn.Module):
    """Multi-head self-attention with additive causal mask, residual, LayerNorm."""

    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        assert hidden_units % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(hidden_units, hidden_units)
        self.k = nn.Linear(hidden_units, hidden_units)
        self.v = nn.Linear(hidden_units, hidden_units)
        self.out_proj = nn.Linear(hidden_units, hidden_units)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def _split_heads(self, x):
        # (B, T, D) -> (B, H, T, head_dim)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x, attention_mask, residual_scale: float = 1.0):
        # attention_mask: (B, 1, T, T) additive, -10000.0 for positions to ignore
        Q = self._split_heads(self.q(x))
        K = self._split_heads(self.k(x))
        V = self._split_heads(self.v(x))

        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V)  # (B, H, T, head_dim)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(x.shape)
        out = self.out_dropout(self.out_proj(ctx))
        return self.layernorm(residual_scale * x + out)


class BSARecLayer(nn.Module):
    """Blends DSP (frequency) and GSP (attention) features with fixed alpha."""

    def __init__(self, hidden_units, num_heads, dropout_rate, c, alpha):
        super().__init__()
        self.freq = BSARecFrequencyLayer(hidden_units, dropout_rate, c)
        self.attn = BSARecMHA(hidden_units, num_heads, dropout_rate)
        self.alpha = alpha

    def forward(self, x, attention_mask):
        dsp = self.freq(x)
        gsp = self.attn(x, attention_mask)
        return self.alpha * dsp + (1.0 - self.alpha) * gsp


class BSARecBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, c, alpha):
        super().__init__()
        self.layer = BSARecLayer(hidden_units, num_heads, dropout_rate, c, alpha)
        self.ffn = DenseFeedForward(hidden_units, dropout_rate)

    def forward(self, x, attention_mask):
        return self.ffn(self.layer(x, attention_mask))


class BSARec(nn.Module):
    """BSARec: frequency + attention hybrid sequential recommender.

    Reference: https://arxiv.org/abs/2312.13613 (ported from WEARec/src/model/bsarec.py)
    """

    def __init__(self, item_num, maxlen=128, hidden_units=64, num_blocks=1,
                 num_heads=1, dropout_rate=0.1, initializer_range=0.02,
                 add_head=True, padding_idx=0, alpha=0.5, c=4):
        super().__init__()
        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.padding_idx = padding_idx

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        import copy
        block = BSARecBlock(hidden_units, num_heads, dropout_rate, c, alpha)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_blocks)])
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == self.padding_idx)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # Build additive causal mask: (1, 1, T, T), -10000 for upper triangle
        tl = seqs.shape[1]
        causal_mask = torch.tril(torch.ones((tl, tl), dtype=seqs.dtype, device=seqs.device))
        causal_mask = (1.0 - causal_mask) * -10000.0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        for block in self.blocks:
            seqs = block(seqs, causal_mask)
            seqs *= ~timeline_mask.unsqueeze(-1)

        outputs = self.last_layernorm(seqs)
        if self.add_head:
            outputs = torch.matmul(outputs, self.item_emb.weight.transpose(0, 1))
        return outputs


# ---------------------------------------------------------------------------
# WEARec
# Blends FFT global features (with adaptive MLP modulation) and
# Haar wavelet local features.
# Reference: WEARec/src/model/wearec.py
# ---------------------------------------------------------------------------

class WEARecLayer(nn.Module):
    """Hybrid FFT + Haar-wavelet layer with adaptive frequency modulation."""

    def __init__(self, hidden_units, maxlen, num_heads, dropout_rate, alpha):
        super().__init__()
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.seq_len = maxlen
        self.freq_bins = maxlen // 2 + 1
        self.alpha = alpha

        # Wavelet: learnable scale on detail coefficients
        # shape (1, num_heads, maxlen//2, head_dim)
        self.complex_weight = nn.Parameter(
            torch.randn(1, num_heads, maxlen // 2, self.head_dim) * 0.02
        )

        # FFT base filter and bias (per head × freq bin)
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))
        self.base_bias = nn.Parameter(torch.full((num_heads, self.freq_bins, 1), -0.1))

        # Adaptive MLP: maps global context → per-(head, freq_bin) scale & bias
        self.adaptive_mlp = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, num_heads * self.freq_bins * 2),
        )

        self.out_dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def wavelet_transform(self, x_heads):
        """Single-level Haar wavelet decomposition + reconstruction.

        Args:
            x_heads: (B, num_heads, seq_len, head_dim)
        Returns:
            Reconstructed tensor of the same shape.
        """
        B, H, N, D = x_heads.shape
        N_even = N if (N % 2 == 0) else (N - 1)
        x = x_heads[:, :, :N_even, :]

        x_even = x[:, :, 0::2, :]
        x_odd = x[:, :, 1::2, :]

        approx = 0.5 * (x_even + x_odd)
        detail = 0.5 * (x_even - x_odd)

        # Scale detail with learnable weights (clamp to valid length in case of mismatch)
        detail_len = detail.shape[2]
        w = self.complex_weight[:, :, :detail_len, :]
        detail = detail * w

        x_even_recon = approx + detail
        x_odd_recon = approx - detail

        out = torch.zeros_like(x_heads)
        # Use explicit N_even-bounded slices to avoid mismatch when N is odd
        out[:, :, 0:N_even:2, :] = x_even_recon
        out[:, :, 1:N_even:2, :] = x_odd_recon
        # Position N_even (when N is odd) stays 0

        return out

    def forward(self, x):
        batch, seq_len, hidden = x.shape
        # Actual freq bins may be < self.freq_bins when seq_len < maxlen (short batches)
        actual_freq_bins = seq_len // 2 + 1

        # Reshape to multi-head: (B, num_heads, seq_len, head_dim)
        x_heads = x.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # --- (1) FFT global features ---
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')  # (B, H, actual_freq_bins, head_dim)

        context = x.mean(dim=1)  # (B, hidden)
        adapt_params = self.adaptive_mlp(context)  # (B, num_heads*self.freq_bins*2)
        adapt_params = adapt_params.view(batch, self.num_heads, self.freq_bins, 2)
        # Slice to actual freq bins in case seq_len < maxlen
        adapt_params = adapt_params[:, :, :actual_freq_bins, :]
        adaptive_scale = adapt_params[..., 0:1]  # (B, num_heads, actual_freq_bins, 1)
        adaptive_bias = adapt_params[..., 1:2]

        effective_filter = self.base_filter[:, :actual_freq_bins, :] * (1.0 + adaptive_scale)
        effective_bias = self.base_bias[:, :actual_freq_bins, :] + adaptive_bias

        F_fft_mod = F_fft * effective_filter + effective_bias
        x_fft = torch.fft.irfft(F_fft_mod, dim=2, n=seq_len, norm='ortho')  # (B, H, seq_len, head_dim)

        # --- (2) Wavelet local features ---
        x_wavelet = self.wavelet_transform(x_heads)

        # --- (3) Alpha blend ---
        x_combined = (1.0 - self.alpha) * x_wavelet + self.alpha * x_fft

        # Reshape back: (B, seq_len, hidden)
        x_out = x_combined.permute(0, 2, 1, 3).reshape(batch, seq_len, hidden)

        out = self.out_dropout(x_out)
        return self.layernorm(out + x)


class WEARecBlock(nn.Module):
    def __init__(self, hidden_units, maxlen, num_heads, dropout_rate, alpha):
        super().__init__()
        self.layer = WEARecLayer(hidden_units, maxlen, num_heads, dropout_rate, alpha)
        self.ffn = DenseFeedForward(hidden_units, dropout_rate)

    def forward(self, x):
        return self.ffn(self.layer(x))


class WEARec(nn.Module):
    """WEARec: wavelet + FFT (adaptive) hybrid sequential recommender.

    Reference: WEARec/src/model/wearec.py
    """

    def __init__(self, item_num, maxlen=128, hidden_units=64, num_blocks=1,
                 num_heads=4, dropout_rate=0.1, initializer_range=0.02,
                 add_head=True, padding_idx=0, alpha=0.5):
        super().__init__()
        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.padding_idx = padding_idx

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        import copy
        block = WEARecBlock(hidden_units, maxlen, num_heads, dropout_rate, alpha)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_blocks)])
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == self.padding_idx)
        seqs *= ~timeline_mask.unsqueeze(-1)

        for block in self.blocks:
            seqs = block(seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        outputs = self.last_layernorm(seqs)
        if self.add_head:
            outputs = torch.matmul(outputs, self.item_emb.weight.transpose(0, 1))
        return outputs


# ---------------------------------------------------------------------------
# FreqRec
# Frequency-domain MLP (temporal + channel) blended with self-attention.
# Optionally adds a Fourier reconstruction auxiliary loss.
# Reference: FreqRec/src/model/FreqRec.py
# ---------------------------------------------------------------------------

class FreqRecFreMLP(nn.Module):
    """Complex linear transform in frequency domain with ReLU + softshrink sparsity."""

    def __init__(self, hidden_units, sparsity_threshold=0.02, scale=0.02):
        super().__init__()
        self.r = nn.Parameter(scale * torch.randn(hidden_units, hidden_units))
        self.i = nn.Parameter(scale * torch.randn(hidden_units, hidden_units))
        self.rb = nn.Parameter(scale * torch.randn(hidden_units))
        self.ib = nn.Parameter(scale * torch.randn(hidden_units))
        self.sparsity_threshold = sparsity_threshold

    def forward(self, x):
        # x: complex tensor of shape (..., hidden_units)
        o_real = F.relu((x.real @ self.r) - (x.imag @ self.i) + self.rb)
        o_imag = F.relu((x.imag @ self.r) + (x.real @ self.i) + self.ib)
        y = torch.stack([o_real, o_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        return torch.view_as_complex(y)


class FreqRecFilterModel(nn.Module):
    """Core filter: parallel or cascade frequency-domain MLP (temporal + channel)."""

    def __init__(self, hidden_units, dropout_rate, gama=0.5, chux='p',
                 sparsity_threshold=0.02, scale=0.02):
        super().__init__()
        self.gama = gama
        self.chux = chux
        self.hidden_units = hidden_units
        # Unused but kept for parameter parity with original
        self.embeddings = nn.Parameter(torch.randn(1, hidden_units))
        # Channel MLP params (FFT along batch dim)
        self.fre_channel = FreqRecFreMLP(hidden_units, sparsity_threshold, scale)
        # Temporal MLP params (FFT along sequence dim)
        self.fre_temporal = FreqRecFreMLP(hidden_units, sparsity_threshold, scale)
        self.out_dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def _mlp_temporal(self, x):
        # x: (B, S, H)
        B, S, H = x.shape
        X = torch.fft.rfft(x, dim=1, norm='ortho')  # (B, S//2+1, H) complex
        Y = self.fre_temporal(X)
        return torch.fft.irfft(Y, n=S, dim=1, norm='ortho')  # (B, S, H)

    def _mlp_channel(self, x):
        # x: (B, S, H) — FFT along B dimension (cross-batch channel mixing)
        B, S, H = x.shape
        x_perm = x.permute(1, 0, 2)  # (S, B, H)
        X = torch.fft.rfft(x_perm, dim=1, norm='ortho')  # (S, B//2+1, H) complex
        Y = self.fre_channel(X)
        out = torch.fft.irfft(Y, n=B, dim=1, norm='ortho')  # (S, B, H)
        return out.permute(1, 0, 2)  # (B, S, H)

    def forward(self, x, residual_scale: float = 1.0):
        B, S, H = x.shape
        bias = x
        if self.chux == 'p':
            x_channel = self._mlp_channel(x)
            x_temporal = self._mlp_temporal(x)
            out = (1.0 - self.gama) * x_channel + self.gama * x_temporal
        else:  # 'c': cascade
            x_channel = self._mlp_channel(x)
            x = bias + x_channel
            out = self._mlp_temporal(x)
        out = self.out_dropout(out)
        return self.layernorm(out + residual_scale * bias)


class FreqRecFilterLayer(nn.Module):
    """Blends frequency-domain filter and self-attention with alpha."""

    def __init__(self, hidden_units, num_heads, dropout_rate, gama, chux, alpha):
        super().__init__()
        self.filter = FreqRecFilterModel(hidden_units, dropout_rate, gama=gama, chux=chux)
        self.attn = BSARecMHA(hidden_units, num_heads, dropout_rate)
        self.alpha = alpha

    def forward(self, x, attention_mask, residual_scale: float = 1.0):
        filt = self.filter(x, residual_scale=residual_scale)
        att = self.attn(x, attention_mask, residual_scale=residual_scale)
        return self.alpha * filt + (1.0 - self.alpha) * att


class FreqRecFilterBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, gama, chux, alpha):
        super().__init__()
        self.layer = FreqRecFilterLayer(hidden_units, num_heads, dropout_rate, gama, chux, alpha)
        self.ffn = DenseFeedForward(hidden_units, dropout_rate)

    def forward(self, x, attention_mask, residual_scale: float = 1.0):
        return self.ffn(self.layer(x, attention_mask, residual_scale=residual_scale),
                        residual_scale=residual_scale)


class FreqRec(nn.Module):
    """FreqRec: frequency-domain MLP + attention hybrid sequential recommender.

    Reference: FreqRec/src/model/FreqRec.py

    When fourier_loss=True, forward() returns (logits, aux_loss) during training.
    The SeqRec lightning module handles this tuple automatically.
    """

    def __init__(self, item_num, maxlen=128, hidden_units=64, num_blocks=2,
                 num_heads=2, dropout_rate=0.1, initializer_range=0.02,
                 add_head=True, padding_idx=0,
                 alpha=0.5, gama=0.5, chux='p',
                 fourier_loss=False, fft_loss_type='l2', alpha_loss=0.5,
                 residual_scale_eval=1.0):
        super().__init__()
        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.padding_idx = padding_idx
        self.fourier_loss = fourier_loss
        self.fft_loss_type = fft_loss_type
        self.alpha_loss = alpha_loss
        self.residual_scale_eval = residual_scale_eval

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        import copy
        block = FreqRecFilterBlock(hidden_units, num_heads, dropout_rate, gama, chux, alpha)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_blocks)])
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _fft_loss_fn(self, a, b):
        if self.fft_loss_type == 'l1':
            return F.l1_loss(a, b)
        elif self.fft_loss_type == 'l2':
            return F.mse_loss(a, b)
        elif self.fft_loss_type == 'SmoothL1Loss':
            return F.smooth_l1_loss(a, b)
        else:  # mix_loss
            return 0.5 * F.l1_loss(a, b) + 0.5 * F.mse_loss(a, b)

    def _compute_fft_loss(self, seq_output, target_emb):
        # FFT along the sequence dim (after transposing to (B, H, S))
        fft1 = torch.fft.fft(seq_output.transpose(1, 2), norm='forward').transpose(1, 2)
        fft2 = torch.fft.fft(target_emb.transpose(1, 2), norm='forward').transpose(1, 2)
        return self._fft_loss_fn(fft1.real, fft2.real) + self._fft_loss_fn(fft1.imag, fft2.imag)

    def forward(self, input_ids, attention_mask, residual_scale: float = 1.0):
        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == self.padding_idx)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # Preserve embedding for Fourier reconstruction loss
        sequence_emb = seqs

        # Causal attention mask: (1, 1, T, T) additive
        tl = seqs.shape[1]
        causal_mask = torch.tril(torch.ones((tl, tl), dtype=seqs.dtype, device=seqs.device))
        causal_mask = (1.0 - causal_mask) * -10000.0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            seqs = block(seqs, causal_mask, residual_scale=residual_scale)
            seqs *= ~timeline_mask.unsqueeze(-1)

        seq_output = self.last_layernorm(seqs)

        if self.add_head:
            outputs = torch.matmul(seq_output, self.item_emb.weight.transpose(0, 1))
        else:
            outputs = seq_output

        if self.fourier_loss and self.training:
            # Return 3-tuple (logits, fft_loss, alpha_loss) so modules.py can compute:
            # total_loss = alpha_loss * CE + (1 - alpha_loss) * fft_loss
            fft_loss = self._compute_fft_loss(seq_output, sequence_emb)
            return outputs, fft_loss, self.alpha_loss

        return outputs


class UniformBertSelfAttention(nn.Module):
    """
    Drop-in replacement for BertSelfAttention that uses uniform attention
    over unmasked positions.
    """
    def __init__(self, base_self_attn: nn.Module):
        super().__init__()
        self.num_attention_heads = base_self_attn.num_attention_heads
        self.attention_head_size = base_self_attn.attention_head_size
        self.all_head_size = base_self_attn.all_head_size
        self.query = base_self_attn.query
        self.key = base_self_attn.key
        self.value = base_self_attn.value
        self.dropout = base_self_attn.dropout
        self.is_decoder = getattr(base_self_attn, "is_decoder", False)
        self.position_embedding_type = getattr(base_self_attn, "position_embedding_type", "absolute")

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs,
    ):
        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        batch_size, num_heads, query_len, _ = query_layer.size()
        key_len = key_layer.size(2)

        attention_scores = torch.zeros(
            (batch_size, num_heads, query_len, key_len),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                mask_val = torch.finfo(attention_scores.dtype).min
                attention_scores = attention_scores.masked_fill(~attention_mask, mask_val)
            else:
                attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + ((key_layer, value_layer),)
        return outputs


class BERT4Rec(nn.Module):

    def __init__(self, vocab_size, bert_config, add_head=True,
                 tie_weights=True, padding_idx=0, init_std=0.02,
                 save_norms=False, analysis_dir=None, residual_scale=1.0,
                 uniform_attention=False):

        super().__init__()

        self.vocab_size = vocab_size
        self.bert_config = bert_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std
        self.save_norms = save_norms
        self.analysis_dir = analysis_dir
        self.residual_scale = residual_scale  # Added by Author
        self.uniform_attention = uniform_attention

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=bert_config['hidden_size'],
                                        padding_idx=padding_idx)
        self.transformer_model = BertModel(BertConfig(**bert_config))
        if self.uniform_attention:
            self._enable_uniform_attention()

        if self.add_head:
            self.head = nn.Linear(bert_config['hidden_size'], vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight

        self.init_weights()

    def init_weights(self):

        # initialization in huggingface transformers
        # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L462
        # initialization in pytorch Embeddings
        # https://github.com/pytorch/pytorch/blob/1.7/torch/nn/modules/sparse.py#L117

        self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.embed_layer.weight.data[self.padding_idx].zero_()

    def _enable_uniform_attention(self):
        encoder = getattr(self.transformer_model, "encoder", None)
        if encoder is None or not hasattr(encoder, "layer"):
            raise RuntimeError("BERT model encoder layers not found; cannot enable uniform attention.")
        for layer in encoder.layer:
            layer.attention.self = UniformBertSelfAttention(layer.attention.self)

    def forward(
        self,
        input_ids,
        attention_mask,
        output_norms=False,
        return_norms=False,
        save_analysis=False,
        analysis_batch_idx=0,
        residual_scale=1.0,
    ):

        embeds = self.embed_layer(input_ids)
        want_norms = output_norms or return_norms or save_analysis
        if want_norms:
            transformer_outputs = self.transformer_model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_norms=True,
                residual_scale=residual_scale,  # Added by Author
            )
        else:
            transformer_outputs = self.transformer_model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                residual_scale=residual_scale,  # Added by Author
            )
        # if os.environ.get("BERT4REC_DEBUG_RESIDUAL_SCALE") == "1":
        #     print(f"[BERT4Rec] residual_scale={residual_scale}")
        # norm-analysis transformers return a tuple; keep compatibility
        outputs = (
            transformer_outputs[0]
            if isinstance(transformer_outputs, tuple)
            else transformer_outputs.last_hidden_state
        )

        if self.add_head:
            outputs = self.head(outputs)

        if not want_norms:
            return outputs

        norms = transformer_outputs[-1] if isinstance(transformer_outputs, tuple) else None
        if norms is None:
            if return_norms:
                return outputs, {"layer": []}
            return outputs
        if os.environ.get("BERT4REC_DEBUG_NORMS") == "1":
            try:
                first = norms[0]
                shapes = [tuple(x.shape) for x in first]
                print(f"[BERT4Rec] norms layers={len(norms)} shapes={shapes}")
            except Exception as exc:
                print(f"[BERT4Rec] norms debug failed: {exc}")

        layer_stats = []
        for layer_norms in norms:
            (
                weighted_norm,
                summed_weighted_norm,
                residual_weighted_norm,
                post_ln_norm,
                attn_mixing_ratio,
                attnres_mixing_ratio,
                attnresln_mixing_ratio,
            ) = layer_norms
            layer_stats.append(
                {
                    "weighted_norm": weighted_norm,
                    "summed_weighted_norm": summed_weighted_norm,
                    "residual_weighted_norm": residual_weighted_norm,
                    "post_ln_norm": post_ln_norm,
                    "attn_mixing_ratio": attn_mixing_ratio,
                    "attnres_mixing_ratio": attnres_mixing_ratio,
                    "mixing_ratio": attnresln_mixing_ratio,
                }
            )

        analysis = {"layer": layer_stats}
        if save_analysis and self.analysis_dir is not None:
            save_analysis_batch_npz(
                analysis=analysis,
                input_ids=input_ids,
                analysis_dir=self.analysis_dir,
                batch_idx=analysis_batch_idx,
                npz_keys=getattr(self, "npz_keys", None),
            )

        if return_norms:
            return outputs, analysis
        return outputs
    

class SFSRec(nn.Module):
    def __init__(
        self,
        item_num: int,
        maxlen: int = 128,
        hidden_units: int = 64,
        num_blocks: int = 1,
        dropout_rate: float = 0.1,
        add_head: bool = True,
        padding_idx: int = 0,
        use_causal_mask: bool = True,
        analysis_dir: str = "./analysis_out",
    ):
        super().__init__()

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        self.sfs_layers = nn.ModuleList([
            AnalyzableUniformAttention(hidden_units, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        self.attn_post_lns = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8)
            for _ in range(num_blocks)
        ])

        self.ffn_layers = nn.ModuleList([
            PointWiseFFNNoResidual(hidden_units, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.ffn_post_lns = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8)
            for _ in range(num_blocks)
        ])

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        self.add_head = add_head
        self.padding_idx = padding_idx
        self.use_causal_mask = use_causal_mask
        self.analysis_dir = analysis_dir
        self.initializer_range = 0.02

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,             # optional [B, L]
        return_mixing: bool = False,
        save_analysis: bool = False,
        analysis_batch_idx: int = 0,
    ):
        if isinstance(return_mixing, torch.Tensor):
            return_mixing = False
        B, L = input_ids.shape
        device = input_ids.device

        # ----- Embedding -----
        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        seqs = seqs + self.pos_emb(pos)
        seqs = self.emb_dropout(seqs)
        seqs = self.emb_layernorm(seqs)

        if attention_mask is None:
            attention_mask = (input_ids != self.padding_idx)
        timeline_mask = (input_ids == self.padding_idx)
        seqs = seqs * (~timeline_mask).unsqueeze(-1)

        extended_attention_mask = None
        if self.use_causal_mask:
            attn = attention_mask.to(dtype=seqs.dtype)
            ext = attn.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            max_len = attn.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(
                torch.ones(attn_shape, device=seqs.device), diagonal=1
            )
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            ext = ext * subsequent_mask
            extended_attention_mask = (1.0 - ext) * -10000.0

        layer_stats = []

        # ----- Blocks -----
        for i in range(len(self.sfs_layers)):
            pre_ln, mixing = self.sfs_layers[i](
                x=seqs,
                layer_norm=self.attn_post_lns[i],
                output_mixing=return_mixing,
                apply_causal_mask=self.use_causal_mask,
                attention_mask=attention_mask,
            )
            seqs = self.attn_post_lns[i](pre_ln)
            seqs = seqs * (~timeline_mask).unsqueeze(-1)

            if return_mixing:
                layer_stats.append(mixing)

            ffn_residual = seqs
            ffn_delta = self.ffn_layers[i](seqs)
            seqs = self.ffn_post_lns[i](ffn_residual + ffn_delta)
            seqs = seqs * (~timeline_mask).unsqueeze(-1)

        outputs = seqs

        if self.add_head:
            logits = torch.matmul(outputs, self.item_emb.weight.t())
        else:
            logits = outputs

        if not return_mixing:
            return logits

        analysis = {"layer": layer_stats}

        if save_analysis:
            save_analysis_batch_npz(
                analysis=analysis,
                input_ids=input_ids,
                analysis_dir=self.analysis_dir,
                batch_idx=analysis_batch_idx,
                npz_keys=getattr(self, "npz_keys", None),
            )

        return logits, analysis


class AnalyzableUniformAttention(nn.Module):
    """
    SASRec-compatible attention replacement:
    - attention weights = causal uniform average
    - value/out projection identical to SASRec
    """
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.value_proj = nn.Linear(hidden_units, hidden_units, bias=False)
        self.out_proj   = nn.Linear(hidden_units, hidden_units, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,              # [B,L,H] (LN-ed input)
        layer_norm: nn.LayerNorm,
        output_mixing: bool = False,
        apply_causal_mask: bool = True,
        attention_mask: Optional[torch.Tensor] = None,  # [B,L]
    ):
        B, L, H = x.shape
        device = x.device

        # ---- uniform T with optional causal + valid-token re-normalization ----
        base = torch.ones((L, L), device=device, dtype=x.dtype)
        if apply_causal_mask:
            base = torch.tril(base)

        if attention_mask is None:
            allowed = base.unsqueeze(0).expand(B, -1, -1)  # [B,L,L]
        else:
            key_valid = attention_mask.to(device=device)
            if key_valid.dtype != torch.bool:
                key_valid = key_valid > 0
            key_valid = key_valid.to(dtype=x.dtype)
            allowed = base.unsqueeze(0) * key_valid.unsqueeze(1)  # [B,L,L]

        denom = allowed.sum(dim=-1, keepdim=True).clamp_min(1.0)
        T = allowed / denom  # [B,L,L]

        # ---- value transform (same role as SASRec) ----
        v = self.out_proj(self.value_proj(x))   # [B,L,H]

        # ---- mixing ----
        mixed = torch.einsum("bij,bjh->bih", T, v)
        mixed = self.dropout(mixed)

        # ---- residual ----
        pre_ln = x + mixed

        analysis = None
        if output_mixing:
            # G[t,j] = T[t,j] * f(x_j)
            G = T.unsqueeze(-1) * v.unsqueeze(1)

            preserving = torch.diagonal(G, dim1=1, dim2=2).permute(0, 2, 1)
            mixing = G.sum(dim=2) - preserving

            p_norm = torch.norm(preserving, dim=-1)
            m_norm = torch.norm(mixing, dim=-1)

            post_ln = layer_norm(pre_ln)

            analysis = {
                "mixing_ratio": m_norm / (m_norm + p_norm + 1e-12),
                "post_ln_norm": torch.norm(post_ln, dim=-1),
            }

        return pre_ln, analysis


def compute_fft_transfer_matrix(weight: torch.Tensor, L: int, device):
    """
    weight: complex tensor [1, L//2+1, H]
    return: T [L, L]
    """
    T = torch.zeros(L, L, device=device)

    for j in range(L):
        x = torch.zeros(1, L, 1, device=device)
        x[0, j, 0] = 1.0

        X = torch.fft.rfft(x, dim=1, norm="ortho")
        X = X * weight[:, :X.size(1), :1]
        y = torch.fft.irfft(X, n=L, dim=1, norm="ortho")

        T[:, j] = y[0, :, 0]

    return T

def compute_position_contribution(x: torch.Tensor, T: torch.Tensor):
    """
    x: [B, L, H]
    T: [L, L] or [B, L, L]
    return: G [B, L, L, H]
    """
    if T.dim() == 2:
        return T.view(1, x.size(1), x.size(1), 1) * x.unsqueeze(1)
    return T.unsqueeze(-1) * x.unsqueeze(1)


def compute_mixing_ratio_from_G(G: torch.Tensor):
    """
    G: [B, L, L, H]
    """
    preserving = torch.diagonal(G, dim1=1, dim2=2).permute(0, 2, 1)
    mixing = G.sum(dim=2) - preserving

    p_norm = torch.norm(preserving, dim=-1)
    m_norm = torch.norm(mixing, dim=-1)

    return m_norm / (m_norm + p_norm + 1e-12)

    



#added by me
class LightSASRec(nn.Module):
    """Adaptation of code from
    https://github.com/pmixer/SASRec.pytorch.
    """

    def __init__(self, item_num, maxlen=128, hidden_units=64, num_blocks=1,
                 num_heads=1, dropout_rate=0.1, initializer_range=0.02,
                 add_head=True, padding_idx=0):

        super(LightSASRec, self).__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.padding_idx=padding_idx

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=self.padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        # self.forward_layernorms = nn.ModuleList()
        # self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(hidden_units,
                                                   num_heads,
                                                   dropout_rate)
            self.attention_layers.append(new_attn_layer)

            # new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            # self.forward_layernorms.append(new_fwd_layernorm)

            # new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            # self.forward_layers.append(new_fwd_layer)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # parameter attention mask added for compatibility with GPT Lightning module, not used
    def forward(self, input_ids, attention_mask):

        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        # need to be on the same device
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.Tensor(input_ids == self.padding_idx)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        # need to be on the same device
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            #! changed to Q,Q,Q
            mha_outputs, _ = self.attention_layers[i](Q, Q, Q, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            # seqs = self.forward_layernorms[i](seqs)
            # seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        outputs = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        if self.add_head:
            outputs = torch.matmul(outputs, self.item_emb.weight.transpose(0, 1))

        return outputs
    


# light_sasrec_analyze.py
# -*- coding: utf-8 -*-
"""
LightSASRecAnalyze: an extension that preserves the same forward structure as
LightSASRec while enabling Kobayashi (2020) mixing analysis for each block's
Self-Attention.

Important:
- "independent LN per block" = keep attention_layernorms[i] as in LightSASRec
- AnalyzableMHA carries no LN at all (clear separation of responsibilities)
- forward order matches LightSASRec:
    Embedding + Pos + Dropout
    mask (pad)
    for each block:
        transpose
        Q = LN_i(seqs)
        mha(Q,Q,Q)
        seqs = Q + mha_out
        transpose back
        mask (pad)
    last_layernorm
    head (matmul) optional
"""

import os
import math
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def save_analysis_batch_npz(
    analysis: Dict,
    input_ids: torch.Tensor,
    analysis_dir: str,
    batch_idx: int,
    npz_keys: Optional[List[str]] = None,
):
    """Save analysis data to npz.

    Args:
        npz_keys: List of key suffixes to save (e.g. ["mixing_ratio"]).
                  None means save all keys including input_ids.
    """
    os.makedirs(analysis_dir, exist_ok=True)

    save_dict = {}
    if npz_keys is None or "input_ids" in npz_keys:
        save_dict["input_ids"] = input_ids.detach().cpu().numpy()

    for layer_idx, layer_dict in enumerate(analysis["layer"]):
        for k, v in layer_dict.items():
            if npz_keys is None or k in npz_keys:
                save_dict[f"layer{layer_idx}_{k}"] = (
                    v.detach().cpu().numpy()
                )

    path = os.path.join(analysis_dir, f"batch{batch_idx:06d}.npz")
    np.savez(path, **save_dict)

    print(f"[Saved] {path}")

class NormMixingOutput(nn.Module):
    """
    Analysis module that applies the Kobayashi (2020) norm-based decomposition
    to MultiheadAttention in SASRec / LightSASRec.

    Supported mixing ratios:
      - Attn-N        : attention only
      - AttnRes-N     : attention + residual (pre-LN)
      - AttnResLN-N   : attention + residual + LayerNorm (post-LN)
    """
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(
        self,
        hidden_states,      # [B, L, H] = Q (after LN)
        attention_probs,   # [B, Hh, L, L]
        value_layer,       # [B, Hh, L, Dh]
        out_proj,          # nn.Linear(H, H)
        layer_norm,        # nn.LayerNorm
        pre_ln_states,     # [B, L, H] = z_i = Q + Attn(Q)
        residual_scale: float = 1.0,
    ):
        B, L, H = hidden_states.shape
        Hh = self.num_heads
        Dh = self.head_dim

        # ==================================================
        # 1. f_h(x_j) = V_h(x_j) W_o
        # ==================================================
        Wo = out_proj.weight.view(H, Hh, Dh).permute(1, 2, 0)  # [Hh, Dh, H]

        transformed_layer = torch.einsum(
            "bhjd,hdv->bhjv", value_layer, Wo
        )  # [B, Hh, L, H]

        # ==================================================
        # 2. Attention weighted sum: α_ij f(x_j)
        # ==================================================
        weighted = torch.einsum(
            "bhij,bhjd->bhijd", attention_probs, transformed_layer
        )  # [B, Hh, L, L, H]

        summed_weighted = weighted.sum(dim=1)  # [B, L, L, H]

        # ==================================================
        # 3. Residual
        # ==================================================
        eye = torch.eye(L, device=hidden_states.device)
        residual = residual_scale * torch.einsum(
            "ij,bjd->bijd", eye, hidden_states
        )  # [B, L, L, H]
        # residual = residual_scale * hidden_states[:, :, None, :]

        z_parts = summed_weighted + residual  # [B, L, L, H]

        # ==================================================
        # 4. LayerNorm (Kobayashi-style Jacobian decomposition)
        # ==================================================
        mean = pre_ln_states.mean(-1, keepdim=True)           # [B, L, 1]
        var = (pre_ln_states - mean).pow(2).mean(-1, keepdim=True)
        sigma = torch.sqrt(var + layer_norm.eps)              # [B, L, 1]

        each_mean = z_parts.mean(-1, keepdim=True)            # [B, L, L, 1]
        normalized = (z_parts - each_mean) / sigma.unsqueeze(2)

        gamma = layer_norm.weight
        post_ln = torch.einsum("bijd,d->bijd", normalized, gamma)

        # ==================================================
        # 5. Mixing ratio (3 variants)
        # ==================================================

        # ---------- Attn-N ----------
        attn_preserving = torch.diagonal(
            summed_weighted, dim1=1, dim2=2
        ).permute(0, 2, 1)  # [B, L, H]

        attn_mixing = summed_weighted.sum(dim=2) - attn_preserving

        attn_preserving_norm = torch.norm(attn_preserving, dim=-1)
        attn_mixing_norm = torch.norm(attn_mixing, dim=-1)

        attn_mixing_ratio = attn_mixing_norm / (
            attn_mixing_norm + attn_preserving_norm + 1e-12
        )

        # ---------- AttnRes-N (before LN) ----------
        before_preserving = torch.diagonal(
            z_parts, dim1=1, dim2=2
        ).permute(0, 2, 1)

        before_mixing = z_parts.sum(dim=2) - before_preserving

        before_preserving_norm = torch.norm(before_preserving, dim=-1)
        before_mixing_norm = torch.norm(before_mixing, dim=-1)

        attnres_mixing_ratio = before_mixing_norm / (
            before_mixing_norm + before_preserving_norm + 1e-12
        )

        # ---------- AttnResLN-N (after LN) ----------
        post_preserving = torch.diagonal(
            post_ln, dim1=1, dim2=2
        ).permute(0, 2, 1)

        post_mixing = post_ln.sum(dim=2) - post_preserving

        post_preserving_norm = torch.norm(post_preserving, dim=-1)
        post_mixing_norm = torch.norm(post_mixing, dim=-1)

        mixing_ratio = post_mixing_norm / (
            post_mixing_norm + post_preserving_norm + 1e-12
        )

        # ==================================================
        # 6. Row-wise normalized entropy
        # ==================================================
        eps = 1e-12
        device = attention_probs.device

        # entropy per head & position
        entropy = -(attention_probs * torch.log(attention_probs + eps)).sum(dim=-1)
        # [B, Hh, L]

        # max entropy for causal mask
        positions = torch.arange(1, L + 1, device=device).float()
        max_entropy = torch.log(positions).view(1, 1, L)

        normalized_entropy = entropy / (max_entropy + eps)
        # average over heads and positions
        entropy_mean = normalized_entropy.mean(dim=(1, 2))
        # [B]

        # ==================================================
        # return
        # ==================================================
        return {
            "post_ln_norm": torch.norm(post_ln, dim=-1),   # ||AttnResLN||
            "attn_mixing_ratio": attn_mixing_ratio,        # Attn-N
            "attnres_mixing_ratio": attnres_mixing_ratio,  # AttnRes-N
            "mixing_ratio": mixing_ratio,                  # AttnResLN-N
            "normalized_attention_entropy": entropy_mean,
        }



# ======================================================
# Analyzable MHA (NO LayerNorm inside!)
# ======================================================
class AnalyzableMHA(nn.Module):
    """
    Use nn.MultiheadAttention as-is, extract attention_weights and value for
    mixing analysis.

    Notes:
    - The input Q is expected to already pass through the block LN.
    - forward returns out = Q + MHA(Q,Q,Q), matching LightSASRec.
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # match LightSASRec
        )
        self.out_dropout = nn.Dropout(dropout)

        self.mixing_analyzer = NormMixingOutput(hidden_size=hidden_size, num_heads=num_heads)

    def _extract_value_layer(
        self,
        Q: torch.Tensor,  # [L,B,H]
    ) -> torch.Tensor:
        """
        Build V using the same in_proj as nn.MultiheadAttention, then reshape to heads.
        value_layer: [B, Hh, L, Dh]
        """
        # convert Q to [B,L,H]
        q_blh = Q.transpose(0, 1)  # [B,L,H]
        B, L, H = q_blh.shape
        Hh = self.num_heads
        Dh = H // Hh

        # in_proj_weight: [3H, H], in_proj_bias: [3H]
        W = self.mha.in_proj_weight
        b = self.mha.in_proj_bias

        # split V part
        W_v = W[2 * H: 3 * H, :]         # [H,H]
        b_v = b[2 * H: 3 * H] if b is not None else None  # [H]

        v = torch.matmul(q_blh, W_v.t())  # [B,L,H]
        if b_v is not None:
            v = v + b_v

        v = v.view(B, L, Hh, Dh).permute(0, 2, 1, 3).contiguous()  # [B,Hh,L,Dh]
        return v

    def forward(
            self,
            Q: torch.Tensor,              # [L,B,H]
            attn_mask: torch.Tensor,      # [L,L]
            layer_norm: nn.LayerNorm,
            output_mixing: bool = False,
            residual_scale: float = 1.0,
        ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

            attn_out, _ = self.mha(Q, Q, Q, attn_mask=attn_mask, need_weights=True)

            # --- head-wise attention (for analysis) ---
            _, attn_weights_h = self.mha(
                Q, Q, Q,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False,   # [B,H,L,L]
            )

            attn_out = self.out_dropout(attn_out)
            pre_ln_states = (residual_scale * Q + attn_out)
            out = pre_ln_states

            analysis_dict = None

            if output_mixing:
                value_layer = self._extract_value_layer(Q)

                attn_probs = (
                    attn_weights_h.unsqueeze(1)
                    if attn_weights_h.dim() == 3
                    else attn_weights_h
                )

                mixing = self.mixing_analyzer(
                    hidden_states=Q.transpose(0, 1),
                    attention_probs=attn_probs,
                    value_layer=value_layer,
                    out_proj=self.mha.out_proj,
                    layer_norm=layer_norm,
                    pre_ln_states=pre_ln_states.transpose(0, 1),
                    residual_scale=residual_scale,
                )

                analysis_dict = {
                    **mixing,
                    "attention": attn_probs.mean(dim=1),  # [B,L,L]
                }

            return out, analysis_dict

    

class LightSASRecAnalyze(nn.Module):
    """
    Follow the specified order:
    Embedding -> Dropout -> LN ->
      [ MHA -> Dropout -> Residual -> LN ] x N ->
    Prediction
    """

    def __init__(
        self,
        item_num: int,
        maxlen: int = 128,
        hidden_units: int = 64,
        num_blocks: int = 1,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
        initializer_range: float = 0.02,
        add_head: bool = True,
        padding_idx: int = 0,
        analysis_dir: Optional[str] = "./analysis_out",
        residual_scale_eval: float = 1.0,
    ):
        super().__init__()

        self.item_num = item_num
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.add_head = add_head
        self.padding_idx = padding_idx
        self.analysis_dir = analysis_dir
        self.residual_scale_eval = residual_scale_eval

        # ===== Embedding =====
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        # ===== Attention Blocks =====
        self.attention_layers = nn.ModuleList([
            AnalyzableMHA(hidden_units, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_blocks)
        ])

        # Per-block Post-LN
        self.block_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8)
            for _ in range(num_blocks)
        ])

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,   # unused
        return_mixing: bool = False,
        save_analysis: bool = False,
        analysis_batch_idx: int = 0,
        apply_residual_scale: bool = False,
    ):
        if isinstance(return_mixing, torch.Tensor):
            return_mixing = False

        B, L = input_ids.size()
        device = input_ids.device

        # ===== Embedding → Dropout → LN =====
        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        seqs = seqs + self.pos_emb(positions)

        seqs = self.emb_dropout(seqs)
        seqs = self.emb_layernorm(seqs)

        # padding mask
        timeline_mask = (input_ids == self.padding_idx)  # [B,L]
        seqs = seqs * (~timeline_mask).unsqueeze(-1)

        # causal mask (True = mask)
        attn_mask = ~torch.tril(
            torch.ones((L, L), dtype=torch.bool, device=device)
        )

        layer_stats = []

        # ===== [ MHA → Dropout → Residual → LN ] × N =====
        for i in range(self.num_blocks):
            # [B,L,H] → [L,B,H]
            seqs_t = seqs.transpose(0, 1)

            # MHA (no LN)
            attn_out, mixing = self.attention_layers[i](
                Q=seqs_t,
                attn_mask=attn_mask,
                layer_norm=self.block_layernorms[i],  # for analysis
                output_mixing=return_mixing,
                residual_scale=self.residual_scale_eval if apply_residual_scale else 1.0,
            )

            # back to [B,L,H]
            attn_out = attn_out.transpose(0, 1)

            # Dropout → Residual → LN
            seqs = self.block_layernorms[i](attn_out)

            # disable padding
            seqs = seqs * (~timeline_mask).unsqueeze(-1)

            if return_mixing:
                layer_stats.append(mixing)

        # ===== Prediction =====
        if self.add_head:
            logits = torch.matmul(seqs, self.item_emb.weight.t())
        else:
            logits = seqs

        if not return_mixing:
            return logits

        analysis = {"layer": layer_stats}

        if save_analysis and self.analysis_dir is not None:
            save_analysis_batch_npz(
                analysis=analysis,
                input_ids=input_ids,
                analysis_dir=self.analysis_dir,
                batch_idx=analysis_batch_idx,
                npz_keys=getattr(self, "npz_keys", None),
            )

        return logits, analysis
    

class PointWiseFFNNoResidual(nn.Module):
    """
    PointWise FFN equivalent to SASRec (Conv1d kernel=1), but without residual
    inside; residual is added externally to preserve analysis design.
    """
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,H] -> [B,H,L]
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.dropout1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.dropout2(y)
        # back: [B,L,H]
        return y.transpose(1, 2)
    



class SASRecAnalyze(nn.Module):
    """
    Forward SASRec including FFN, while analyzing only "Attn + Residual + LN"
    (FFN is not analyzed).
    """

    def __init__(
        self,
        item_num: int,
        maxlen: int = 128,
        hidden_units: int = 64,
        num_blocks: int = 1,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
        initializer_range: float = 0.02,
        add_head: bool = True,
        padding_idx: int = 0,
        analysis_dir: Optional[str] = "./analysis_out",
        residual_scale_eval: float = 1.0,
        use_key_padding_mask: bool = True,  # set True to strictly mask PAD
    ):
        super().__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.padding_idx = padding_idx
        self.analysis_dir = analysis_dir
        self.residual_scale_eval = residual_scale_eval
        self.use_key_padding_mask = use_key_padding_mask

        # Embedding
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        # Attention blocks (analysis target)
        self.attn_layers = nn.ModuleList([
            AnalyzableMHA(hidden_units, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_blocks)
        ])
        self.attn_post_lns = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8)
            for _ in range(num_blocks)
        ])

        # FFN blocks (forward only, not analyzed)
        self.ffn_layers = nn.ModuleList([
            PointWiseFFNNoResidual(hidden_units, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.ffn_post_lns = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8)
            for _ in range(num_blocks)
        ])

        # Final LN (optional; SASRec variants often add it at the end)
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,         # [B,L]
        attention_mask=None,             # unused (for compatibility)
        return_mixing: bool = False,
        save_analysis: bool = False,
        analysis_batch_idx: int = 0,
        apply_residual_scale: bool = False,
    ):
        if isinstance(return_mixing, torch.Tensor):
            return_mixing = False

        B, L = input_ids.shape
        device = input_ids.device

        # ===== Embedding → Dropout → LN =====
        seqs = self.item_emb(input_ids) * (self.hidden_units ** 0.5)  # [B,L,H]
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        seqs = self.emb_layernorm(seqs)

        # padding mask
        timeline_mask = (input_ids == self.padding_idx)  # bool [B,L]
        seqs = seqs * (~timeline_mask).unsqueeze(-1)

        # causal mask (True=mask)
        attn_mask = ~torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))

        layer_stats: List[Dict[str, torch.Tensor]] = []

        # ===== Blocks =====
        for i in range(self.num_blocks):
            # ---- (Attn) ----
            seqs_t = seqs.transpose(0, 1)  # [L,B,H]

            # AnalyzableMHA returns "out = residual_scale*Q + Attn(Q)" (pre-LN)
            attn_out_t, mixing = self.attn_layers[i](
                Q=seqs_t,
                attn_mask=attn_mask,
                layer_norm=self.attn_post_lns[i],  # LN for analysis (post-Attn)
                output_mixing=return_mixing,
                residual_scale=self.residual_scale_eval if (apply_residual_scale) else 1.0,
            )

            # Post-LN (output of Attn+Res+LN)
            seqs = self.attn_post_lns[i](attn_out_t.transpose(0, 1))  # [B,L,H]
            seqs = seqs * (~timeline_mask).unsqueeze(-1)

            # Analysis is finalized here (FFN is not analyzed)
            if return_mixing:
                # mixing includes attn_mixing_ratio / attnres_mixing_ratio / mixing_ratio(=AttnResLN) + attention
                layer_stats.append(mixing)

            # ---- (FFN) forward only ----
            ffn_residual = seqs
            ffn_delta = self.ffn_layers[i](seqs)        # [B,L,H]
            seqs = ffn_residual + ffn_delta             # Residual
            seqs = self.ffn_post_lns[i](seqs)           # LN
            seqs = seqs * (~timeline_mask).unsqueeze(-1)

        # seqs = self.last_layernorm(seqs)

        # ===== Prediction =====
        if self.add_head:
            logits = torch.matmul(seqs, self.item_emb.weight.t())  # [B,L,V]
        else:
            logits = seqs

        if not return_mixing:
            return logits

        analysis = {"layer": layer_stats}

        if save_analysis and (self.analysis_dir is not None):
            save_analysis_batch_npz(
                analysis=analysis,
                input_ids=input_ids,
                analysis_dir=self.analysis_dir,
                batch_idx=analysis_batch_idx,
            )

        return logits, analysis




# ======================================================
# Minimal usage example (optional)
# ======================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    model = LightSASRecAnalyze(
        item_num=100,
        maxlen=8,
        hidden_units=16,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.1,
        add_head=True,
        padding_idx=0,
        analysis_dir="./analysis_out",
    )

    x = torch.tensor([
        [1, 2, 3, 4, 0, 0, 0, 0],
        [5, 6, 7, 0, 0, 0, 0, 0],
    ], dtype=torch.long)

    logits, analysis = model(x, return_mixing=True, save_analysis=True, analysis_batch_idx=0)
    print("logits:", logits.shape)
    print("num_layers:", len(analysis["layer"]))
    print("layer0 attention:", analysis["layer"][0]["attention"].shape)  # [B,Hh,L,L]



# ======================================================
# LightSASRecAnalyze (compatible with LightSASRec)
# ======================================================
# class LightSASRecAnalyze(nn.Module):
#     """
#     Model that returns mixing analysis while matching LightSASRec's forward structure.

#     When return_mixing=True:
#       return logits, analysis_dict
#     """
#     def __init__(
#         self,
#         item_num: int,
#         maxlen: int = 128,
#         hidden_units: int = 64,
#         num_blocks: int = 1,
#         num_heads: int = 1,
#         dropout_rate: float = 0.1,
#         initializer_range: float = 0.02,
#         add_head: bool = True,
#         padding_idx: int = 0,
#         analysis_dir: Optional[str] = "./analysis_out",
#     ):
#         super().__init__()

#         self.item_num = item_num
#         self.maxlen = maxlen
#         self.hidden_units = hidden_units
#         self.num_blocks = num_blocks
#         self.num_heads = num_heads
#         self.dropout_rate = dropout_rate
#         self.initializer_range = initializer_range
#         self.add_head = add_head
#         self.padding_idx = padding_idx
#         self.analysis_dir = analysis_dir

#         self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=self.padding_idx)
#         self.pos_emb = nn.Embedding(maxlen, hidden_units)
#         self.emb_dropout = nn.Dropout(dropout_rate)

#         # Independent LN per block (same as original LightSASRec)
#         self.attention_layernorms = nn.ModuleList([
#             nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)
#         ])

#         # Attention core (no LN inside)
#         self.attention_layers = nn.ModuleList([
#             AnalyzableMHA(hidden_units, num_heads=num_heads, dropout=dropout_rate)
#             for _ in range(num_blocks)
#         ])

#         self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

#         # init
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Conv1d)):
#             module.weight.data.normal_(mean=0.0, std=self.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask=None,                 # ignored (for compatibility)
#         return_mixing: bool = True,
#         save_analysis: bool = True,
#         analysis_batch_idx: int = 0,
#     ):
#         """
#         Returns:
#           - return_mixing=False: logits
#           - return_mixing=True : (logits, analysis_dict)
#         """
#         B, L = input_ids.size()
#         device = input_ids.device

#         # ===== Embedding + Pos + Dropout (same as LightSASRec) =====
#         seqs = self.item_emb(input_ids)
#         seqs *= self.item_emb.embedding_dim ** 0.5

#         positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
#         seqs = seqs + self.pos_emb(positions)
#         seqs = self.emb_dropout(seqs)

#         # pad mask (same as LightSASRec: apply timeline_mask)
#         timeline_mask = (input_ids == self.padding_idx)  # bool [B,L]
#         seqs = seqs * (~timeline_mask).unsqueeze(-1)

#         # causal mask (same as LightSASRec: True=mask)
#         attn_mask = ~torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))

#         layer_stats: List[Dict[str, torch.Tensor]] = []

#         # ===== blocks =====
#         for i in range(len(self.attention_layers)):
#             # [B,L,H] -> [L,B,H]
#             seqs_t = torch.transpose(seqs, 0, 1)

#             # block LN (independent)
#             Q = self.attention_layernorms[i](seqs_t)

#             # mha + residual (LightSASRec-compatible: seqs = Q + mha(Q,Q,Q))
#             out, mixing = self.attention_layers[i](
#                 Q,
#                 attn_mask=attn_mask,
#                 layer_norm=self.attention_layernorms[i],
#                 output_mixing=return_mixing,
#             )

#             seqs = torch.transpose(out, 0, 1)  # back to [B,L,H]
#             seqs = seqs * (~timeline_mask).unsqueeze(-1)

#             if return_mixing:
#                 # mixing is a dict
#                 layer_stats.append(mixing)

#         # ===== last LN (same as LightSASRec) =====
#         outputs = self.last_layernorm(seqs)

#         # ===== head (no LN, same as LightSASRec) =====
#         if self.add_head:
#             logits = torch.matmul(outputs, self.item_emb.weight.t())
#         else:
#             logits = outputs

#         if not return_mixing:
#             return logits

#         analysis = {"layer": layer_stats}

#         if save_analysis and (self.analysis_dir is not None):
#             save_analysis_batch_npz(
#                 analysis=analysis,
#                 input_ids=input_ids,
#                 out_dir=self.analysis_dir,
#                 batch_idx=analysis_batch_idx,
#             )

#         return logits, analysis


class DuoRec(nn.Module):
    """DuoRec: Contrastive Learning for Sequential Recommendation.

    Ported from WEARec (https://github.com/QinHsiu/WEARec).
    ssl='un': unsupervised NCE via dropout augmentation only.
    Reference: Qiu et al., "Contrastive Learning for Representation Degeneration
    Problem in Sequential Recommendation", WSDM 2022.
    """

    def __init__(
        self,
        item_num,
        maxlen=128,
        hidden_units=64,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.1,
        initializer_range=0.02,
        add_head=True,
        padding_idx=0,
        tau=1.0,
        lmd=0.1,
        lmd_sem=None,   # defaults to lmd if None
        sim='dot',
        ssl='un',       # 'un', 'su', 'us', 'us_x'
        analysis_dir: Optional[str] = "./analysis_out",
        residual_scale_eval=1.0,
        **kwargs,
    ):
        super().__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.padding_idx = padding_idx
        self.tau = tau
        self.lmd = lmd
        self.lmd_sem = lmd_sem if lmd_sem is not None else lmd
        self.sim = sim
        self.ssl = ssl
        self.analysis_dir = analysis_dir
        self.residual_scale_eval = residual_scale_eval

        # Embeddings
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Transformer blocks (SASRec-style)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                AnalyzableMHA(hidden_units, num_heads=num_heads, dropout=dropout_rate)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        # Contrastive loss
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self._mask_cache: dict = {}

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _encode(
        self,
        input_ids: torch.Tensor,
        residual_scale: float = 1.0,
        return_mixing: bool = False,
    ):
        """Encode input_ids to hidden states [B, L, H]."""
        B, L = input_ids.shape
        device = input_ids.device

        # Position ids [0, 1, ..., L-1] broadcast over batch
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, L)

        # item embedding (scaled) + position embedding -> Dropout  [SASRec-style]
        seqs = self.item_emb(input_ids) * (self.hidden_units ** 0.5)
        seqs = seqs + self.pos_emb(position_ids)
        seqs = self.emb_dropout(seqs)

        # Padding mask: True where padded
        timeline_mask = (input_ids == self.padding_idx)   # [B, L]
        seqs = seqs * (~timeline_mask).unsqueeze(-1)

        # Causal (left-to-right) attention mask: True means "do not attend"
        causal_mask = ~torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
        layer_stats: List[Dict[str, torch.Tensor]] = []

        for i in range(self.num_blocks):
            seqs = seqs.transpose(0, 1)                    # [L, B, H]
            Q = self.attention_layernorms[i](seqs)
            attn_out_t, mixing = self.attention_layers[i](
                Q=Q,
                attn_mask=causal_mask,
                layer_norm=self.forward_layernorms[i],
                output_mixing=return_mixing,
                residual_scale=residual_scale,
            )
            seqs = attn_out_t
            seqs = seqs.transpose(0, 1)                    # [B, L, H]
            seqs = self.forward_layernorms[i](seqs)
            if return_mixing:
                layer_stats.append(mixing)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask).unsqueeze(-1)

        hidden = self.last_layernorm(seqs)                 # [B, L, H]
        if not return_mixing:
            return hidden

        return hidden, {"layer": layer_stats}

    def _mask_correlated_samples(self, batch_size: int) -> torch.Tensor:
        """2N x 2N boolean mask: True for negative pairs (excludes self and positives)."""
        if batch_size in self._mask_cache:
            return self._mask_cache[batch_size]
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        self._mask_cache[batch_size] = mask
        return mask

    def _info_nce(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        batch_size: int,
    ):
        """Compute InfoNCE logits and labels for a pair of hidden-state tensors [B, L, H]."""
        N = 2 * batch_size
        # Use the last position of each sequence as the sequence representation
        z = torch.cat((z_i[:, -1, :], z_j[:, -1, :]), dim=0)   # [2B, H]

        if self.sim == 'cos':
            sim = torch.nn.functional.cosine_similarity(
                z.unsqueeze(1), z.unsqueeze(0), dim=2
            ) / self.tau
        else:  # 'dot'
            sim = torch.mm(z, z.T) / self.tau                    # [2B, 2B]

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self._mask_correlated_samples(batch_size).to(z.device)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N, dtype=torch.long, device=z.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        same_target: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
        return_mixing: bool = False,
        save_analysis: bool = False,
        analysis_batch_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, L] item indices
            attention_mask: unused (kept for Lightning module compatibility)
            same_target:    [B, L] sequences sharing the same target item (supervised NCE)
            residual_scale: scales residual connections in attention blocks (default 1.0)
        Returns:
            logits [B, L, V]  during inference / eval
            (logits, ssl_loss) tuple during training when lmd > 0
        """
        if isinstance(return_mixing, torch.Tensor):
            return_mixing = False

        encoded = self._encode(
            input_ids,
            residual_scale=residual_scale,
            return_mixing=return_mixing,
        )
        if return_mixing:
            hidden, analysis = encoded
        else:
            hidden = encoded

        if self.add_head:
            logits = torch.matmul(hidden, self.item_emb.weight.T)  # [B, L, V]
        else:
            logits = hidden

        if self.training and self.lmd > 0:
            B = input_ids.shape[0]
            ssl_loss = torch.tensor(0.0, device=input_ids.device)

            # Unsupervised NCE: two forward passes exploit different dropout masks
            if self.ssl in ('un', 'us', 'us_x'):
                hidden_aug = self._encode(input_ids, residual_scale=residual_scale)
                nce_logits, nce_labels = self._info_nce(hidden, hidden_aug, batch_size=B)
                ssl_loss = ssl_loss + self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

            # Supervised NCE: original vs semantic augmentation
            if self.ssl in ('su', 'us') and same_target is not None:
                hidden_sem = self._encode(same_target, residual_scale=residual_scale)
                sem_logits, sem_labels = self._info_nce(hidden, hidden_sem, batch_size=B)
                ssl_loss = ssl_loss + self.lmd_sem * self.aug_nce_fct(sem_logits, sem_labels)

            # Cross-mode NCE: dropout aug vs semantic aug
            if self.ssl == 'us_x' and same_target is not None:
                hidden_aug2 = self._encode(input_ids, residual_scale=residual_scale)
                hidden_sem = self._encode(same_target, residual_scale=residual_scale)
                cross_logits, cross_labels = self._info_nce(hidden_aug2, hidden_sem, batch_size=B)
                ssl_loss = ssl_loss + self.lmd_sem * self.aug_nce_fct(cross_logits, cross_labels)

            return logits, ssl_loss

        if not return_mixing:
            return logits

        if save_analysis and (self.analysis_dir is not None):
            save_analysis_batch_npz(
                analysis=analysis,
                input_ids=input_ids,
                analysis_dir=self.analysis_dir,
                batch_idx=analysis_batch_idx,
                npz_keys=getattr(self, "npz_keys", None),
            )

        return logits, analysis
