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
    
class BERT4Rec(nn.Module):

    def __init__(self, vocab_size, bert_config, add_head=True,
                 tie_weights=True, padding_idx=0, init_std=0.02,
                 save_norms=False, analysis_dir=None, residual_scale=1.0):

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

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=bert_config['hidden_size'],
                                        padding_idx=padding_idx)
        self.transformer_model = BertModel(BertConfig(**bert_config))

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

        self.fmlp_layers = nn.ModuleList([
            AnalyzableFMLPFilter(hidden_units, maxlen=maxlen, dropout_rate=dropout_rate)
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
        for i in range(len(self.fmlp_layers)):
            pre_ln, mixing = self.fmlp_layers[i](
                x=seqs,
                layer_norm=self.attn_post_lns[i],
                output_mixing=return_mixing,
                apply_causal_mask=self.use_causal_mask,
                extended_attention_mask=extended_attention_mask,
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
            )

        return logits, analysis
    

class AnalyzableFMLPFilter(nn.Module):
    """
    FFT-based mixing block (no FFN). Returns pre-LN output and optional analysis.
    """
    def __init__(self, hidden_units: int, maxlen: int, dropout_rate: float):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(1, maxlen // 2 + 1, hidden_units, 2) * 0.02
        )
        self.filter_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,              # [B, L, H]
        layer_norm: nn.LayerNorm,
        output_mixing: bool = False,
        apply_causal_mask: bool = False,
        extended_attention_mask: Optional[torch.Tensor] = None,  # [B,1,L,L] or None
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] or None
    ):
        if isinstance(output_mixing, torch.Tensor):
            output_mixing = False
        B, L, H = x.shape

        weight = torch.view_as_complex(self.complex_weight[:, : (L // 2 + 1)])

        # DC-only encoder with optional causal (prefix) averaging.
        if attention_mask is None:
            attention_mask = torch.ones((B, L), device=x.device, dtype=x.dtype)
        else:
            attention_mask = attention_mask.to(dtype=x.dtype)

        if apply_causal_mask:
            masked_x = x * attention_mask.unsqueeze(-1)
            cumsum = masked_x.cumsum(dim=1)
            counts = attention_mask.cumsum(dim=1).unsqueeze(-1)
            x_ifft = cumsum / counts.clamp_min(1.0)
        else:
            total = (x * attention_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
            counts = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
            x_ifft = total / counts.clamp_min(1.0)

        pre_ln = x + self.filter_dropout(x_ifft)

        analysis = None
        if output_mixing:
            if apply_causal_mask:
                tril = torch.tril(torch.ones((L, L), device=x.device))
                counts = tril.sum(dim=1, keepdim=True).clamp_min(1.0)
                T_used = tril / counts
            else:
                T_used = torch.full((L, L), 1.0 / L, device=x.device)
            G = compute_position_contribution(x, T_used)
            post_ln = layer_norm(pre_ln)
            analysis = {
                "mixing_ratio": compute_mixing_ratio_from_G(G),
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
):
    os.makedirs(analysis_dir, exist_ok=True)

    save_dict = {
        "input_ids": input_ids.detach().cpu().numpy()
    }

    for layer_idx, layer_dict in enumerate(analysis["layer"]):
        for k, v in layer_dict.items():
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
        # return
        # ==================================================
        return {
            "post_ln_norm": torch.norm(post_ln, dim=-1),   # ||AttnResLN||
            "attn_mixing_ratio": attn_mixing_ratio,        # Attn-N
            "attnres_mixing_ratio": attnres_mixing_ratio,  # AttnRes-N
            "mixing_ratio": mixing_ratio,                  # AttnResLN-N
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

        seqs = self.last_layernorm(seqs)

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
