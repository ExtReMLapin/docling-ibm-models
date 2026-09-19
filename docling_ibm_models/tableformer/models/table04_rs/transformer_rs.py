import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import docling_ibm_models.tableformer.utils.utils as u

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class DecoderCache:
    """Incremental decoding state: layer outputs and the projected memory keys/values.

    The previous implementation rebuilt the cache with two ``torch.cat`` calls per
    decoded tag, copying the whole prefix at every step (quadratic in the sequence
    length). Here the prefix lives in a pre-allocated buffer that is written in
    place and grown by doubling; the prefix handed to the next layer is a view.

    ``memory`` is constant for a given table, so its cross-attention key/value
    projections are computed once instead of once per tag and per layer.
    """

    __slots__ = ("buffer", "step", "memory_kv")

    INITIAL_CAPACITY = 64

    def __init__(self, num_layers: int, bsz: int, dim: int, device, dtype):
        self.buffer = torch.empty(
            (num_layers, self.INITIAL_CAPACITY, bsz, dim), device=device, dtype=dtype
        )
        self.step = 0
        self.memory_kv: list = []

    def _grow(self) -> None:
        num_layers, capacity, bsz, dim = self.buffer.shape
        grown = torch.empty(
            (num_layers, 2 * capacity, bsz, dim),
            device=self.buffer.device,
            dtype=self.buffer.dtype,
        )
        grown[:, :capacity] = self.buffer
        self.buffer = grown

    def append(self, layer: int, output: Tensor) -> Tensor:
        """Store this layer's output for the current step and return the prefix."""
        if self.step >= self.buffer.shape[1]:
            self._grow()
        self.buffer[layer, self.step] = output[0]
        return self.buffer[layer, : self.step + 1]


def _project_memory_kv(
    layer: "TMTransformerDecoderLayer", memory: Tensor, n_heads: int
):
    """Project the cross-attention keys/values of ``memory``, shaped for SDPA."""
    attn = layer.multihead_attn
    embed_dim = attn.embed_dim
    head_dim = embed_dim // n_heads
    weight, bias = attn.in_proj_weight, attn.in_proj_bias
    # in_proj_weight stacks [Wq; Wk; Wv]
    key = F.linear(
        memory,
        weight[embed_dim : 2 * embed_dim],
        None if bias is None else bias[embed_dim : 2 * embed_dim],
    )
    value = F.linear(
        memory,
        weight[2 * embed_dim :],
        None if bias is None else bias[2 * embed_dim :],
    )
    src_len, bsz, _ = key.shape
    key = key.reshape(src_len, bsz * n_heads, head_dim).transpose(0, 1)
    value = value.reshape(src_len, bsz * n_heads, head_dim).transpose(0, 1)
    return key, value


class TMTransformerDecoder(nn.TransformerDecoder):
    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional["DecoderCache"] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): encoded tags. (tags_len,bsz,hidden_dim)
            memory (Tensor): encoded image (enc_image_size,bsz,hidden_dim)
            cache (Optional[DecoderCache]): None on the first step of a table, then
                the state returned by the previous step.
        Returns:
            output (Tensor): (tags_len,bsz,hidden_dim)
        """

        if cache is None:
            cache = DecoderCache(
                len(self.layers), tgt.shape[1], tgt.shape[2], tgt.device, tgt.dtype
            )

        n_heads = self.layers[0].self_attn.num_heads
        if memory is not None and not cache.memory_kv:
            cache.memory_kv = [
                _project_memory_kv(mod, memory, n_heads) for mod in self.layers
            ]

        output = tgt
        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                memory,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_kv=cache.memory_kv[i] if cache.memory_kv else None,
            )
            output = cache.append(i, output)

        cache.step += 1

        return output, cache  # type: ignore


class TMTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def _cross_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Cross-attention against keys/values projected once per table."""
        attn = self.multihead_attn
        embed_dim = attn.embed_dim
        n_heads = attn.num_heads
        head_dim = embed_dim // n_heads
        weight, bias = attn.in_proj_weight, attn.in_proj_bias
        query = F.linear(
            query, weight[:embed_dim], None if bias is None else bias[:embed_dim]
        )
        tgt_len, bsz, _ = query.shape
        query = query.reshape(tgt_len, bsz * n_heads, head_dim).transpose(0, 1)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(0, 1).reshape(tgt_len, bsz, embed_dim)
        return attn.out_proj(attended)

    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_kv: Optional[tuple] = None,
    ) -> Tensor:
        """
        Args:
            same as TMTransformerDecoder, plus:
            memory_kv: cross-attention keys and values already projected from
                ``memory``, reused across the decoding steps of one table.
        Returns:
            Tensor:
                During training (seq_len,bsz,hidden_dim)
                If eval mode: embedding of last tag: (1,bsz,hidden_dim)
        """

        # From PyTorch but modified to only use the last tag
        tgt_last_tok = tgt[-1:, :, :]

        tmp_tgt = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # None, because we only care about the last tag
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,  # Optimization: Don't compute attention weights
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        if memory is not None:
            if memory_kv is not None:
                tmp_tgt = self._cross_attention(
                    tgt_last_tok, memory_kv[0], memory_kv[1]
                )
            else:
                tmp_tgt = self.multihead_attn(
                    tgt_last_tok,
                    memory,
                    memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask,
                    need_weights=False,  # Optimization: Don't compute attention weights
                )[0]
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok


class Tag_Transformer(nn.Module):
    """
    "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        device,
        vocab_size,
        td_encode,
        embed_dim,
        encoder_layers,
        decoder_layers,
        enc_image_size,
        dropout=0.1,
        n_heads=4,
        dim_ff=1024,
    ):

        super(Tag_Transformer, self).__init__()

        self._device = device
        self._n_heads = n_heads
        self._embedding = nn.Embedding(vocab_size, embed_dim)
        self._positional_encoding = PositionalEncoding(embed_dim)
        self._td_encode = td_encode

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_ff
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False
        )

        self._decoder = TMTransformerDecoder(
            TMTransformerDecoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=dim_ff,
            ),
            num_layers=decoder_layers,
        )

        self._decoder_dim = embed_dim
        self._enc_image_size = enc_image_size
        self._input_filter = u.resnet_block(stride=1)
        self._fc = nn.Linear(embed_dim, vocab_size)

    def inference(self, enc_inputs, tags, tag_lens, num_cells):
        # CNN backbone image encoding
        enc_inputs = self._input_filter(enc_inputs.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )

        batch_size = enc_inputs.size(0)
        encoder_dim = enc_inputs.size(-1)

        enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim).to(self._device)

        enc_inputs = enc_inputs.permute(1, 0, 2)
        positions = enc_inputs.shape[0]
        # Transformer Encoder Encoded Image mask need to check if its useful
        encoder_mask = torch.zeros(
            (batch_size * self._n_heads, positions, positions), device=self._device
        ) == torch.ones(
            (batch_size * self._n_heads, positions, positions), device=self._device
        )

        # Transformer Encoder
        encoder_out = self._encoder(enc_inputs, mask=encoder_mask)

        decode_lengths = (tag_lens - 1).tolist()

        tgt = self._positional_encoding(self._embedding(tags).permute(1, 0, 2))

        decoded = self._decoder(tgt, memory=encoder_out)
        decoded = decoded.permute(1, 0, 2)
        predictions = self._fc(decoded)
        return predictions, decode_lengths
