import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import docling_ibm_models.tableformer.utils.utils as u
from docling_ibm_models.tableformer.models.table04_rs import graph_decoder

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


class DecoderCache:
    """Decoding state of one table: the tag cache, and the memory projections.

    ``tags`` is what the decoder used to hand back on its own. ``memory_kv`` is
    what this cache exists for: ``memory`` does not change while a table is being
    decoded, so its cross-attention keys and values are projected once instead of
    once per tag and per layer.
    """

    __slots__ = ("tags", "memory_kv")

    def __init__(self, memory_kv: list):
        self.tags: Optional[Tensor] = None
        self.memory_kv = memory_kv


class TMTransformerDecoder(nn.TransformerDecoder):
    def _graph_cache(
        self,
        tgt: Tensor,
        memory: Optional[Tensor],
        memory_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
        memory_key_padding_mask: Optional[Tensor],
    ) -> Optional["graph_decoder.GraphDecoderCache"]:
        """Set up CUDA graph decoding for a table, or return None to decline.

        Declining is always safe: the caller then decodes through the regular
        implementation, which this one has to agree with.
        """
        if not graph_decoder.graphs_enabled() or memory is None:
            return None
        if not (tgt.is_cuda and memory.is_cuda) or tgt.dtype != memory.dtype:
            return None
        if memory_mask is not None or tgt_key_padding_mask is not None:
            return None
        if memory_key_padding_mask is not None and bool(memory_key_padding_mask.any()):
            # A mask that hides nothing is common here and harmless; a real one
            # is not handled by the static path.
            return None

        attn = self.layers[0].self_attn
        n_heads, dim = attn.num_heads, attn.embed_dim
        if dim % n_heads:
            return None
        if any(
            layer.self_attn.in_proj_weight is None
            or layer.multihead_attn.in_proj_weight is None
            for layer in self.layers
        ):
            return None

        pool = getattr(self, "_state_pool", None)
        if pool is None:
            pool = graph_decoder.StatePool()
            self._state_pool = pool

        state = pool.get(
            self.layers,
            graph_decoder.INITIAL_CAPACITY,
            tgt.shape[1],
            memory.shape[0],
            n_heads,
            dim,
            tgt.device,
            tgt.dtype,
        )
        if state is None:
            return None

        state.reset()
        state.set_memory(
            [_project_memory_kv(layer, memory, n_heads) for layer in self.layers]
        )
        return graph_decoder.GraphDecoderCache(state, pool, self.layers, n_heads, dim)

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
            cache (Optional[DecoderCache]): None during training and on the first
                decoding step, then the state returned by the previous step.
        Returns:
            output (Tensor): (tags_len,bsz,hidden_dim)
        """

        if isinstance(cache, graph_decoder.GraphDecoderCache):
            cache.prepare()
            # Only the newest tag matters: the prefix is already in the buffers.
            return cache.decode(tgt[-1:]), cache  # type: ignore

        if cache is None:
            graphed = self._graph_cache(
                tgt, memory, memory_mask, tgt_key_padding_mask, memory_key_padding_mask
            )
            if graphed is not None:
                graphed.prepare()
                return graphed.decode(tgt[-1:]), graphed  # type: ignore

            n_heads = self.layers[0].self_attn.num_heads
            cache = DecoderCache(
                []
                if memory is None
                else [_project_memory_kv(mod, memory, n_heads) for mod in self.layers]
            )

        output = tgt

        # cache
        tag_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                memory,
                memory_kv=cache.memory_kv[i] if cache.memory_kv else None,
            )
            tag_cache.append(output)
            if cache.tags is not None:
                output = torch.cat([cache.tags[i], output], dim=0)

        if cache.tags is not None:
            cache.tags = torch.cat([cache.tags, torch.stack(tag_cache, dim=0)], dim=1)
        else:
            cache.tags = torch.stack(tag_cache, dim=0)

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
        position: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            same as TMTransformerDecoder, plus:
            memory_kv: cross-attention keys and values already projected from
                ``memory``, reused across the decoding steps of one table. They
                are enough on their own: ``memory`` may then be None.
            position: index of the tag being decoded. Without it the tag is
                taken to be the last of ``tgt``, which is no longer true when
                ``tgt`` is a padded buffer.
        Returns:
            Tensor:
                During training (seq_len,bsz,hidden_dim)
                If eval mode: embedding of last tag: (1,bsz,hidden_dim)
        """

        # From PyTorch but modified to only use the last tag
        if position is None:
            tgt_last_tok = tgt[-1:, :, :]
        else:
            tgt_last_tok = tgt.index_select(0, position)

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

        if memory_kv is not None:
            tmp_tgt = self._cross_attention(tgt_last_tok, memory_kv[0], memory_kv[1])
        elif memory is not None:
            tmp_tgt = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,  # Optimization: Don't compute attention weights
            )[0]
        else:
            tmp_tgt = None

        if tmp_tgt is not None:
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
