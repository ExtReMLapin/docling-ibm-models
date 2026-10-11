import logging
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

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


class TMTransformerDecoder(nn.TransformerDecoder):
    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): encoded tags. (tags_len,bsz,hidden_dim)
            memory (Tensor): encoded image (enc_image_size,bsz,hidden_dim)
            cache (Optional[Tensor]): None during training, only used during inference.
        Returns:
            output (Tensor): (tags_len,bsz,hidden_dim)
        """

        output = tgt

        # cache
        tag_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            tag_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            out_cache = torch.cat([cache, torch.stack(tag_cache, dim=0)], dim=1)
        else:
            out_cache = torch.stack(tag_cache, dim=0)

        return output, out_cache  # type: ignore


class TMTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            same as TMTransformerDecoder
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


def _single_query_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    # q: (1, H, 1, D). Plain matmuls: SDPA's memory-efficient kernel tiles 64
    # queries per block, so with a single query it runs only H blocks.
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.size(-1) ** -0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(scores.softmax(-1), v)


class TagDecoderCache:
    """
    Incremental tag decoding for inference, one tag at a time.

    Same result as running TMTransformerDecoder (eval mode) on the whole tag
    sequence at each step, but each layer's self-attention keys/values and the
    memory keys/values are kept, so a step only processes the new tag.
    On CUDA the step is captured once in a CUDA graph and replayed.
    Not thread-safe: use one instance per model and thread.
    """

    def __init__(self, tag_transformer: "Tag_Transformer", max_len: int):
        self._tt = tag_transformer
        self._layers = list(tag_transformer._decoder.layers)
        attn = self._layers[0].self_attn
        self._n_heads = attn.num_heads
        self._head_dim = attn.embed_dim // attn.num_heads
        self._max_len = max_len

        weight = tag_transformer._fc.weight
        device = weight.device
        shape = (len(self._layers), 1, self._n_heads, max_len, self._head_dim)
        self._k = torch.zeros(shape, device=device, dtype=weight.dtype)
        self._v = torch.zeros_like(self._k)
        self._memory_k: Optional[Tensor] = None
        self._memory_v: Optional[Tensor] = None

        # Static inputs/outputs of the CUDA graph
        self._use_graph = device.type == "cuda"
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._tag = torch.zeros(1, dtype=torch.long, device=device)
        self._pos = torch.zeros(1, dtype=torch.long, device=device)
        self._positions = torch.arange(max_len, device=device)
        self._out_tag: Optional[Tensor] = None
        self._out_H: Optional[Tensor] = None

    def _heads(self, t: Tensor) -> Tensor:  # (L, 1, E) -> (1, H, L, D)
        return t.unflatten(-1, (self._n_heads, self._head_dim)).permute(1, 2, 0, 3)

    def set_memory(self, memory: Tensor):
        r"""
        memory: encoder output (positions, 1, embed_dim)
        """
        mem_k, mem_v = [], []
        for layer in self._layers:
            attn = layer.multihead_attn
            E = attn.embed_dim
            k, v = F.linear(
                memory, attn.in_proj_weight[E:], attn.in_proj_bias[E:]
            ).chunk(2, dim=-1)
            mem_k.append(self._heads(k))
            mem_v.append(self._heads(v))
        k, v = torch.stack(mem_k), torch.stack(mem_v)
        if (
            self._memory_k is not None
            and self._memory_v is not None
            and self._memory_k.shape == k.shape
        ):
            # The CUDA graph reads these buffers: update them in place
            self._memory_k.copy_(k)
            self._memory_v.copy_(v)
        else:
            self._memory_k, self._memory_v = k, v
            self._graph = None

    def _forward(self, tag: Tensor, pos: Tensor, num_keys: Optional[int]):
        memory_k, memory_v = self._memory_k, self._memory_v
        assert memory_k is not None and memory_v is not None, "call set_memory()"
        mask = None
        if num_keys is None:  # graph: static shapes, mask the unused positions
            mask = (self._positions <= pos).view(1, 1, 1, -1)
        x = self._tt._embedding(tag.view(1, 1))
        x = x + self._tt._positional_encoding.pe.index_select(0, pos)
        for i, layer in enumerate(self._layers):
            attn = layer.self_attn
            q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(
                3, dim=-1
            )
            self._k[i].index_copy_(2, pos, self._heads(k))
            self._v[i].index_copy_(2, pos, self._heads(v))
            keys, values = self._k[i], self._v[i]
            if num_keys is not None:
                keys, values = keys[:, :, :num_keys], values[:, :, :num_keys]
            out = _single_query_attention(self._heads(q), keys, values, mask)
            x = layer.norm1(x + attn.out_proj(out.permute(2, 0, 1, 3).flatten(-2)))

            attn = layer.multihead_attn
            E = attn.embed_dim
            q = F.linear(x, attn.in_proj_weight[:E], attn.in_proj_bias[:E])
            out = _single_query_attention(self._heads(q), memory_k[i], memory_v[i])
            x = layer.norm2(x + attn.out_proj(out.permute(2, 0, 1, 3).flatten(-2)))

            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))
        last_H = x[0]  # (1, embed_dim)
        return self._tt._fc(last_H).argmax(1), last_H

    def _capture(self):
        stream = torch.cuda.Stream(self._pos.device)
        stream.wait_stream(torch.cuda.current_stream(self._pos.device))
        with torch.cuda.stream(stream):
            for _ in range(2):  # warm up before capture
                self._forward(self._tag, self._pos, None)
        torch.cuda.current_stream(self._pos.device).wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        # thread_local: other pipeline threads may use CUDA during the capture
        with torch.cuda.graph(self._graph, capture_error_mode="thread_local"):
            self._out_tag, self._out_H = self._forward(self._tag, self._pos, None)

    def step(self, tag: int, pos: int):
        r"""
        Feed `tag` at position `pos` and return (predicted next tag, last hidden state)
        """
        if not self._use_graph:
            device = self._k.device
            tag_t = torch.tensor([tag], device=device)
            pos_t = torch.tensor([pos], device=device)
            new_tag, last_H = self._forward(tag_t, pos_t, pos + 1)
            return int(new_tag.item()), last_H
        if self._graph is None:
            self._capture()
        assert self._graph is not None
        assert self._out_tag is not None and self._out_H is not None
        self._tag.fill_(tag)
        self._pos.fill_(pos)
        self._graph.replay()
        return int(self._out_tag.item()), self._out_H.clone()
