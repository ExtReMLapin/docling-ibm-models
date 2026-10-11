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


class TagDecoderCache:
    """
    Inference-only incremental TMTransformerDecoder: keeps each layer's keys/values
    so a step only processes the new tag. On CUDA the step is a CUDA graph.
    """

    def __init__(self, tt: "Tag_Transformer", max_len: int):
        self._tt, self._layers = tt, tt._decoder.layers
        attn = self._layers[0].self_attn
        E, H, L = attn.embed_dim, attn.num_heads, len(self._layers)
        dev = tt._fc.weight.device
        self._heads = lambda t: t.unflatten(-1, (H, E // H)).permute(1, 2, 0, 3)
        self._k = torch.zeros(L, 1, H, max_len, E // H, device=dev)
        self._v = torch.zeros_like(self._k)
        mem = (L, 1, H, tt._enc_image_size**2, E // H)
        self._mem_k, self._mem_v = torch.zeros(mem, device=dev), torch.zeros(
            mem, device=dev
        )
        self._tag = torch.zeros(1, dtype=torch.long, device=dev)
        self._pos = torch.zeros(1, dtype=torch.long, device=dev)
        self._positions = torch.arange(max_len, device=dev)
        self._graph: Optional[torch.cuda.CUDAGraph] = None

    def set_memory(self, memory: Tensor):
        for i, layer in enumerate(self._layers):
            attn = layer.multihead_attn
            E = attn.embed_dim
            w, b = attn.in_proj_weight[E:], attn.in_proj_bias[E:]
            k, v = F.linear(memory, w, b).chunk(2, dim=-1)
            self._mem_k[i].copy_(self._heads(k))
            self._mem_v[i].copy_(self._heads(v))

    def _attend(self, attn, q, k, v, mask=None):
        # Plain matmuls: SDPA's kernels tile many queries per block, slow for one
        scores = self._heads(q) @ k.transpose(-1, -2) * k.size(-1) ** -0.5
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        out = scores.softmax(-1) @ v
        return attn.out_proj(out.permute(2, 0, 1, 3).flatten(-2))

    def _forward(self):
        pos, mask = self._pos, (self._positions <= self._pos).view(1, 1, 1, -1)
        x = self._tt._embedding(self._tag.view(1, 1))
        x = x + self._tt._positional_encoding.pe.index_select(0, pos)
        for i, layer in enumerate(self._layers):
            sa, ca = layer.self_attn, layer.multihead_attn
            q, k, v = F.linear(x, sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
            self._k[i].index_copy_(2, pos, self._heads(k))
            self._v[i].index_copy_(2, pos, self._heads(v))
            x = layer.norm1(x + self._attend(sa, q, self._k[i], self._v[i], mask))
            q = F.linear(
                x, ca.in_proj_weight[: ca.embed_dim], ca.in_proj_bias[: ca.embed_dim]
            )
            x = layer.norm2(x + self._attend(ca, q, self._mem_k[i], self._mem_v[i]))
            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))
        self._out_H = x[0]
        self._out_tag = self._tt._fc(self._out_H).argmax(1)

    def step(self, tag: int, pos: int):
        """Feed `tag` at `pos`, return (next tag, last hidden state)."""
        self._tag.fill_(tag)
        self._pos.fill_(pos)
        if self._k.device.type != "cuda":
            self._forward()
            return int(self._out_tag.item()), self._out_H
        if self._graph is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):  # warm up before capture
                self._forward()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            # thread_local: other pipeline threads may use CUDA during the capture
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self._forward()
            self._graph = graph
        self._graph.replay()
        return int(self._out_tag.item()), self._out_H.clone()
