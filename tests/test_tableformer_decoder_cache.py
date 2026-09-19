"""Tests for the incremental decoding cache of the TableFormer tag decoder.

The decoder used to rebuild its cache with `torch.cat` at every step and to
re-project the cross-attention keys/values of `memory` for every tag. These tests
pin the behaviour of the replacement: identical outputs, a cache that grows past
its initial capacity, and memory keys/values that are only projected once.
"""

import torch

from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import (
    DecoderCache,
    TMTransformerDecoder,
    TMTransformerDecoderLayer,
)

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
MEMORY_LEN = 12
BATCH = 1


def _build_decoder(seed: int = 0) -> TMTransformerDecoder:
    torch.manual_seed(seed)
    decoder = TMTransformerDecoder(
        TMTransformerDecoderLayer(d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=64),
        num_layers=N_LAYERS,
    )
    # Inference path only: eval() makes dropout a no-op, so runs are comparable.
    decoder.eval()
    return decoder


def _reference_forward(decoder, tgt, memory, cache):
    """The decoding step as it was implemented before the cache was introduced."""
    output = tgt
    tag_cache = []
    for i, mod in enumerate(decoder.layers):
        output = mod(output, memory)
        tag_cache.append(output)
        if cache is not None:
            output = torch.cat([cache[i], output], dim=0)

    if cache is not None:
        out_cache = torch.cat([cache, torch.stack(tag_cache, dim=0)], dim=1)
    else:
        out_cache = torch.stack(tag_cache, dim=0)

    return output, out_cache


def _decode(decoder, memory, embeddings, reference: bool):
    """Decode `len(embeddings)` steps, returning the last tag embedding of each."""
    cache = None
    outputs = []
    for step in range(len(embeddings)):
        tgt = torch.cat(embeddings[: step + 1], dim=0)
        if reference:
            out, cache = _reference_forward(decoder, tgt, memory, cache)
        else:
            out, cache = decoder(tgt, memory, cache)
        outputs.append(out[-1])
    return outputs, cache


def _inputs(steps: int, seed: int = 1):
    torch.manual_seed(seed)
    memory = torch.randn(MEMORY_LEN, BATCH, D_MODEL)
    embeddings = [torch.randn(1, BATCH, D_MODEL) for _ in range(steps)]
    return memory, embeddings


def test_decoder_matches_reference_implementation():
    """Cached decoding returns what the previous torch.cat implementation returned."""
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=8)

    with torch.no_grad():
        expected, _ = _decode(decoder, memory, embeddings, reference=True)
        actual, _ = _decode(decoder, memory, embeddings, reference=False)

    assert len(actual) == len(expected)
    for step, (got, want) in enumerate(zip(actual, expected)):
        assert torch.allclose(got, want, atol=1e-5), f"mismatch at step {step}"


def test_decoder_matches_reference_beyond_initial_capacity():
    """The buffer grows past DecoderCache.INITIAL_CAPACITY without losing the prefix."""
    steps = DecoderCache.INITIAL_CAPACITY + 5
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=steps, seed=2)

    with torch.no_grad():
        expected, _ = _decode(decoder, memory, embeddings, reference=True)
        actual, cache = _decode(decoder, memory, embeddings, reference=False)

    assert cache.step == steps
    assert cache.buffer.shape[1] > DecoderCache.INITIAL_CAPACITY
    for step, (got, want) in enumerate(zip(actual, expected)):
        assert torch.allclose(got, want, atol=1e-5), f"mismatch at step {step}"


def test_memory_keys_and_values_are_projected_once():
    """`memory` is constant for a table, so its key/value projections are reused."""
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=6, seed=3)

    with torch.no_grad():
        _, cache = _decode(decoder, memory, embeddings, reference=False)
        first_step_kv = None
        cache_after_first = None
        for step in range(len(embeddings)):
            tgt = torch.cat(embeddings[: step + 1], dim=0)
            _, cache_after_first = decoder(tgt, memory, cache_after_first)
            if first_step_kv is None:
                first_step_kv = cache_after_first.memory_kv[0]

    assert len(cache.memory_kv) == N_LAYERS
    for key, value in cache.memory_kv:
        assert key.shape == (BATCH * N_HEADS, MEMORY_LEN, D_MODEL // N_HEADS)
        assert value.shape == key.shape
    # The very tensors projected at the first step are still the ones in use.
    assert cache_after_first.memory_kv[0][0] is first_step_kv[0]
    assert cache_after_first.memory_kv[0][1] is first_step_kv[1]


def test_layer_falls_back_to_multihead_attn_without_cached_kv():
    """Calling a layer without `memory_kv` keeps the original attention path."""
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=3, seed=4)
    layer = decoder.layers[0]
    tgt = torch.cat(embeddings, dim=0)

    with torch.no_grad():
        from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import (
            _project_memory_kv,
        )

        fallback = layer(tgt, memory)
        cached = layer(
            tgt, memory, memory_kv=_project_memory_kv(layer, memory, N_HEADS)
        )

    assert torch.allclose(fallback, cached, atol=1e-5)
