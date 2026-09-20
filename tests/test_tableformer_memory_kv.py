"""Tests for the reuse of the cross-attention projections of ``memory``.

``memory`` is the encoded table image: it does not change while a table is being
decoded, yet its keys and values used to be projected again for every tag and
every layer. These tests pin what the reuse must not change — the decoded
values — and what it must do: project once, and leave the original attention
path in place for callers that do not pass projections.
"""

import torch

from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import (
    DecoderCache,
    TMTransformerDecoder,
    TMTransformerDecoderLayer,
    _project_memory_kv,
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
    """The decoding step as it was implemented before the projections were reused."""
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
    """Decode one tag per step, returning the state produced by each."""
    cache = None
    outputs = []
    for step in range(embeddings.shape[0]):
        tgt = embeddings[: step + 1]
        if reference:
            out, cache = _reference_forward(decoder, tgt, memory, cache)
        else:
            out, cache = decoder(tgt, memory, cache)
        outputs.append(out[-1].clone())
    return outputs, cache


def _inputs(steps: int, seed: int = 1):
    torch.manual_seed(seed)
    memory = torch.randn(MEMORY_LEN, BATCH, D_MODEL)
    embeddings = torch.randn(steps, BATCH, D_MODEL)
    return memory, embeddings


def test_decoding_matches_the_previous_implementation():
    """Reusing the projections decodes what projecting every step decoded."""
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=8)

    with torch.no_grad():
        expected, _ = _decode(decoder, memory, embeddings, reference=True)
        actual, _ = _decode(decoder, memory, embeddings, reference=False)

    assert len(actual) == len(expected)
    for step, (got, want) in enumerate(zip(actual, expected)):
        assert torch.allclose(got, want, atol=1e-5), f"mismatch at step {step}"


def test_memory_is_projected_once_per_table():
    """Every step of a table reuses the tensors projected at the first step."""
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=6, seed=3)

    with torch.no_grad():
        cache = None
        first = None
        for step in range(embeddings.shape[0]):
            _, cache = decoder(embeddings[: step + 1], memory, cache)
            if first is None:
                first = cache.memory_kv[0]

    assert isinstance(cache, DecoderCache)
    assert len(cache.memory_kv) == N_LAYERS
    for key, value in cache.memory_kv:
        assert key.shape == (BATCH * N_HEADS, MEMORY_LEN, D_MODEL // N_HEADS)
        assert value.shape == key.shape
    assert cache.memory_kv[0][0] is first[0]
    assert cache.memory_kv[0][1] is first[1]


def test_a_new_table_projects_its_own_memory():
    """Decoding starts with `cache=None`, which is what makes the reuse safe."""
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=3, seed=4)
    other_memory = memory + 1.0

    with torch.no_grad():
        _, first_table = decoder(embeddings[:1], memory, None)
        _, second_table = decoder(embeddings[:1], other_memory, None)

    assert not torch.allclose(first_table.memory_kv[0][0], second_table.memory_kv[0][0])


def test_layer_without_projections_keeps_the_original_attention():
    """`memory_kv=None` leaves the nn.MultiheadAttention path untouched."""
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=3, seed=5)
    layer = decoder.layers[0]

    with torch.no_grad():
        fallback = layer(embeddings, memory)
        cached = layer(
            embeddings, memory, memory_kv=_project_memory_kv(layer, memory, N_HEADS)
        )

    assert torch.allclose(fallback, cached, atol=1e-5)
