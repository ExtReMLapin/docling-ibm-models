"""Tests for the buffer holding the decoded prefix of a table.

The decoder used to rebuild its cache with `torch.cat` at every tag. These tests
pin what the buffer must not change -- the decoded values -- and the one thing
it has to get right on its own: growing past its initial capacity without losing
the prefix.
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
    """The decoding step as it was implemented before the buffer."""
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
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=8)

    with torch.no_grad():
        expected, _ = _decode(decoder, memory, embeddings, reference=True)
        actual, _ = _decode(decoder, memory, embeddings, reference=False)

    assert len(actual) == len(expected)
    for step, (got, want) in enumerate(zip(actual, expected)):
        assert torch.allclose(got, want, atol=1e-5), f"mismatch at step {step}"


def test_buffer_grows_without_losing_the_prefix():
    """A table longer than the initial capacity decodes the same values."""
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


def test_earlier_states_survive_later_tags():
    """Each position is written once, so what a step returned stays valid.

    `predict` keeps the decoded states for bbox decoding. The references taken
    during decoding are compared, at the end, against copies taken at the time:
    a buffer position rewritten by a later tag would show up here.
    """
    decoder = _build_decoder()
    memory, embeddings = _inputs(steps=6, seed=3)

    with torch.no_grad():
        cache = None
        live, snapshots = [], []
        for step in range(embeddings.shape[0]):
            out, cache = decoder(embeddings[: step + 1], memory, cache)
            live.append(out[-1])
            snapshots.append(out[-1].clone())

    for step, (now, then) in enumerate(zip(live, snapshots)):
        assert torch.equal(now, then), f"state of step {step} was overwritten"
