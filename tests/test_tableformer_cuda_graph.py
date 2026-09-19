"""Tests for the CUDA graph path of the TableFormer tag decoder.

The graph path is an optimisation: it has to decode exactly what the regular
implementation decodes, and it has to decline cleanly whenever it cannot run.
The tests that need a GPU are skipped elsewhere, so the fallback tests below are
what runs on a CPU-only CI.
"""

import pytest
import torch

from docling_ibm_models.tableformer.models.table04_rs import graph_decoder
from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import (
    DecoderCache,
    TMTransformerDecoder,
    TMTransformerDecoderLayer,
)

DIM = 64
HEADS = 4
LAYERS = 2
MEMORY_LEN = 24
BATCH = 1

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs need a GPU"
)


def _decoder(device="cpu"):
    torch.manual_seed(0)
    decoder = TMTransformerDecoder(
        TMTransformerDecoderLayer(d_model=DIM, nhead=HEADS, dim_feedforward=128),
        num_layers=LAYERS,
    )
    return decoder.to(device).eval()


def _inputs(steps, device="cpu"):
    torch.manual_seed(1)
    memory = torch.randn(MEMORY_LEN, BATCH, DIM, device=device)
    embeddings = torch.randn(steps, BATCH, DIM, device=device)
    return memory, embeddings


def _decode(decoder, memory, embeddings, **kwargs):
    cache = None
    outputs = []
    for step in range(embeddings.shape[0]):
        out, cache = decoder(embeddings[: step + 1], memory, cache, **kwargs)
        outputs.append(out[-1].clone())
    return outputs, cache


def test_cpu_uses_the_regular_cache():
    """Without CUDA there is nothing to capture, so the regular path is used."""
    decoder = _decoder()
    memory, embeddings = _inputs(steps=5)

    with torch.no_grad():
        _, cache = _decode(decoder, memory, embeddings)

    assert isinstance(cache, DecoderCache)


def test_a_real_memory_mask_is_declined():
    """A mask that actually hides positions is not handled by the static path."""
    decoder = _decoder()
    memory, embeddings = _inputs(steps=3)
    mask = torch.zeros(BATCH, MEMORY_LEN, dtype=torch.bool)
    mask[:, -1] = True

    with torch.no_grad():
        _, cache = _decode(decoder, memory, embeddings, memory_key_padding_mask=mask)

    assert isinstance(cache, DecoderCache)


@requires_cuda
def test_graph_path_matches_regular_path(monkeypatch):
    """Replaying the captured step decodes what the regular implementation does."""
    steps = 12
    memory, embeddings = _inputs(steps=steps, device="cuda")

    monkeypatch.setenv(graph_decoder._DISABLE_ENV, "1")
    with torch.no_grad():
        expected, regular_cache = _decode(_decoder("cuda"), memory, embeddings)

    monkeypatch.delenv(graph_decoder._DISABLE_ENV)
    with torch.no_grad():
        actual, graph_cache = _decode(_decoder("cuda"), memory, embeddings)

    assert isinstance(regular_cache, DecoderCache)
    assert isinstance(graph_cache, graph_decoder.GraphDecoderCache)
    assert graph_cache.step == steps
    for step, (got, want) in enumerate(zip(actual, expected)):
        assert torch.allclose(got, want, atol=1e-4), f"mismatch at step {step}"


@requires_cuda
def test_graph_path_can_be_disabled(monkeypatch):
    monkeypatch.setenv(graph_decoder._DISABLE_ENV, "1")
    memory, embeddings = _inputs(steps=3, device="cuda")

    with torch.no_grad():
        _, cache = _decode(_decoder("cuda"), memory, embeddings)

    assert isinstance(cache, DecoderCache)


@requires_cuda
def test_capacity_grows_for_a_long_table(monkeypatch):
    """A table longer than the initial capacity keeps decoding correctly."""
    monkeypatch.setattr(graph_decoder, "INITIAL_CAPACITY", 8)
    steps = 20  # forces two doublings: 8 -> 16 -> 32
    memory, embeddings = _inputs(steps=steps, device="cuda")

    monkeypatch.setenv(graph_decoder._DISABLE_ENV, "1")
    with torch.no_grad():
        expected, _ = _decode(_decoder("cuda"), memory, embeddings)

    monkeypatch.delenv(graph_decoder._DISABLE_ENV)
    with torch.no_grad():
        actual, cache = _decode(_decoder("cuda"), memory, embeddings)

    assert cache.state.capacity >= steps
    assert cache.step == steps
    for step, (got, want) in enumerate(zip(actual, expected)):
        assert torch.allclose(got, want, atol=1e-4), f"mismatch at step {step}"


@requires_cuda
def test_decoded_states_are_not_aliased():
    """The caller keeps the decoded states, so each step must return its own.

    `predict` collects them for bbox decoding, so a step handing back the buffer
    the next step overwrites would silently corrupt every earlier state. Nothing
    is copied here on purpose: the copy is what is under test.
    """
    decoder = _decoder("cuda")
    memory, embeddings = _inputs(steps=6, device="cuda")

    cache = None
    outputs = []
    with torch.no_grad():
        for step in range(embeddings.shape[0]):
            out, cache = decoder(embeddings[: step + 1], memory, cache)
            outputs.append(out[-1])

    assert isinstance(cache, graph_decoder.GraphDecoderCache)
    pointers = {out.data_ptr() for out in outputs}
    assert len(pointers) == len(outputs), "the decoded states share storage"
    assert not torch.allclose(outputs[0], outputs[-1])
