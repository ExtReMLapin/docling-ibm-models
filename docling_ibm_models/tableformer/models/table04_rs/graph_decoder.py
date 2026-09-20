"""CUDA graph replay for the TableFormer tag decoder.

Decoding a table emits one tag at a time, and each step runs a handful of small
kernels per layer. At these sizes the GPU is idle most of the time: the cost is
dominated by launching the kernels, not by the arithmetic. Capturing one step in
a CUDA graph and replaying it removes that launch cost.

A graph can only be replayed if every tensor it touches keeps its shape and its
address, so a table is decoded through buffers allocated once: the prefix each
layer reads, padded to the capacity and masked past the current tag, the
cross-attention keys/values of ``memory``, and the step input and output. The
layers themselves are called as usual, which is what keeps this module small:
the decoding arithmetic lives in one place only.

Attention runs over the whole padded prefix, so an oversized buffer costs real
work: decoding starts with room for `INITIAL_CAPACITY` tags and the capacity
doubles only when a table outgrows it. Each capacity gets its own captured
graph, and buffers and graphs are reused by every later table.

Everything here is an optimisation of the inference path. Without CUDA, when a
capture fails, or in any case this module declines to handle, the decoder falls
back to its regular implementation, which stays the reference.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

# Most tables decode well under a hundred tags; starting small keeps the masked
# attention cheap for them.
INITIAL_CAPACITY = 128

_DISABLE_ENV = "DOCLING_TABLEFORMER_DISABLE_CUDA_GRAPH"


def graphs_enabled() -> bool:
    """CUDA graphs are used by default; set the env var to fall back."""
    return os.environ.get(_DISABLE_ENV, "") not in ("1", "true", "True")


class StaticDecoderState:
    """Buffers for one capacity, and the graph captured against them.

    Instances are owned by the decoder module and reused across tables: starting
    a new table only writes into the buffers, which keeps the addresses the
    graph was captured against valid.
    """

    def __init__(self, layers, capacity, bsz, memory_len, n_heads, dim, device, dtype):
        self.layers = list(layers)
        self.capacity = capacity
        self.bsz = bsz
        self.memory_len = memory_len
        self.head_dim = dim // n_heads

        kwargs = {"device": device, "dtype": dtype}
        # One prefix per layer input, plus one for the output of the last layer.
        self.hidden = [
            torch.zeros(capacity, bsz, dim, **kwargs)
            for _ in range(len(self.layers) + 1)
        ]
        rows = bsz * n_heads
        self.memory_k = [
            torch.zeros(rows, memory_len, self.head_dim, **kwargs) for _ in self.layers
        ]
        self.memory_v = [
            torch.zeros(rows, memory_len, self.head_dim, **kwargs) for _ in self.layers
        ]
        # Key padding mask: True hides a position that has not been decoded yet.
        self.mask = torch.ones(bsz, capacity, dtype=torch.bool, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.step_input = torch.zeros(1, bsz, dim, **kwargs)
        self.step_output = torch.zeros(1, bsz, dim, **kwargs)
        self.graph: Optional["torch.cuda.CUDAGraph"] = None

    def _run_step(self) -> None:
        """One decoding step, reading `step_input` and writing `step_output`."""
        self.mask.index_fill_(1, self.position, False)
        self.hidden[0].index_copy_(0, self.position, self.step_input)
        for i, layer in enumerate(self.layers):
            output = layer(
                self.hidden[i],
                None,
                tgt_key_padding_mask=self.mask,
                memory_kv=(self.memory_k[i], self.memory_v[i]),
                position=self.position,
            )
            self.hidden[i + 1].index_copy_(0, self.position, output)
        self.step_output.copy_(output)
        self.position += 1

    def capture(self) -> bool:
        """Capture one decoding step. Returns False if capture is not possible."""
        try:
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(3):  # required warmup before a capture
                    self._run_step()
            torch.cuda.current_stream().wait_stream(side_stream)
            self.reset()

            graph = torch.cuda.CUDAGraph()
            # `thread_local` instead of the default `global`: the decoder runs
            # inside pipelines that keep other threads busy on the same device,
            # and a global capture forbids their CUDA calls for its duration.
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self._run_step()
            self.graph = graph
        except Exception:
            # A capture can fail for reasons outside this module (another thread
            # capturing, a driver that declines). Decoding must still work.
            self.graph = None
        self.reset()
        return self.graph is not None

    def reset(self) -> None:
        """Forget the decoded prefix, keeping the buffers and their addresses."""
        self.position.zero_()
        self.mask.fill_(True)

    def set_memory(self, memory_kv: List[Tuple[Tensor, Tensor]]) -> None:
        for i, (key, value) in enumerate(memory_kv):
            self.memory_k[i].copy_(key)
            self.memory_v[i].copy_(value)

    def adopt(self, other: "StaticDecoderState") -> None:
        """Carry a decoded prefix, and its memory, over to a larger capacity."""
        length = other.capacity
        for i, hidden in enumerate(other.hidden):
            self.hidden[i][:length].copy_(hidden)
        for i in range(len(self.layers)):
            self.memory_k[i].copy_(other.memory_k[i])
            self.memory_v[i].copy_(other.memory_v[i])
        self.mask[:, :length].copy_(other.mask)
        self.position.copy_(other.position)

    def step(self, hidden: Tensor) -> Tensor:
        self.step_input.copy_(hidden)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._run_step()
        return self.step_output


class StatePool:
    """The decoding states a decoder module keeps, one per capacity."""

    def __init__(self):
        self._states: Dict[Tuple[int, int, int], StaticDecoderState] = {}

    def get(
        self,
        layers,
        capacity,
        bsz,
        memory_len,
        n_heads,
        dim,
        device,
        dtype,
        require_graph: bool = True,
    ) -> Optional[StaticDecoderState]:
        """Return a state for this capacity, capturing its graph on first use.

        `require_graph` is set while the regular implementation is still an
        option: without a graph the static path has no reason to exist, so a
        failed capture declines. Once a table is being decoded there is no way
        back, and a graph-less state is used rather than none.
        """
        key = (capacity, bsz, memory_len)
        state = self._states.get(key)
        if state is None:
            state = StaticDecoderState(
                layers, capacity, bsz, memory_len, n_heads, dim, device, dtype
            )
            if not state.capture() and require_graph:
                return None
            self._states[key] = state
        return state


class GraphDecoderCache:
    """Decoding state handed back to the caller between the steps of one table.

    The number of decoded tags is tracked here, on the CPU, so that a step needs
    no synchronisation of its own.
    """

    __slots__ = ("state", "pool", "length", "layers", "n_heads", "dim")

    def __init__(self, state, pool, layers, n_heads, dim):
        self.state = state
        self.pool = pool
        self.length = 0
        self.layers = layers
        self.n_heads = n_heads
        self.dim = dim

    @property
    def step(self) -> int:
        """Number of decoded tags, named like `DecoderCache.step`."""
        return self.length

    def prepare(self) -> bool:
        """Make room for one more tag, growing the capacity when needed."""
        state = self.state
        if self.length < state.capacity:
            return True
        bigger = self.pool.get(
            self.layers,
            2 * state.capacity,
            state.bsz,
            state.memory_len,
            self.n_heads,
            self.dim,
            state.position.device,
            state.step_input.dtype,
            require_graph=False,
        )
        if bigger is None:
            return False
        bigger.adopt(state)
        self.state = bigger
        return True

    def decode(self, hidden: Tensor) -> Tensor:
        # The caller keeps the decoded hidden states around (they feed bbox
        # decoding), while the step output is a buffer the next step overwrites,
        # so what leaves here has to be a copy.
        output = self.state.step(hidden).clone()
        self.length += 1
        return output
