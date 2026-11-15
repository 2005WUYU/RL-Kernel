# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""C550 checks for the native LogP path, including both halves of a wave64."""

import pytest
import torch

from rl_engine.backends.maca.logprob.logp import MacaFusedLogpOp
from rl_engine.platforms.device import device_ctx
from rl_engine.runtime.operators import selected_logprobs_with_operator
from rl_engine.runtime.registry import KernelRegistry
from rl_engine.runtime.semantic_registry import OperatorRequirements

pytestmark = pytest.mark.skipif(
    device_ctx.device_type != "maca", reason="requires a MetaX MACA device"
)
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOLERANCES = {
    torch.float32: (2e-5, 2e-6),
    torch.float16: (8e-3, 1e-3),
    torch.bfloat16: (6e-2, 8e-3),
}


def _inputs(vocab, dtype, layout="contiguous"):
    generator = torch.Generator().manual_seed(110)
    values = torch.randn(9, vocab, generator=generator)
    # Each row's largest value sits on a different logical/physical wave edge.
    edges = torch.tensor([0, 31, 32, 63, 64, 127, 128, 255, 256]) % vocab
    values[torch.arange(9), edges] = 7.0
    values = values.to(dtype)
    if layout == "transpose":
        logits = values.T.contiguous().cuda().T
    elif layout == "stride_offset":
        storage = torch.empty(9, 2 * vocab + 3, dtype=dtype, device="cuda")
        logits = storage[:, 1 : 2 * vocab + 1 : 2]
        logits.copy_(values)
    else:
        logits = values.cuda()
    # Select peaks in some rows and background probabilities in others.
    labels = edges.clone()
    labels[::2] = (labels[::2] + 1) % vocab
    return logits, labels.cuda(), values.double(), labels


def _reference(values, labels):
    return values.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "vocab", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1025, 151936]
)
def test_forward_backward_wave_edges(dtype, vocab, record_property):
    op = MacaFusedLogpOp()
    logits, labels, values, cpu_labels = _inputs(vocab, dtype)
    logits.requires_grad_()
    values.requires_grad_()
    expected = _reference(values, cpu_labels)
    fp32 = op.apply_fp32(logits, labels)
    actual = op(logits, labels)
    torch.testing.assert_close(fp32.cpu().double(), expected.detach(), atol=2e-5, rtol=2e-6)
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(actual.cpu().double(), expected.detach(), atol=atol, rtol=rtol)
    weights = torch.linspace(-1.0, 1.0, 9, dtype=dtype)
    actual.backward(weights.cuda())
    expected.backward(weights.double())
    torch.testing.assert_close(
        logits.grad.cpu().double(),
        values.grad,
        atol={torch.float32: 3e-6, torch.float16: 1e-3, torch.bfloat16: 8e-3}[dtype],
        rtol=rtol,
    )
    record_property("dtype", str(dtype))
    record_property("vocab", vocab)
    record_property("forward_fp32_max_abs", (fp32.cpu().double() - expected).abs().max().item())
    record_property("forward_output_max_abs", (actual.cpu().double() - expected).abs().max().item())
    record_property(
        "backward_max_abs", (logits.grad.cpu().double() - values.grad).abs().max().item()
    )
    logits.grad = None
    fp32.backward(weights.float().cuda())
    torch.testing.assert_close(
        logits.grad.cpu().double(),
        values.grad,
        atol={torch.float32: 3e-6, torch.float16: 1e-3, torch.bfloat16: 8e-3}[dtype],
        rtol=rtol,
    )
    record_property(
        "fp32_output_backward_max_abs",
        (logits.grad.cpu().double() - values.grad).abs().max().item(),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", [33, 65, 257, 151936])
def test_online_indexed_and_out_variants(dtype, vocab):
    op = MacaFusedLogpOp()
    logits, labels, values, cpu_labels = _inputs(vocab, dtype)
    reference = _reference(values, cpu_labels)
    selected = torch.tensor([0, 3, 8], device="cuda")
    for method in (op.apply_fp32, op.online_fp32):
        torch.testing.assert_close(
            method(logits, labels).cpu().double(), reference, atol=2e-5, rtol=2e-6
        )
    for method in (op.out, op.online_out):
        out = torch.full((9,), -777.0, device="cuda")
        result = method(logits, labels, out)
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(result.cpu().double(), reference, atol=2e-5, rtol=2e-6)
    for method in (op.indexed_out, op.online_indexed_out):
        out = torch.full((9,), -777.0, device="cuda")
        result = method(logits, labels, selected, out)
        expected = torch.full((9,), -777.0, dtype=torch.float64)
        expected[selected.cpu()] = reference[selected.cpu()]
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(result.cpu().double(), expected, atol=2e-5, rtol=2e-6)
    for method in (op.indexed_fp32, op.online_indexed_fp32):
        expected = torch.zeros(9, dtype=torch.float64)
        expected[selected.cpu()] = reference[selected.cpu()]
        torch.testing.assert_close(
            method(logits, labels, selected).cpu().double(), expected, atol=2e-5, rtol=2e-6
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["transpose", "stride_offset"])
def test_input_layouts(dtype, layout):
    logits, labels, values, cpu_labels = _inputs(257, dtype, layout)
    result = MacaFusedLogpOp().apply_fp32(logits, labels)
    torch.testing.assert_close(
        result.cpu().double(), _reference(values, cpu_labels), atol=2e-5, rtol=2e-6
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_current_stream_producer_operator_consumer(dtype):
    op = MacaFusedLogpOp()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        logits, labels, values, cpu_labels = _inputs(151936, dtype)
        logits.mul_(0.75).add_(1.0)
        expected_input = (values.to(dtype) * 0.75).to(dtype).add(1.0).double()
        result = op.apply_fp32(logits, labels) * 3.0 + 0.125
    stream.synchronize()
    expected = _reference(expected_input, cpu_labels) * 3.0 + 0.125
    torch.testing.assert_close(result.cpu().double(), expected, atol=6e-5, rtol=2e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_registry_and_semantic_catalog_reach_maca_extension(dtype):
    registry = KernelRegistry()
    for device in (None, "cuda:0"):
        assert isinstance(registry.get_op("logp", device=device), MacaFusedLogpOp)
    session = registry.semantic.session()
    resolution = session.resolve(
        semantic_op="selected_logprob",
        requested_backend="rlkernel.maca.fused_logp",
        target="training",
        requirements=OperatorRequirements(
            device=device_ctx.device_type,
            dtype=str(dtype).removeprefix("torch."),
            topology={"world_size": 1, "sharding": "unsharded"},
        ),
    )
    operator = session.instantiate(resolution)
    assert isinstance(operator, MacaFusedLogpOp)
    logits, labels, values, cpu_labels = _inputs(257, dtype)
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(
        operator(logits, labels).cpu().double(),
        _reference(values, cpu_labels),
        atol=atol,
        rtol=rtol,
    )
    logits.requires_grad_()
    result = selected_logprobs_with_operator(operator, logits, labels)
    result.sum().backward()
    values.requires_grad_()
    _reference(values, cpu_labels).sum().backward()
    torch.testing.assert_close(
        logits.grad.cpu().double(),
        values.grad,
        atol={torch.float32: 3e-6, torch.float16: 1e-3, torch.bfloat16: 8e-3}[dtype],
        rtol=rtol,
    )


def test_missing_extension_selects_pytorch_on_same_gpu(monkeypatch):
    from rl_engine.backends.cuda.logprob import logp as generic_module
    from rl_engine.reference.logprob.logp import NativeLogpOp

    monkeypatch.setattr(generic_module, "_EXT_AVAILABLE", False)
    registry = KernelRegistry()
    operator = registry.get_op("logp", device="cuda:0")
    assert isinstance(operator, NativeLogpOp)
    logits, labels, values, cpu_labels = _inputs(257, torch.float32)
    result = operator(logits, labels)
    assert result.device == logits.device
    torch.testing.assert_close(
        result.cpu().double(), _reference(values, cpu_labels), atol=2e-5, rtol=2e-6
    )
