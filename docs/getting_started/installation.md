# Installation

RL-Kernel requires Python 3.10 or newer and PyTorch. CUDA builds require a working
CUDA toolchain; ROCm builds require a compatible ROCm environment.

## From Source

For CUDA (or ROCm) source builds, install PyTorch first (matching your CUDA/ROCm
runtime), then build with build isolation disabled so `setup.py` can access
PyTorch's extension build utilities and compile the native kernels (`rl_engine._C`):

```bash
git clone https://github.com/RL-Align/RL-Kernel.git
cd RL-Kernel
# Optional: pin the compile target. If unset, the build targets your GPU's arch.
# export TORCH_CUDA_ARCH_LIST="9.0+PTX"   # e.g. Hopper; or "8.6+PTX", "12.0+PTX"
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e .
```

`RL_KERNEL_REQUIRE_EXT=1` makes the build fail if `_C` cannot be compiled. Without
`--no-build-isolation`, PyTorch is invisible to the isolated build environment.
Confirm the compiled extension is present with:

```bash
python -c "import rl_engine._C as _C; assert hasattr(_C, 'fused_logp'); print(_C.__file__)"
```

A CPU-only install (plain `pip install -e .` on a machine with no GPU) remains
supported and runs on the pure-PyTorch backends.

## Optional Backends

The extras add optional dependencies on top of the compiled package, so they use
the same `--no-build-isolation` flag as the source build above.

```bash
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e ".[cuda]"
```

```bash
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e ".[rocm]"
```

```bash
python -m pip install --no-build-isolation -e ".[vllm]"
```

Install the vLLM extra only on rollout or benchmark environments that need the
vLLM runtime. Core CI and mocked integration tests do not require it.

For common CUDA, ROCm, vLLM, fallback, and CI questions, see the
[FAQ](faq.md).

### ROCm Backend

Use a ROCm PyTorch build that matches the installed ROCm toolchain. Then install
FlashAttention with an AMD backend:

```bash
python -m pip install ninja packaging wheel psutil einops
git clone --recurse-submodules https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  python -m pip install --no-build-isolation --no-deps .
cd ..
```

Verify the environment from the RL-Kernel checkout:

```bash
python tools/env/check_rocm_env.py
```

RL-Kernel uses external FlashAttention as the default ROCm attention path. To
fall back to PyTorch SDPA for ROCm attention dispatch, set:

```bash
export RL_KERNEL_ROCM_ATTN_BACKEND=sdpa
```

### MetaX MACA Backend

Use a MACA environment containing mcPyTorch and the matching cu-bridge compiler.
RL-Kernel reads `torch.version.maca` to identify the `maca` platform. Tensors and
streams continue to use PyTorch's `cuda` API; `torch.version.cuda` is the
compatibility API version, not the MACA runtime version.

From the source checkout, build and inspect the native extension:

```bash
MAX_JOBS=1 python setup.py build_ext --inplace
python - <<'PY'
from rl_engine import _C
from rl_engine.platforms.device import device_ctx
from rl_engine.runtime.registry import kernel_registry

print(device_ctx.device, device_ctx.device_type, device_ctx.backend_version)
print(_C.__file__)
print(type(kernel_registry.get_op("logp", device="cuda:0")).__name__)
PY
```

The device interface is `cuda`, the platform is `maca`, and LogP resolves to
`MacaFusedLogpOp`. The MACA build compiles the generic LogP source with dedicated
bindings. mcPyTorch's extension builder supplies cu-bridge, the include paths,
and runtime libraries. NVIDIA SM90/PTX attention and CUDA Driver IPC sources
belong to the CUDA build and are excluded from this source selection.

The initial native operator computes selected-token LogP from existing logits.
Its forward uses the generic C++ GPU kernel; backward uses the existing PyTorch
FP32 softmax VJP, with the final gradient cast to the input dtype. FP32-output,
online, indexed, and output-buffer LogP interfaces are also exposed. Other
operators use their corresponding PyTorch implementations on the input device.
LinearLogP fusion and GRPO kernel optimization are separate follow-up work.

The semantic catalog exposes this implementation as
`rlkernel.maca.fused_logp` for single-device `selected_logprob` requests with
`device="maca"`. This descriptor records the concrete implementation and its
FP32 reduction path. Both the regular and FP32-output calls support autograd,
including the `selected_logprobs_with_operator` training entry point.

Run the focused validation on a MACA device:

```bash
python -m pytest tests/platforms/test_maca_detection.py \
  tests/build/test_maca_build.py \
  tests/backends/maca/test_maca_fused_logp.py -q
```

The C550 validation uses FP32, FP16, and BF16, including vocabulary sizes at
32/64-thread boundaries and 151936. A CPU FP64 reference consumes the same
quantized inputs. Additional cases cover transposed/strided inputs, indexed
output writes, current-stream ordering, semantic and legacy dispatch, and
PyTorch execution when the extension is absent. The generic kernel's width-32
logical reductions are tested on the C550's physical wave64.

Recorded bring-up environment: C550, mcPyTorch `2.10.0+metax3.8.2.2`, MACA metadata
`3.8.2.2`, and FlagTree `0.7.0.post2+metax3.6` (Triton `3.6.0`). This native LogP
path does not require Triton.

## Development Dependencies

```bash
pip install -e ".[dev]"
pip install -r requirements/docs.txt
```

## Documentation Preview

```bash
mkdocs serve
```

Then open the local URL printed by MkDocs.
