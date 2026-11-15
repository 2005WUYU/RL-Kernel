# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.backends.cuda.logprob.logp import FusedLogpGenericOp, _FusedLogpAutograd


class _MacaFP32LogpAutograd(_FusedLogpAutograd):
    """Keep FP32 native outputs connected to the shared FP32 softmax VJP."""

    @staticmethod
    def forward(ctx, logits, token_ids, backend):
        logits_2d = logits.reshape(-1, logits.size(-1)).contiguous()
        labels = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
        output = backend.fused_logp_forward_fp32(logits_2d, labels)
        ctx.save_for_backward(logits_2d, labels)
        ctx.backend = backend
        ctx.input_shape = tuple(logits.shape)
        ctx.input_dtype = logits.dtype
        return output.reshape(logits.shape[:-1])


class MacaFusedLogpOp(FusedLogpGenericOp):
    """Generic native LogP compiled by cu-bridge, with PyTorch FP32 backward."""

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return _MacaFP32LogpAutograd.apply(logits, token_ids, self._backend)
