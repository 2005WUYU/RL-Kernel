// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

// Implementations and input handling are shared with the generic LogP source.
torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_out(
    torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor fused_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_indexed_out(
    torch::Tensor logits, torch::Tensor token_ids,
    torch::Tensor row_indices, torch::Tensor output);
torch::Tensor fused_logp_forward_indexed_fp32(
    torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);
torch::Tensor fused_logp_forward_online_out(
    torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor fused_logp_forward_online_fp32(
    torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_online_indexed_out(
    torch::Tensor logits, torch::Tensor token_ids,
    torch::Tensor row_indices, torch::Tensor output);
torch::Tensor fused_logp_forward_online_indexed_fp32(
    torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RL-Kernel MACA fused selected-token log-probability";
    m.def("fused_logp", &fused_logp_forward);
    m.def("fused_logp_forward_out", &fused_logp_forward_out);
    m.def("fused_logp_forward_fp32", &fused_logp_forward_fp32);
    m.def("fused_logp_forward_indexed_out", &fused_logp_forward_indexed_out);
    m.def("fused_logp_forward_indexed_fp32", &fused_logp_forward_indexed_fp32);
    m.def("fused_logp_forward_online_out", &fused_logp_forward_online_out);
    m.def("fused_logp_forward_online_fp32", &fused_logp_forward_online_fp32);
    m.def("fused_logp_forward_online_indexed_out", &fused_logp_forward_online_indexed_out);
    m.def("fused_logp_forward_online_indexed_fp32", &fused_logp_forward_online_indexed_fp32);
}
