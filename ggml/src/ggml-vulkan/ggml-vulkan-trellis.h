#pragma once

#include "ggml-vulkan-types.h"

struct vk_trellis_io {
    uint32_t k, n, first;
    uint32_t offsets[7];
    uint32_t strides[3];
    uint32_t ne1, ne2, use_gather;
    int32_t row0;
    uint32_t vocab;
};

struct vk_trellis_product {
    uint32_t k, n, batch, scale_offset;
};

struct vk_trellis_transform {
    uint32_t width, len, run, m, shards, nparts;
    uint32_t parts[GGML_TRELLIS_MAX_PARTS];
    uint32_t step, inverse;
    float scale;
};

size_t ggml_vk_trellis_scratch_stride(const vk_device & device, const ggml_tensor * op);
bool ggml_vk_trellis_supported(const vk_device & device, const ggml_tensor * op);
void ggml_vk_trellis(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * op);
