#include "ggml-vulkan-common.h"
#include "ggml-vulkan-trellis.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

static constexpr uint32_t VK_TRELLIS_BATCH = 128;
static constexpr uint32_t VK_TRELLIS_BLOCK_LIMIT = 4096;

size_t ggml_vk_trellis_scratch_stride(const vk_device & device, const ggml_tensor * op) {
    const bool rows = op->op == GGML_OP_TRELLIS_GET_ROWS;
    const size_t width = rows ? op->ne[0] : std::max(op->ne[0], op->src[1]->ne[0]);
    const size_t batch = std::min<int64_t>(ggml_nrows(op), VK_TRELLIS_BATCH);
    return ggml_vk_align_size(width * batch * sizeof(float), device->properties.limits.minStorageBufferOffsetAlignment);
}

bool ggml_vk_trellis_supported(const vk_device & device, const ggml_tensor * op) {
    const bool rows = op->op == GGML_OP_TRELLIS_GET_ROWS;
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    const int32_t * desc = op->op_params;
    const auto & limits = device->properties.limits;
    const size_t scratch = ggml_vk_trellis_scratch_stride(device, op);
    if (!ggml_is_trellis_type(w->type) || !ggml_is_contiguous(w) || w->view_src ||
        !ggml_trellis_desc_supported(w->ne[0], w->ne[1], desc, 7 + desc[6]) ||
        2 * scratch > device->max_buffer_size || scratch > limits.maxStorageBufferRange ||
        w->ne[1] / 16 > limits.maxComputeWorkGroupCount[0] ||
        ggml_nbytes(w) > limits.maxStorageBufferRange || ggml_nbytes(op) > limits.maxStorageBufferRange ||
        ggml_nbytes(x) > limits.maxStorageBufferRange) {
        return false;
    }
    return rows ? desc[0] == 0 && x->type == GGML_TYPE_I32 && ggml_is_contiguous(x) &&
                  op->nb[0] == sizeof(float) && op->nb[1] % sizeof(float) == 0
                : x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && ggml_is_contiguous(op) &&
                  x->nb[1] % sizeof(float) == 0 && x->nb[2] % sizeof(float) == 0 && x->nb[3] % sizeof(float) == 0;
}

static void ggml_vk_trellis_side(ggml_backend_vk_context * ctx, vk_context & subctx,
        vk_subbuffer & values, vk_subbuffer & spare, uint32_t width, uint32_t batch,
        const int32_t * side, const int32_t * parts, uint32_t nparts, bool inverse) {
    if (side[0] == 0) {
        return;
    }

    vk_trellis_transform pc = {};
    pc.width = width;
    pc.inverse = inverse;
    if (side[0] == 1) {
        pc.len = side[1];
        pc.shards = width / pc.len;
        pc.m = 1;
    } else {
        pc.shards = side[2];
        pc.len = width / pc.shards;
        pc.m = ggml_trellis_hadfull_m(pc.len);
        pc.nparts = nparts;
        if (nparts) {
            std::copy_n(parts, nparts, pc.parts);
        }
    }
    pc.run = pc.len / pc.m;
    auto cross = [&] {
        auto & pipeline = ctx->device->pipeline_trellis_cross;
        pc.scale = 1.0 / std::sqrt(double(pc.m));
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { values, spare }, pc, { pc.len, pc.shards, batch });
        ggml_vk_sync_buffers(ctx, subctx);
        std::swap(values, spare);
    };
    if (inverse && pc.m > 1) {
        cross();
    }

    const float norm = 1.0 / std::sqrt(double(pc.run));
    if (pc.m == 1 && pc.nparts == 0 && pc.len <= VK_TRELLIS_BLOCK_LIMIT) {
        auto & pipeline = ctx->device->pipeline_trellis_block;
        pc.scale = norm;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { values }, pc, { pc.shards, batch, 1 });
        ggml_vk_sync_buffers(ctx, subctx);
    } else {
        auto & pipeline = ctx->device->pipeline_trellis_butterfly;
        for (pc.step = 1; pc.step < pc.run; pc.step *= 2) {
            pc.scale = 2 * pc.step == pc.run ? norm : 1.0f;
            ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { values }, pc, { pc.len / 2, pc.shards, batch });
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    if (!inverse && pc.m > 1) {
        cross();
    }
}

void ggml_vk_trellis(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    const ggml_tensor * scales = op->src[2];
    const ggml_tensor * pre = op->src[3];
    const ggml_tensor * post = op->src[4];
    const ggml_tensor * pin = op->src[5];
    const ggml_tensor * pout = op->src[6];
    const int32_t * desc = op->op_params;
    const bool rows = op->op == GGML_OP_TRELLIS_GET_ROWS;
    const uint32_t k = w->ne[0], n = w->ne[1];
    const uint32_t bits = w->type - GGML_TYPE_TRL1;
    const uint32_t tokens = ggml_nrows(op);
    const size_t stride = ggml_vk_trellis_scratch_stride(ctx->device, op);

    const auto buffer = [&](const ggml_tensor * t) { return ggml_vk_tensor_subbuffer(ctx, t, true); };
    const auto offset = [&](const ggml_tensor * t) -> uint32_t {
        return t ? get_misalign_bytes(ctx, t) / ggml_type_size(t->type) : 0;
    };
    const auto w_buf = ggml_vk_tensor_subbuffer(ctx, w);
    const auto x_buf = buffer(x), scale_buf = buffer(scales), pre_buf = buffer(pre), post_buf = buffer(post);
    const auto dst_buf = buffer(op), pin_buf = pin ? buffer(pin) : x_buf, pout_buf = pout ? buffer(pout) : post_buf;
    vk_trellis_io pc = {};
    pc.k = k;
    pc.n = n;

    for (pc.first = 0; pc.first < tokens; pc.first += VK_TRELLIS_BATCH) {
        const uint32_t batch = std::min(tokens - pc.first, VK_TRELLIS_BATCH);
        vk_subbuffer a = { ctx->trellis_scratch, 0, stride };
        vk_subbuffer b = { ctx->trellis_scratch, stride, stride };
        // Scratch is shared by unrelated graph nodes.
        ggml_vk_sync_buffers(ctx, subctx);
        pc.k = k;
        if (rows) {
            pc.offsets[0] = 0;
            pc.offsets[1] = offset(scales);
            pc.offsets[2] = offset(x);
            pc.offsets[3] = offset(pin);
            pc.offsets[4] = offset(pre);
            pc.offsets[5] = offset(post);
            pc.offsets[6] = offset(op);
            pc.strides[0] = op->nb[1] / sizeof(float);
            pc.use_gather = pin != nullptr;
            pc.row0 = pin ? desc[GGML_TRELLIS_BAND_ROW0] : 0;
            pc.vocab = pin ? pin->ne[0] : 0;
            auto & decode = ctx->device->pipeline_trellis_decode_rows[bits];
            ggml_pipeline_request_descriptor_sets(ctx, decode, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, decode, { w_buf, scale_buf, x_buf, pin_buf, a }, pc, { k, batch, 1 });
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_vk_trellis_side(ctx, subctx, a, b, k, batch, desc + 3, nullptr, 0, true);
            auto & store = ctx->device->pipeline_trellis_store_rows;
            ggml_pipeline_request_descriptor_sets(ctx, store, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, store, { a, x_buf, pin_buf, pre_buf, post_buf, dst_buf }, pc, { k, batch, 1 });
            continue;
        }

        pc.offsets[0] = offset(x);
        pc.offsets[1] = offset(pre);
        pc.offsets[2] = offset(pin);
        pc.ne1 = x->ne[1];
        pc.ne2 = x->ne[2];
        pc.use_gather = pin != nullptr;
        for (int i = 0; i < 3; ++i) {
            pc.strides[i] = x->nb[i + 1] / sizeof(float);
        }
        auto & input = ctx->device->pipeline_trellis_input;
        ggml_pipeline_request_descriptor_sets(ctx, input, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, input, { x_buf, pre_buf, pin_buf, a }, pc, { k, batch, 1 });
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_trellis_side(ctx, subctx, a, b, k, batch, desc + 3, nullptr, 0, false);

        static const bool coopmat_enabled = [] {
            const char * opt = std::getenv("GGML_VK_TRELLIS_COOPMAT");
            return opt && std::strcmp(opt, "1") == 0;
        }();
        auto & coopmat = ctx->device->pipeline_trellis_matmul_cm1[bits];
        auto & product = batch >= 16 && coopmat_enabled && coopmat ? coopmat :
                         batch <= 4 ? ctx->device->pipeline_trellis_matvec[bits] : ctx->device->pipeline_trellis_matmul[bits];
        ggml_pipeline_request_descriptor_sets(ctx, product, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, product, { w_buf, scale_buf, a, b },
                                  vk_trellis_product{ k, n, batch, offset(scales) }, { n, batch, 1 });
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_trellis_side(ctx, subctx, b, a, n, batch, desc, desc + 7, desc[6], true);

        pc.k = n;
        pc.offsets[1] = offset(post);
        pc.offsets[2] = offset(pout);
        pc.offsets[3] = offset(op);
        pc.use_gather = pout != nullptr;
        auto & output = ctx->device->pipeline_trellis_output;
        ggml_pipeline_request_descriptor_sets(ctx, output, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, output, { b, post_buf, pout_buf, dst_buf }, pc, { n, batch, 1 });
    }
}
