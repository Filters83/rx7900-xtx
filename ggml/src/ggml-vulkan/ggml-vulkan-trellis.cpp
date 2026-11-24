#include "ggml-vulkan-common.h"
#include "ggml-vulkan-trellis.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

static constexpr uint32_t VK_TRELLIS_BATCH = 128;
static constexpr uint32_t VK_TRELLIS_BLOCK_LIMIT = 4096;

enum vk_trellis_stage : uint8_t {
    TRELLIS_PROFILE_START,
    TRELLIS_PROFILE_INPUT,
    TRELLIS_PROFILE_ROTATE_IN,
    TRELLIS_PROFILE_PRODUCT,
    TRELLIS_PROFILE_ROTATE_OUT,
    TRELLIS_PROFILE_OUTPUT,
    TRELLIS_PROFILE_ROWS,
    TRELLIS_PROFILE_STORE_ROWS,
    TRELLIS_PROFILE_COUNT,
};

uint32_t ggml_vk_trellis_profile_queries(const ggml_cgraph * graph) {
    static const bool enabled = std::getenv("GGML_VK_TRELLIS_PROFILE") != nullptr;
    if (!enabled || vk_perf_logger_concurrent) {
        return 0;
    }
    uint32_t count = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_TRELLIS || node->op == GGML_OP_TRELLIS_GET_ROWS) {
            const uint32_t stages = node->op == GGML_OP_TRELLIS ? 6 : 4;
            count += stages * CEIL_DIV(ggml_nrows(node), VK_TRELLIS_BATCH);
        }
    }
    return count;
}

static void ggml_vk_trellis_stamp(ggml_backend_vk_context * ctx, vk_context & subctx, vk_trellis_stage stage) {
    if (ctx->trellis_query_base == 0) {
        return;
    }
    const uint32_t index = ctx->trellis_query_base + ctx->trellis_query_stages.size();
    GGML_ASSERT(index < static_cast<uint32_t>(ctx->num_queries));
    subctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, index);
    ctx->trellis_query_stages.push_back(stage);
}

void ggml_vk_trellis_print_profile(ggml_backend_vk_context * ctx) {
    const auto & stages = ctx->trellis_query_stages;
    if (stages.empty()) {
        return;
    }
    std::vector<uint64_t> timestamps(stages.size());
    VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, ctx->trellis_query_base, timestamps.size(),
                timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64),
             "get Trellis timestamp results", ctx->device);

    std::array<uint64_t, TRELLIS_PROFILE_COUNT> ticks = {};
    for (size_t i = 1; i < stages.size(); ++i) {
        if (stages[i] != TRELLIS_PROFILE_START) {
            ticks[stages[i]] += timestamps[i] - timestamps[i - 1];
        }
    }
    const char * names[] = { "start", "input", "rotate_in", "product", "rotate_out", "output", "decode_rows", "store_rows" };
    const double us_per_tick = ctx->device->properties.limits.timestampPeriod / 1000.0;
    std::cerr << "Trellis GPU us:";
    for (size_t i = 1; i < ticks.size(); ++i) {
        std::cerr << " " << names[i] << "=" << ticks[i] * us_per_tick;
    }
    std::cerr << '\n';
}

size_t ggml_vk_trellis_scratch_stride(const vk_device & device, const ggml_tensor * op) {
    const bool rows = op->op == GGML_OP_TRELLIS_GET_ROWS;
    const size_t width = rows ? op->ne[0] : std::max(op->ne[0], op->src[1]->ne[0]);
    const size_t batch = std::min<int64_t>(ggml_nrows(op), VK_TRELLIS_BATCH);
    return ggml_vk_align_size(width * batch * sizeof(float), device->properties.limits.minStorageBufferOffsetAlignment);
}

// int8 activations of trellis_matvec_q8 (trellis_quant.comp): per input 4 bytes of replicated values and, per
// group of 4 inputs, 8 bytes of sum and scale, so 6 bytes per input and token
static constexpr uint32_t VK_TRELLIS_Q8_TOKENS = 4;
size_t ggml_vk_trellis_quant_size(const vk_device & device, const ggml_tensor * op) {
    if (op->op != GGML_OP_TRELLIS) {
        return 0;
    }
    if (ggml_nrows(op) > VK_TRELLIS_Q8_TOKENS) {
        // FP16 activations of trellis_prep16 and their residuals for a whole chunk, then one float per token
        return ggml_vk_align_size(4 * size_t(op->src[0]->ne[0]) * VK_TRELLIS_BATCH + sizeof(float) * VK_TRELLIS_BATCH,
                                  device->properties.limits.minStorageBufferOffsetAlignment);
    }
    return ggml_vk_align_size(6 * size_t(op->src[0]->ne[0]) * ggml_nrows(op), device->properties.limits.minStorageBufferOffsetAlignment);
}

static bool ggml_vk_trellis_mm_enabled() {
    static const bool enabled = [] { const char * opt = std::getenv("GGML_VK_TRELLIS_MM"); return !opt || std::strcmp(opt, "0") != 0; }();
    return enabled;
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

    // GGML_VK_TRELLIS_DEBUG=1: print each distinct transform shape once (width, shards, segment, Paley order, path)
    static const bool debug = std::getenv("GGML_VK_TRELLIS_DEBUG") != nullptr;
    if (debug) {
        static std::mutex mtx;
        static std::set<std::string> seen;
        const bool block = pc.m == 1 && pc.nparts == 0 && pc.len <= VK_TRELLIS_BLOCK_LIMIT;
        const std::string key = "width=" + std::to_string(width) + " kind=" + std::to_string(side[0]) + " shards=" + std::to_string(pc.shards) +
            " len=" + std::to_string(pc.len) + " m=" + std::to_string(pc.m) + " run=" + std::to_string(pc.run) + " nparts=" + std::to_string(pc.nparts) +
            " inverse=" + std::to_string(inverse ? 1 : 0) + " path=" + (block ? "block" : "butterfly");
        std::lock_guard<std::mutex> lock(mtx);
        if (seen.insert(key).second) {
            std::cerr << "trellis transform: " << key << " batch=" << batch << std::endl;
        }
    }
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
        ggml_vk_trellis_stamp(ctx, subctx, TRELLIS_PROFILE_START);
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
            ggml_vk_trellis_stamp(ctx, subctx, TRELLIS_PROFILE_ROWS);
            ggml_vk_trellis_side(ctx, subctx, a, b, k, batch, desc + 3, nullptr, 0, true);
            ggml_vk_trellis_stamp(ctx, subctx, TRELLIS_PROFILE_ROTATE_OUT);
            auto & store = ctx->device->pipeline_trellis_store_rows;
            ggml_pipeline_request_descriptor_sets(ctx, store, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, store, { a, x_buf, pin_buf, pre_buf, post_buf, dst_buf }, pc, { k, batch, 1 });
            ggml_vk_trellis_stamp(ctx, subctx, TRELLIS_PROFILE_STORE_ROWS);
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
        ggml_vk_trellis_stamp(ctx, subctx, TRELLIS_PROFILE_INPUT);
        ggml_vk_trellis_side(ctx, subctx, a, b, k, batch, desc + 3, nullptr, 0, false);
        ggml_vk_trellis_stamp(ctx, subctx, TRELLIS_PROFILE_ROTATE_IN);

        // cooperative-matrix prompt batches by default where the device has them (GGML_VK_TRELLIS_COOPMAT=0 keeps
        // the F32 product)
        static const bool coopmat_enabled = [] {
            const char * opt = std::getenv("GGML_VK_TRELLIS_COOPMAT");
            return !opt || std::strcmp(opt, "0") != 0;
        }();
        auto & coopmat = ctx->device->pipeline_trellis_matmul_cm1[bits];
        // decode and verify batches: int8 activations and dot4 (GGML_VK_TRELLIS_Q8=0 keeps the float product)
        static const bool q8_enabled = [] { const char * opt = std::getenv("GGML_VK_TRELLIS_Q8"); return !opt || std::strcmp(opt, "0") != 0; }();
        const size_t q8_size = ggml_vk_trellis_quant_size(ctx->device, op);
        const bool use_q8 = q8_enabled && batch <= VK_TRELLIS_Q8_TOKENS && q8_size > 0 && ctx->device->pipeline_trellis_matvec_q8[bits] &&
                            ctx->device->pipeline_trellis_matvec_q8x4[bits] &&
                            ctx->trellis_scratch->size >= 2 * stride + q8_size;
        if (use_q8) {
            vk_subbuffer c = { ctx->trellis_scratch, 2 * stride, q8_size };
            auto & quant = ctx->device->pipeline_trellis_quant;
            ggml_pipeline_request_descriptor_sets(ctx, quant, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, quant, { a, c }, vk_trellis_product{ k, n, batch, 0 }, { k / 16, batch, 1 });
            ggml_vk_sync_buffers(ctx, subctx);
            auto & product = batch == 1 ? ctx->device->pipeline_trellis_matvec_q8[bits] : ctx->device->pipeline_trellis_matvec_q8x4[bits];
            ggml_pipeline_request_descriptor_sets(ctx, product, 1);
            // one workgroup per 16 outputs for all tokens of the batch. Products with few outputs (the 48-row delta-net
            // projections of Qwen3.8-27B: 3 workgroups for 5120 inputs) split k over up to 16 workgroups, the partial
            // slices fitting in b, and add the slices afterwards
            const uint32_t groups = n / 16, tiles = k / 16;
            uint32_t splits = 1;
            if (groups < 64) {
                splits = std::min<uint32_t>({ 16, CEIL_DIV(64, groups), tiles / 16, uint32_t(stride / (sizeof(float) * n * batch)) });
                splits = std::max<uint32_t>(splits, 1);
            }
            ggml_vk_dispatch_pipeline(ctx, subctx, product, { w_buf, scale_buf, c, b },
                                      vk_trellis_product{ k, n, batch, offset(scales) }, { n, splits, 1 });
            if (splits > 1) {
                ggml_vk_sync_buffers(ctx, subctx);
                auto & reduce = ctx->device->pipeline_trellis_reduce;
                ggml_pipeline_request_descriptor_sets(ctx, reduce, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, reduce, { b }, vk_trellis_product{ splits, n, batch, 0 }, { n * batch, 1, 1 });
            }
        } else if (ggml_vk_trellis_mm_enabled() && batch >= 16 && coopmat_enabled && ctx->device->pipeline_trellis_mm_cm1[bits] &&
                   q8_size > 0 && ctx->trellis_scratch->size >= 2 * stride + q8_size) {
            // prompt batches: FP16 activations with the tile scales and a power-of-two range per token, then the
            // 32 x 128 cooperative-matrix product (GGML_VK_TRELLIS_MM=0 keeps trellis_matmul_cm1)
            vk_subbuffer c = { ctx->trellis_scratch, 2 * stride, q8_size };
            auto & prep = ctx->device->pipeline_trellis_prep16;
            ggml_pipeline_request_descriptor_sets(ctx, prep, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, prep, { a, scale_buf, c }, vk_trellis_product{ k, n, batch, offset(scales) },
                                      { VK_TRELLIS_BATCH, 1, 1 });
            ggml_vk_sync_buffers(ctx, subctx);
            // GGML_VK_TRELLIS_MM_RESIDUAL=1: also multiply the FP16 rounding residuals of the activations
            static const bool residual = [] { const char * opt = std::getenv("GGML_VK_TRELLIS_MM_RESIDUAL"); return opt && std::strcmp(opt, "1") == 0; }();
            auto & product = residual && ctx->device->pipeline_trellis_mm_res_cm1[bits] ? ctx->device->pipeline_trellis_mm_res_cm1[bits]
                                                                                        : ctx->device->pipeline_trellis_mm_cm1[bits];
            ggml_pipeline_request_descriptor_sets(ctx, product, 1);
            // few outputs (the 48-row delta-net projections: 2 workgroups): split k as in the int8 product
            const uint32_t groups = CEIL_DIV(n, 64), tiles = k / 16;
            uint32_t splits = 1;
            if (groups < 32) {
                splits = std::min<uint32_t>({ 16, CEIL_DIV(64, groups), tiles / 16, uint32_t(stride / (sizeof(float) * n * batch)) });
                splits = std::max<uint32_t>(splits, 1);
            }
            ggml_vk_dispatch_pipeline(ctx, subctx, product, { w_buf, scale_buf, c, b },
                                      vk_trellis_product{ k, n, batch, offset(scales) }, { n, splits, 1 });
            if (splits > 1) {
                ggml_vk_sync_buffers(ctx, subctx);
                auto & reduce = ctx->device->pipeline_trellis_reduce;
                ggml_pipeline_request_descriptor_sets(ctx, reduce, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, reduce, { b }, vk_trellis_product{ splits, n, batch, 0 }, { n * batch, 1, 1 });
            }
        } else {
            auto & product = batch >= 16 && coopmat_enabled && coopmat ? coopmat :
                             batch <= 4 ? ctx->device->pipeline_trellis_matvec[bits] : ctx->device->pipeline_trellis_matmul[bits];
            ggml_pipeline_request_descriptor_sets(ctx, product, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, product, { w_buf, scale_buf, a, b },
                                      vk_trellis_product{ k, n, batch, offset(scales) }, { n, batch, 1 });
        }
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_trellis_stamp(ctx, subctx, TRELLIS_PROFILE_PRODUCT);
        ggml_vk_trellis_side(ctx, subctx, b, a, n, batch, desc, desc + 7, desc[6], true);
        ggml_vk_trellis_stamp(ctx, subctx, TRELLIS_PROFILE_ROTATE_OUT);

        pc.k = n;
        pc.offsets[1] = offset(post);
        pc.offsets[2] = offset(pout);
        pc.offsets[3] = offset(op);
        pc.use_gather = pout != nullptr;
        auto & output = ctx->device->pipeline_trellis_output;
        ggml_pipeline_request_descriptor_sets(ctx, output, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, output, { b, post_buf, pout_buf, dst_buf }, pc, { n, batch, 1 });
        ggml_vk_trellis_stamp(ctx, subctx, TRELLIS_PROFILE_OUTPUT);
    }
}
