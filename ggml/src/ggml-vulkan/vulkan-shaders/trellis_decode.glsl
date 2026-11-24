#extension GL_EXT_shader_16bit_storage : require

layout(binding = 0) readonly buffer Codes { uint codes[]; };
layout(binding = 1) readonly buffer Scales { float16_t scales[]; };

layout(constant_id = 0) const uint BITS = 3;
const uint TILE_WORDS = 8 * BITS;
const uint TILE_BITS = 256 * BITS;

float trellis_grid(uint state) {
    const uint h = (state & 65535u) * 0x83dcd12du;
    // sum of the four 6-bit bytes of h >> 2 with shifts and adds instead of a multiply by 0x01010101 (a quarter-rate
    // 32-bit integer multiply on RDNA): every byte is below 64, so no sum carries out of its byte
    const uint b = (h >> 2) & 0x3f3f3f3fu;
    const uint t = b + (b >> 16);
    return float(int((t + (t >> 8)) & 0xffu) - 126);
}

// Codes store cyclic 16-bit states in the exllamav3 16x16 tile order.
float trellis_value(uint tile, uint row, uint col) {
    const uint p = 8 * (4 * (row % 8) + (col % 8) / 2) + 4 * (row / 8) + 2 * (col / 8) + col % 2;
    const uint start = ((p + 1) * BITS + TILE_BITS - 16) % TILE_BITS;
    const uint word = start / 32;
    const uint offset = start % 32;
    const uint hi = codes[tile + word];
    uint state = hi >> (16 - min(offset, 16));
    if (offset > 16) {
        const uint lo = codes[tile + (word + 1) % TILE_WORDS];
        state = (hi << (offset - 16)) | (lo >> (48 - offset));
    }
    return trellis_grid(state);
}

vec4 trellis_decode4(uint tile, uint p) {
    const uint end = (p + 1) * BITS;
    const uint start = end >= 16 ? end - 16 : end + TILE_BITS - 16;
    const uint word = start / 32;
    const uint offset = start % 32;
    const uint hi = codes[tile + word];
    const uint lo = codes[tile + (word + 1 == TILE_WORDS ? 0 : word + 1)];
    // Callers start on a multiple of four states. For TRL1-6 this gives a nonzero bit offset.
    const uint window = (hi << offset) | (lo >> (32 - offset));
    vec4 values;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        const uint bit = j * BITS;
        // Only the fourth TRL6 state needs the next two bits after the aligned word.
        const uint state = bit <= 16 ? window >> (16 - min(bit, 16)) : (window << 2) | ((lo >> ((30 - offset) & 31)) & 3);
        values[j] = trellis_grid(state);
    }
    return values;
}
