// Pure CABAC functions extracted from libde265 for comparison testing
// These are simplified/standalone versions without the full decoder context

#include <stdint.h>
#include <stdbool.h>

extern "C" {

// Context model (same as libde265)
struct ContextModel {
    uint8_t state;
    uint8_t mps;
};

// Simplified CABAC state for comparison
struct CabacState {
    uint32_t range;
    uint32_t value;
    int bits_needed;
    const uint8_t* bitstream_curr;
    const uint8_t* bitstream_end;
};

// CABAC tables from H.265 spec (same as libde265)
static const uint8_t LPS_table[64][4] = {
    { 128, 176, 208, 240}, { 128, 167, 197, 227}, { 128, 158, 187, 216}, { 123, 150, 178, 205},
    { 116, 142, 169, 195}, { 111, 135, 160, 185}, { 105, 128, 152, 175}, { 100, 122, 144, 166},
    {  95, 116, 137, 158}, {  90, 110, 130, 150}, {  85, 104, 123, 142}, {  81,  99, 117, 135},
    {  77,  94, 111, 128}, {  73,  89, 105, 122}, {  69,  85, 100, 116}, {  66,  80,  95, 110},
    {  62,  76,  90, 104}, {  59,  72,  86,  99}, {  56,  69,  81,  94}, {  53,  65,  77,  89},
    {  51,  62,  73,  85}, {  48,  59,  69,  80}, {  46,  56,  66,  76}, {  43,  53,  63,  72},
    {  41,  50,  59,  69}, {  39,  48,  56,  65}, {  37,  45,  54,  62}, {  35,  43,  51,  59},
    {  33,  41,  48,  56}, {  32,  39,  46,  53}, {  30,  37,  43,  50}, {  29,  35,  41,  48},
    {  27,  33,  39,  45}, {  26,  31,  37,  43}, {  24,  30,  35,  41}, {  23,  28,  33,  39},
    {  22,  27,  32,  37}, {  21,  26,  30,  35}, {  20,  24,  29,  33}, {  19,  23,  27,  31},
    {  18,  22,  26,  30}, {  17,  21,  25,  28}, {  16,  20,  23,  27}, {  15,  19,  22,  25},
    {  14,  18,  21,  24}, {  14,  17,  20,  23}, {  13,  16,  19,  22}, {  12,  15,  18,  21},
    {  12,  14,  17,  20}, {  11,  14,  16,  19}, {  11,  13,  15,  18}, {  10,  12,  15,  17},
    {  10,  12,  14,  16}, {   9,  11,  13,  15}, {   9,  11,  12,  14}, {   8,  10,  12,  14},
    {   8,   9,  11,  13}, {   7,   9,  11,  12}, {   7,   9,  10,  12}, {   7,   8,  10,  11},
    {   6,   8,   9,  11}, {   6,   7,   9,  10}, {   6,   7,   8,   9}, {   2,   2,   2,   2}
};

static const uint8_t renorm_table[32] = {
    6,  5,  4,  4,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  2,
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
};

static const uint8_t next_state_MPS[64] = {
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
    17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    49,50,51,52,53,54,55,56,57,58,59,60,61,62,62,63
};

static const uint8_t next_state_LPS[64] = {
    0,0,1,2,2,4,4,5,6,7,8,9,9,11,11,12,
    13,13,15,15,16,16,18,18,19,19,21,21,22,22,23,24,
    24,25,26,26,27,27,28,29,29,30,30,30,31,32,32,33,
    33,33,34,34,35,35,35,36,36,36,37,37,37,38,38,63
};

// Initialize CABAC state
void cabac_init(CabacState* state, const uint8_t* data, int length) {
    state->range = 510;
    state->bits_needed = 8;
    state->bitstream_curr = data;
    state->bitstream_end = data + length;

    // Read initial value (9 bits)
    state->value = 0;
    state->bits_needed = -8;
    if (state->bitstream_curr < state->bitstream_end) {
        state->value = *state->bitstream_curr++;
    }
    state->value <<= 8;
    state->bits_needed = 0;
    if (state->bitstream_curr < state->bitstream_end) {
        state->value |= *state->bitstream_curr++;
        state->bits_needed = -8;
    }
}

// Bypass decode - matches libde265 exactly
int cabac_decode_bypass(CabacState* state) {
    state->value <<= 1;
    state->bits_needed++;

    if (state->bits_needed >= 0) {
        if (state->bitstream_end > state->bitstream_curr) {
            state->bits_needed = -8;
            state->value |= *state->bitstream_curr++;
        } else {
            state->bits_needed = -8;
        }
    }

    int bit;
    uint32_t scaled_range = state->range << 7;
    if (state->value >= scaled_range) {
        state->value -= scaled_range;
        bit = 1;
    } else {
        bit = 0;
    }

    return bit;
}

// Decode fixed-length bypass bits
uint32_t cabac_decode_bypass_bits(CabacState* state, int num_bits) {
    uint32_t value = 0;
    for (int i = 0; i < num_bits; i++) {
        value = (value << 1) | cabac_decode_bypass(state);
    }
    return value;
}

// Decode coeff_abs_level_remaining - matches libde265 exactly
int cabac_decode_coeff_abs_level_remaining(CabacState* state, int rice_param) {
    // Count prefix (unary 1s terminated by 0)
    int prefix = 0;
    while (cabac_decode_bypass(state) != 0 && prefix < 32) {
        prefix++;
    }

    int value;
    if (prefix <= 3) {
        // TR part only
        uint32_t suffix = cabac_decode_bypass_bits(state, rice_param);
        value = (prefix << rice_param) + suffix;
    } else {
        // EGk part
        int suffix_bits = prefix - 3 + rice_param;
        uint32_t suffix = cabac_decode_bypass_bits(state, suffix_bits);
        value = (((1 << (prefix - 3)) + 3 - 1) << rice_param) + suffix;
    }

    return value;
}

// Get current state for comparison
void cabac_get_state(const CabacState* state, uint32_t* range, uint32_t* value, int* bits_needed) {
    *range = state->range;
    *value = state->value;
    *bits_needed = state->bits_needed;
}

// Initialize context model for a given init_value and slice_qp
void context_init(ContextModel* ctx, uint8_t init_value, int slice_qp) {
    int slope = (init_value >> 4) * 5 - 45;
    int offset = ((init_value & 15) << 3) - 16;

    int init_state = ((slope * (slice_qp - 16)) >> 4) + offset;
    if (init_state < 1) init_state = 1;
    if (init_state > 126) init_state = 126;

    if (init_state >= 64) {
        ctx->state = init_state - 64;
        ctx->mps = 1;
    } else {
        ctx->state = 63 - init_state;
        ctx->mps = 0;
    }
}

// Get context state for comparison
void context_get_state(const ContextModel* ctx, uint8_t* state, uint8_t* mps) {
    *state = ctx->state;
    *mps = ctx->mps;
}

// Decode a context-coded bin - matches libde265 exactly
int cabac_decode_bin(CabacState* decoder, ContextModel* model) {
    int decoded_bit;
    int LPS = LPS_table[model->state][(decoder->range >> 6) - 4];
    decoder->range -= LPS;

    uint32_t scaled_range = decoder->range << 7;

    if (decoder->value < scaled_range) {
        // MPS path
        decoded_bit = model->mps;
        model->state = next_state_MPS[model->state];

        if (scaled_range < (256 << 7)) {
            // Renormalize: shift range by one bit
            decoder->range = scaled_range >> 6;
            decoder->value <<= 1;
            decoder->bits_needed++;

            if (decoder->bits_needed == 0) {
                decoder->bits_needed = -8;
                if (decoder->bitstream_curr < decoder->bitstream_end) {
                    decoder->value |= *decoder->bitstream_curr++;
                }
            }
        }
    } else {
        // LPS path
        decoder->value = decoder->value - scaled_range;

        int num_bits = renorm_table[LPS >> 3];
        decoder->value <<= num_bits;
        decoder->range = LPS << num_bits;

        decoded_bit = 1 - model->mps;

        if (model->state == 0) {
            model->mps = 1 - model->mps;
        }
        model->state = next_state_LPS[model->state];

        decoder->bits_needed += num_bits;

        if (decoder->bits_needed >= 0) {
            if (decoder->bitstream_curr < decoder->bitstream_end) {
                decoder->value |= (*decoder->bitstream_curr++) << decoder->bits_needed;
            }
            decoder->bits_needed -= 8;
        }
    }

    return decoded_bit;
}

} // extern "C"
