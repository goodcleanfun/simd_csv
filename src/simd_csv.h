#ifndef SIMD_CSV_H
#define SIMD_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "aligned/aligned.h"
#include "bit_utils/bit_utils.h"
#include "index_array/uint32_index_array.h"
#include "really_inline/really_inline.h"
#include "simde_avx2/avx2.h"
#include "simde_clmul/clmul.h"


typedef struct simd_csv_input {
    simde__m256i lo;
    simde__m256i hi;
} simd_csv_input_t;


really_inline simd_csv_input_t simd_csv_fill_input(const uint8_t *buffer) {
    simd_csv_input_t input;
    input.lo = simde_mm256_loadu_si256((const simde__m256i *)buffer);
    input.hi = simde_mm256_loadu_si256((const simde__m256i *)(buffer + 32));
    return input;
}

really_inline uint64_t simd_csv_cmp_mask_against_input(simd_csv_input_t input, uint8_t m) {
    const simde__m256i mask = simde_mm256_set1_epi8(m);
    simde__m256i cmp_res_0 = simde_mm256_cmpeq_epi8(input.lo, mask);
    uint64_t res_0 = (uint32_t)(simde_mm256_movemask_epi8(cmp_res_0));
    simde__m256i cmp_res_1 = simde_mm256_cmpeq_epi8(input.hi, mask);
    uint64_t res_1 = simde_mm256_movemask_epi8(cmp_res_1);
    return res_0 | (res_1 << 32);
}

really_inline uint64_t simd_csv_quote_mask(simd_csv_input_t input, uint64_t *prev_inside_quote) {
    uint64_t quote_bits = simd_csv_cmp_mask_against_input(input, '"');

    /* At this point the quote mask for a given input will logically look like:
     * Input:      ,"foo""bar"baz\n
     * quote_bits: 010001100010000
     * quote_mask: 011110111100000
     *
     * Although the bits will be reversed.
     */ 
    uint64_t quote_mask = simde_mm_cvtsi128_si64(
        simde_mm_clmulepi64_si128(
            simde_mm_set_epi64x(0ULL, quote_bits),
            simde_mm_set1_epi8(0xFF),
            0
        )
    );
    /* If the highest bit of the quote mask was a one, then the previous character in the last chunk
     * was inside a quote, 
     */
    quote_mask ^= *prev_inside_quote;
    *prev_inside_quote = (uint64_t)((int64_t)quote_mask >> 63);
    return quote_mask;
}

typedef struct {
    uint8_t delimiter;
    bool crlf;
    bool has_header;
    bool trim_whitespace;
    bool allow_quoted_newlines;
} simd_csv_options_t;

really_inline void simd_csv_flatten_bits(uint32_t *base_ptr, uint32_t *base_idx, uint32_t idx, uint64_t bits) {
    if (bits != 0ULL) {
        uint32_t base = *base_idx;
        uint32_t cnt = popcount(bits);
        uint32_t next_base = base + cnt;
        base_ptr[base + 0] = idx + ctz(bits);
        bits = bits & (bits - 1);
        base_ptr[base + 1] = idx + ctz(bits);
        bits = bits & (bits - 1);
        base_ptr[base + 2] = idx + ctz(bits);
        bits = bits & (bits - 1);
        base_ptr[base + 3] = idx + ctz(bits);
        bits = bits & (bits - 1);
        base_ptr[base + 4] = idx + ctz(bits);
        bits = bits & (bits - 1);
        base_ptr[base + 5] = idx + ctz(bits);
        bits = bits & (bits - 1);
        base_ptr[base + 6] = idx + ctz(bits);
        bits = bits & (bits - 1);
        base_ptr[base + 7] = idx + ctz(bits);
        bits = bits & (bits - 1);
        if (cnt > 8) {
            base_ptr[base + 8] = idx + ctz(bits);
            bits = bits & (bits - 1);
            base_ptr[base + 9] = idx + ctz(bits);
            bits = bits & (bits - 1);
            base_ptr[base + 10] = idx + ctz(bits);
            bits = bits & (bits - 1);
            base_ptr[base + 11] = idx + ctz(bits);
            bits = bits & (bits - 1);
            base_ptr[base + 12] = idx + ctz(bits);
            bits = bits & (bits - 1);
            base_ptr[base + 13] = idx + ctz(bits);
            bits = bits & (bits - 1);
            base_ptr[base + 14] = idx + ctz(bits);
            bits = bits & (bits - 1);
            base_ptr[base + 15] = idx + ctz(bits);
            bits = bits & (bits - 1);
        }
        if (cnt > 16) {
            base += 16;
            do {
                base_ptr[base] = idx + ctz(bits);
                bits = bits & (bits - 1);
                base++;
            } while (bits != 0ULL);
        }
        *base_idx = next_base;
    }
}


bool simd_csv_find_indexes(const uint8_t *buffer, size_t len, uint32_index_array *csv, simd_csv_options_t options) {
    if (buffer == NULL || csv == NULL) return false;
    uint64_t prev_inside_quote = 0ULL;

    uint64_t prev_cr_end = 0ULL;
    size_t len_last_64 = len < 64 ? 0 : len - (len % 64);
    size_t idx = 0;

    #define SIMD_CSV_BUFFER_SIZE 4
    if (len_last_64 > 64 * SIMD_CSV_BUFFER_SIZE) {
        uint64_t fields[SIMD_CSV_BUFFER_SIZE] = {0ULL};
        uint32_t buffer_indexes[SIMD_CSV_BUFFER_SIZE * 64] = {0};
        for (idx = 0; idx < len_last_64 - (len_last_64 % (64 * SIMD_CSV_BUFFER_SIZE)); idx += 64 * SIMD_CSV_BUFFER_SIZE) {
            for (size_t b = 0; b < SIMD_CSV_BUFFER_SIZE; b++) {
                size_t internal_idx = 64 * b + idx;
                #ifndef _MSC_VER
                __builtin_prefetch(buffer + internal_idx + 128);
                #endif

                simd_csv_input_t input = simd_csv_fill_input(buffer + internal_idx);
                uint64_t quote_mask = simd_csv_quote_mask(input, &prev_inside_quote);
                uint64_t delim = simd_csv_cmp_mask_against_input(input, options.delimiter);
                uint64_t end = 0ULL;
                if (options.crlf) {
                    uint64_t cr = simd_csv_cmp_mask_against_input(input, '\r');
                    uint64_t cr_adjusted = (cr << 1) | prev_cr_end;
                    uint64_t lf = simd_csv_cmp_mask_against_input(input, '\n');
                    end = lf & cr_adjusted;
                    prev_cr_end = cr >> 63;
                } else {
                    end = simd_csv_cmp_mask_against_input(input, '\n');
                }
                fields[b] = (end | delim) & ~quote_mask;
            }

            uint32_t count = 0;
            for (size_t b = 0 ; b < SIMD_CSV_BUFFER_SIZE; b++) {
                size_t internal_idx = 64 * b + idx;
                simd_csv_flatten_bits(buffer_indexes, &count, internal_idx, fields[b]);
            }
            uint32_index_array_extend(csv, buffer_indexes, count);
        }
    }

    for (; idx < len_last_64; idx += 64) {
        #ifndef _MSC_VER
        __builtin_prefetch(buffer + idx + 128);
        #endif

        simd_csv_input_t input = simd_csv_fill_input(buffer + idx);
        uint64_t quote_mask = simd_csv_quote_mask(input, &prev_inside_quote);
        uint64_t delim = simd_csv_cmp_mask_against_input(input, options.delimiter);
        uint64_t end = 0ULL;
        uint32_t indexes[64] = {0};
        uint32_t count = 0;

        if (options.crlf) {
            uint64_t cr = simd_csv_cmp_mask_against_input(input, '\r');
            uint64_t cr_adjusted = (cr << 1) | prev_cr_end;
            uint64_t lf = simd_csv_cmp_mask_against_input(input, '\n');
            end = lf & cr_adjusted;
            prev_cr_end = cr >> 63;
        } else {
            end = simd_csv_cmp_mask_against_input(input, '\n');
        }
        uint64_t fields = (end | delim) & ~quote_mask;

        if (fields != 0ULL) {
            simd_csv_flatten_bits(indexes, &count, idx, fields);
            uint32_index_array_extend(csv, indexes, count);
        }
    }

    if (idx < len) {
        uint8_t last_buffer[64] = {0};
        memcpy(last_buffer, buffer + idx, len - idx);
        simd_csv_input_t input = simd_csv_fill_input(last_buffer);
        uint64_t quote_mask = simd_csv_quote_mask(input, &prev_inside_quote);
        uint64_t delim = simd_csv_cmp_mask_against_input(input, options.delimiter);
        
        uint64_t end = 0ULL;
        uint32_t indexes[64] = {0};
        uint32_t count = 0;

        if (options.crlf) {
            uint64_t cr = simd_csv_cmp_mask_against_input(input, '\r');
            uint64_t cr_adjusted = (cr << 1) | prev_cr_end;
            uint64_t lf = simd_csv_cmp_mask_against_input(input, '\n');
            end = lf & cr_adjusted;
            prev_cr_end = cr >> 63;
        } else {
            end = simd_csv_cmp_mask_against_input(input, '\n');
        }
        uint64_t fields = (end | delim) & ~quote_mask;

        if (fields != 0ULL) {
            simd_csv_flatten_bits(indexes, &count, idx, fields);
            uint32_index_array_extend(csv, indexes, count);
        }
    }
    return true;
}

#endif