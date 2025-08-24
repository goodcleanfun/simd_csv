#ifndef SIMD_CSV_H
#define SIMD_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "aligned/aligned.h"
#include "bit_utils/bit_utils.h"
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

really_inline uint64_t simd_csv_quote_mask(const uint8_t *buffer, uint64_t *prev_inside_quote) {
    if (buffer == NULL) return 0;

    uint64_t quote_bits = simd_csv_cmp_mask_against_input(
        simd_csv_fill_input(buffer), '"'
    );

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



#endif