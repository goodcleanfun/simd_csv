# simd_csv
A fast streaming CSV parser using SIMDe for token classification

In a quick benchmark (with RDTSC) on an Apple x64 laptop, vectorized SIMD CSV parsing gets around 2.5 GB/s on the sample CSV (nfl.csv) provided in the original implementation: https://github.com/geofflangdale/simdcsv, and is about 8.5x faster than a naive implementation (scans along the tokens looking for matching characters, except when inside quotes).