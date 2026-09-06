# ARM audit, 2026-09-06

Baseline main: `9e7f7561`. Apple M4 Pro, Rust 1.98, runtime dispatch without
`target-cpu=native`. Builds serialized with nice -n19 and four workers.

Residual addition used wrapping i16 arithmetic in NEON, AVX2 and WASM.
The new regression failed on the original NEON implementation for an 8-bit
prediction of 2 plus residual 32766: 0 instead of 255. The fix XOR-biases
unsigned predictions into the signed domain, saturates the signed addition,
unbiases and clamps with unsigned minimum. This covers u16 predictions too.

The regression exhausts prediction values at depths 8/10/12/15/16, seven
residual extremes, sizes 4/8/16/17/32, row padding and nonzero offsets.
It compares complete planes against independent widened i32 arithmetic.
Native ARM token permutations, x86_64 through Rosetta and WASM SIMD via
Wasmtime pass. WASM requires its compile-time SIMD token; native tests
require at least two token permutations. x86 results are emulated correctness
checks, not native x86 performance evidence.

Native libraries: 82 heic + 26 heic-core tests pass; strict clippy including
test targets passes. The existing Criterion dependency disables only its
Rayon feature so WASI library tests compile; plotting and benchmark support
remain enabled. No test is skipped or ignored by this change.

Commands (prefix heavy invocations with the resource settings above):

```sh
cargo test -p heic -p heic-core --features backend-rust,std,_dev --lib
cargo clippy -p heic -p heic-core --features backend-rust,std,_dev --lib --tests --no-deps -- -D warnings
cargo test -p heic --target x86_64-apple-darwin --features backend-rust,std,_dev --lib residual_add_matches_widened_scalar_at_every_prediction_value -- --nocapture
RUSTFLAGS='-C target-feature=+simd128' cargo test -p heic --target wasm32-wasip1 --features backend-rust,std,_dev --lib residual_add_matches_widened_scalar_at_every_prediction_value -- --nocapture
```

Source inspection
finds that NEON IDCT16 and IDCT32 currently call the scalar implementations;
this report does not count those dispatch labels as vectorized kernels.

## Paired tiers

`benches/tier_isolation.rs` now uses zenbench 0.1.9, explicit fixture paths,
fail-loud decode checks and untimed token mutation. All three whole-decode
fixtures have identical RGBA8 bytes across scalar and NEON.

| Fixture | Dimensions | Native ms | Scalar ms |
|---|---|---:|---:|
| libheif example.heic | 1280×854 | 45.2 | 45.9 |
| Nokia C002.heic | 1280×720 | 24.5 | 24.8 |
| dsoprea image4.heic | 700×476 | 10.4 | 10.5 |

The harness flagged all three comparisons as inconclusive; no reliable
whole-decode gain is claimed. The example row's printed interval is +0.1%
to +2.7% despite that flag; the raw output is retained without silently
reinterpreting the harness verdict.

Strided limited-range BT.709 8-bit 4:4:4 conversion, native/scalar means:
17² 529/533 ns (tie), 64² 4.1/8.5 us, 256² 65.8/78.5 us,
1024² 636.3/695.8 us, 4096² 8.7/9.4 ms. The four larger paired intervals
favor NEON. CV exceeds 20% in several smaller cells; see the full log.
Allocation happens outside timing for color cells; whole decode includes it.
Every color output byte matches, including the 17-wide vector-tail case.
These are existing-kernel results after the residual correctness fix, not
before/after residual speed measurements.

Fixtures are preserved outside git under
`/Users/lilith/work/codec-artifacts/heic-arm-audit/`:

- example.heic: SHA256 `7f8b363e4936c0666a25f64f3a92fda10bd8e5453be4592530b65a55dd98f3f2`
- C002.heic: SHA256 `2b836102e528f7b465b295724b6928c5725a5ac3fe326c6bb54e5e8a18fc180f`
- image4.heic: SHA256 `676b0a76dcaa7fe9ffc41110b7791bef6cf0bfdb32455b03e149b0f8bfdb0856`

Run `HEIC_BENCH_INPUTS=<colon-separated paths> just arm-tier-audit`.
Strict bench clippy passed. Full measurement took 172.4 seconds.


## Native sampling

The example fixture completed 200 RGBA8 decodes. Sampled costs and assembly
provenance are in [profile.pointer.md](profile.pointer.md). The profile identifies
residual/CABAC work and repeated coefficient-buffer copies; it does not establish
a universal ceiling on HEIC optimization. The older CLAUDE.md claims that ARM
4:4:4 stays scalar and that speedup is capped at 2% have been corrected against
current dispatch and the measured limits.

## Caller-owned residual buffer

After the sampled copy sites were identified, residual parsing was changed to
write into a freshly zeroed caller-owned `CoeffBuffer` and return only the
transform-skip flag. This is an internal function in a private module; the
public API and coefficient arithmetic are unchanged. ARM release assembly
no longer contains the three 2052-byte return copies.

All three retained fixtures produce byte-identical RGBA8 before and after the
change (`just arm-capture-rgba`). 81 heic and 26 heic-core library tests pass
with backend-rust,std; strict library/test clippy passes. The separate earlier
108-test run includes the `_dev` residual regression.

Native whole-decode means from separate before/after builds are 45.2 → 40.4 ms
(example), 24.5 → 22.9 ms (C002), and 10.4 → 10.1 ms (image4). These are not
paired before/after confidence intervals. All three new native/scalar paired
intervals cross zero: the copy change benefits both dispatch paths. The
76.2-second run reports eight noisy rounds. The log's git header is parent
`ed85f07f`; the measured source includes this commit's output-buffer change.
