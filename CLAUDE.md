# HEIC Decoder Project Instructions

## Project Overview

Pure Rust HEIC/HEIF image decoder. No C/C++ dependencies.

## Build Commands

```bash
cargo build
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Test Files

- `/home/lilith/work/heic/libheif/examples/example.heic` (1280x854)
- `/home/lilith/work/heic/test-images/classic-car-iphone12pro.heic` (3024x4032)

## Reference Implementations

- libde265 (C++): `/home/lilith/work/heic/libde265-src/`
- OpenHEVC (C): `/home/lilith/work/heic/openhevc-src/`

Do NOT use web searches for HEVC spec details - read the reference implementations directly.

## API Design Guidelines

Follow `/home/lilith/work/codec-design/README.md` for codec API design patterns:

### Decoder Design Principles
- **Layered API**: Simple one-shot functions + builder for advanced use
- **Info before decode**: Allow inspection without full decode
- **Zero-copy decode_into**: For performance-critical paths
- **Multiple output formats**: RGBA, RGB, YUV, etc.

### Example API Shape (future)
```rust
// Simple one-shot
pub fn decode_rgba(data: &[u8]) -> Result<(Vec<u8>, u32, u32)>;

// Typed pixel output
pub fn decode<P: DecodePixel>(data: &[u8]) -> Result<(Vec<P>, u32, u32)>;

// Builder for advanced options
pub struct Decoder<'a> { ... }
impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self>;
    pub fn info(&self) -> &ImageInfo;
    pub fn decode_rgba(self) -> Result<ImgVec<RGBA8>>;
}

// Zero-copy into pre-allocated buffer
pub fn decode_rgba_into(
    data: &[u8],
    output: &mut [u8],
    stride_bytes: u32
) -> Result<(u32, u32)>;
```

### Essential Crates
- `rgb` - Typed pixel structs (RGB8, RGBA8, etc.)
- `imgref` - Strided 2D image views
- `bytemuck` - Safe transmute for SIMD

### SIMD Strategy
- Use `wide` crate (1.1.1) for portable SIMD types
- Use `multiversed` for runtime dispatch
- Place dispatch at high level, `#[inline(always)]` for inner loops
- See codec-design README for archmage usage in complex operations

## Code Style

- Use `div_ceil()` instead of `(x + n - 1) / n`
- Use `is_multiple_of()` instead of `x % n == 0`
- Collapse nested `if` with `&&` when possible
- Use iterators with `.enumerate()` instead of manual counters

## Current Implementation Status

### Completed
- HEIF container parsing (boxes.rs, parser.rs)
- NAL unit parsing (bitstream.rs)
- VPS/SPS/PPS parsing (params.rs)
- Slice header parsing (slice.rs)
- CTU/CU quad-tree decoding structure (ctu.rs)
- Intra prediction modes (intra.rs)
- Transform matrices and inverse DCT/DST (transform.rs)
- CABAC tables and decoder framework (cabac.rs)
- Frame buffer with YCbCr→RGB conversion (picture.rs)
- Transform coefficient parsing via CABAC (residual.rs)
- Adaptive Golomb-Rice coefficient decoding
- DC coefficient inference for coded sub-blocks
- Sign data hiding (all 280 CTUs now decode)
- Debug infrastructure (debug.rs) with CABAC tracker
- sig_coeff_flag proper H.265 context derivation

### In Progress
- Debug remaining chroma bias (Cb=159, Cr=175 vs expected ~128)
- Investigate coeff_abs_level_remaining producing large values

### Pending
- Compare coefficient output against libde265 reference decoder
- Conformance window cropping
- Deblocking filter
- SAO (Sample Adaptive Offset)
- SIMD optimization

## Known Limitations

- Only I-slices supported (sufficient for HEIC still images)
- No inter prediction (P/B slices)
- coded_sub_block_flag context is simplified (proper derivation caused worse desync)

## Known Bugs

### CABAC Context Derivation (Mostly Fixed)
- **sig_coeff_flag:** ✅ Fixed with proper H.265 9.3.4.2.5 context derivation
- **prev_csbf bit ordering:** ✅ Fixed (bit0=below, bit1=right per H.265)
- **greater1_flag:** ✅ Fixed with ctxSet*4 + greater1Ctx formula per H.265
- **greater2_flag:** ✅ Fixed to use ctxSet (0-3) instead of always 0
- **coded_sub_block_flag:** Uses simplified single-context (proper derivation caused worse desync)
- **Current status after fixes (2026-01-22):**
  - All 280 CTUs decode successfully
  - Large coefficients (>500): 24 (first at call #258, byte 1421)
  - High coefficients (>100): many, first at call #8, byte 55 (value=-175)
  - Pixel average: 160
  - Chroma averages: Cb=161, Cr=173 (improved from 198/210, target ~128)

- **last_significant_coeff_prefix context:**
  - Using flat ctxOffset=0 for luma instead of size-dependent offset
  - The H.265 correct approach (ctxOffset = 3*(log2Size-2) + ((log2Size-1)>>2)) causes early termination at CTU 67
  - This suggests desync is present but masked by the simpler context derivation

### Remaining Chroma Bias - Root Cause Analysis (2026-01-22)

**Symptoms:**
- Cr plane average ~209 instead of expected ~128
- CTU columns 0-3 have reasonable Cr (~124-135)
- CTU columns 4+ have elevated Cr (183-230)

**Root cause traced to Cr TU at (104,0):**
- Coefficient at scan position 3 (buffer position 2) has value 1064
- This coefficient has remaining_level = 1062 (base=2, so 2+1062=1064)
- Value 1062 requires prefix=12 in Golomb-Rice decoding (12 consecutive 1-bits)

**CABAC state analysis:**
- After decoding pos=5 remaining, CABAC state is (range=356, value>>7=355)
- This state is at the boundary where bypass decoding produces many 1-bits
- value=45440, range<<7=45568, so after shift: value*2 >= range<<7 always true

**Corruption propagation:**
1. Large coefficient 1064 after inverse transform gives residual ~248 at some positions
2. Adding residual 248 to prediction 135 overflows to 255 (clipped)
3. Neighboring TUs use corrupted reference samples (255 instead of ~135)
4. Corruption spreads through intra prediction to subsequent CTUs

**Key file locations:**
- `residual.rs:285` - decode_residual call #285 (problematic TU)
- `residual.rs:801-813` - coeff_abs_level_remaining with large prefix
- `ctu.rs:722-725` - Coefficient buffer at (104,0) showing [0, -9, 1064, 4, ...]

**Investigation update (2026-01-22):**
- First high coefficient (>100) appears at call #8 (value=-175), byte 55
- First large coefficient (>500) appears at call #258 (value=514), byte 1421
- Rice parameter reaches max (4) due to accumulating high coefficients
- Pattern suggests bitstream desync starting very early

**Next step:** Create comparison test with hevc-compare crate to compare our CABAC output against libde265 step-by-step, starting from byte 0.

### Other Issues
- Output dimensions 1280x856 vs reference 1280x854 (missing conformance window cropping)

## Investigation Notes

### Sign Data Hiding Progress (2026-01-21)

**Background:** HEVC has a "sign data hiding" feature (`sign_data_hiding_enabled_flag` in PPS)
that allows the encoder to infer one sign bit per 4x4 sub-block from coefficient parity.

**Fixes implemented:**
1. DC coefficient inference for coded sub-blocks (was decoding instead of inferring)
2. sig_coeff_flag decoding for position 15 in non-last sub-blocks (was skipping)
3. Sign decoding order matches libde265 (high scan pos to low)
4. Parity inference for hidden sign (sum & 1 flips sign)

**Progress:**
- Initially: CABAC desync at CTU 49 (49/280)
- After DC inference fix: CTU 161 (161/280)
- After position 15 fix: CTU 272 (272/280)
- After scan table investigation: CTU 269 (269/280)

**Remaining issue at CTU 269:**
- 11 CTUs near end of image fail to decode with sign hiding enabled
- Sign hiding disabled allows all 280 CTUs to decode
- The exact cause is not yet identified

**hevc-compare crate (crates/hevc-compare/):**
- Comparison crate for testing C++/Rust CABAC functions
- All basic CABAC tests pass (bypass decode, bypass bits, coeff_abs_level_remaining)
- Can be extended to test more coefficient decoding operations

### greater1_flag/greater2_flag Context Fix (2026-01-22)

**Problem:** Chroma averages were 198/210 instead of ~128.

**Root cause:** Context index derivation for coeff_abs_level_greater1_flag was incorrect.
Our implementation used `c1` directly (0-3), but H.265/libde265 requires:
- `ctxSet` (0-3): based on subblock position and previous subblock's c1 state
- `greater1Ctx` (0-3): starts at 1 each subblock, modified per coefficient

**Formula:** `ctxSet * 4 + min(greater1Ctx, 3) + (c_idx > 0 ? 16 : 0)`

**ctxSet derivation:**
- DC block (sb_idx==0) or chroma: base = 0
- Non-DC luma: base = 2
- If previous subblock ended with c1==0: ctxSet++

**greater1Ctx state machine:**
- Reset to 1 at start of each subblock
- Before decoding each coefficient (except first):
  - If previous greater1_flag was 1: greater1Ctx = 0
  - Else if greater1Ctx > 0: greater1Ctx++ (capped at 3 when using)

**greater2_flag:** Uses `ctxSet + (c_idx > 0 ? 4 : 0)` instead of just the chroma offset.

**Results:** Chroma averages improved from 198/210 to 161/173. Still not at target ~128.

### Context Derivation Analysis (2026-01-22)

**Debug infrastructure added:** CabacTracker in debug.rs tracks:
- CTU start byte positions
- Large coefficient occurrences (>500, indicates CABAC desync)
- First desync location for debugging

**Findings from example.heic:**
- First large coefficient at byte 1713 (in CTU 1, very early)
- 38 total large coefficients detected
- CABAC state becomes corrupt progressively
- Chroma prediction averages drift: 128 → 156 → 207 → 367 (impossible)

**Root cause identified:**
The simplified context selection for sig_coeff_flag (residual.rs:625):
```rust
let ctx_idx = context::SIG_COEFF_FLAG + if c_idx > 0 { 27 } else { 0 };
```
Uses a single context regardless of position, instead of the full H.265 derivation
which depends on position, sub-block location, TU size, and neighbors.

**Fix needed:** Implement full context derivation per H.265 section 9.3.4.2.5:
- Calculate sigCtx based on position within 4x4 sub-block
- Consider coded sub-block flag of neighbors
- Different logic for luma vs chroma
- Different logic for position 0 (DC) vs others

**Reference:** libde265 `decode_sig_coeff_flag()` in slice.cc

### Chroma Bias Analysis (2026-01-21 Session 1)
- Test image: example.heic (1280x854)
- Y plane: avg=152 (reasonable for outdoor scene)
- Cb plane: avg=167 (should be ~128, ~39 too high)
- Cr plane: avg=209 (should be ~128, ~81 too high)
- First chroma block at (0,0) has values ~100-150 (reasonable)
- Bias is not uniform - some regions more affected than others
- Chroma QP = 17 (same as luma, PPS/slice offsets are 0)
- Diagonal scan tables have unconventional order but consistently so for both
  coefficient and sub-block scanning, suggesting they compensate for each other
- CTU column 0 chroma values are reasonable (avg ~128), bias appears starting at column 1+

## Module Structure

```
src/
├── lib.rs           # Public API
├── error.rs         # Error types
├── heif/
│   ├── mod.rs
│   ├── boxes.rs     # ISOBMFF box definitions
│   └── parser.rs    # Container parsing
└── hevc/
    ├── mod.rs       # Main decode entry point
    ├── bitstream.rs # NAL unit parsing, BitstreamReader
    ├── params.rs    # VPS, SPS, PPS
    ├── slice.rs     # Slice header parsing
    ├── ctu.rs       # CTU/CU decoding, SliceContext
    ├── intra.rs     # Intra prediction (35 modes)
    ├── cabac.rs     # CABAC decoder, context tables
    ├── residual.rs  # Transform coefficient parsing
    ├── transform.rs # Inverse DCT/DST
    ├── debug.rs     # CABAC tracker, invariant checks
    └── picture.rs   # Frame buffer
```

## FEEDBACK.md

See `/home/lilith/.claude/CLAUDE.md` for global instructions including feedback logging.
