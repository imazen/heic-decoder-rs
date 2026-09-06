# HEIC Decoder Project Instructions

## ABSOLUTE MANDATE: NEVER STOP PUSHING FORWARD

**While working on a multi-step task in this repo, you do not stop, pause, lazily
ask "shall I continue?", or wrap up the session early — unless one of the
genuine stop conditions in `~/.claude/CLAUDE.md` ("NEVER PAUSE LAZILY") fires.**
The default is: ship the next chunk. Then the next. Then the next.

**Stopping criteria — these are the ONLY legitimate reasons to pause:**

1. The next step is **destructive or externally visible** (force-push, publish,
   posting outside this org, deleting shared state) — confirm per the global
   "Executing actions with care" rules.
2. The next step **branches in a way that's not derivable from prior
   instructions** — two equally-valid paths with no expressed preference.
3. You hit a **measured-and-verified** blocker (no GPU, no Apple HW, missing
   library version not on crates.io) and there is no chunk you can ship
   that doesn't depend on that block.
4. Native-API runtime tests across all six backend targets pass, AND every
   item on the user's last list is shipped or has a documented chunk that
   cannot be shipped today.

**Forbidden — these are NOT stop conditions:**

* "I've done a lot already, let me summarize."
* "Want me to keep going?"
* "Should I continue with X, or hold?"
* "The next step is heavy."
* "Compile takes a while."
* "CI is yellow on one job."
* "I'm not sure if you want me to..."
* End-of-session-feeling because the conversation is long.

If a chunk is genuinely too large to ship in one pass, decompose into the
smallest demoable chunk (per the global "NEVER GIVE UP ON A USER-DIRECTED
LIST" rule), land that chunk with a passing test, document the next chunk
with file paths + signatures, and **keep going on the next chunk** in the
same turn. Do not stop to ask.

## Writing Good Code — patterns imported from jxl-encoder

These patterns are mandatory reading and apply to every commit. Adapted from
`~/work/zen/jxl-encoder/CLAUDE.md` "Patterns of Mistakes to Avoid" + "Proof-
by-Tests Investigation Methodology" + "Invariant Preservation Across Sessions".

### 1. False positives are the highest-severity bug

Tests that pass without exercising the thing they claim to test are worse
than no tests — they manufacture false confidence and waste future
investigation time. For a decoder, "the parser accepted the bytes" is
**not** evidence "the image decoded correctly" — you must call all the
way through `decode_to_frame` / `to_rgba` / the backend's `decode_hevc`
and inspect actual pixels. The `mf_diff` example + zensim-regress corpus
diff are the canonical templates for this codebase.

Rules:
- **Never** declare a backend "works" based on `is_available()` alone.
  Decode example.heic via the backend and verify dimensions + pixel
  equivalence (zensim ≥ 95) against the rust backend.
- **Never** trust a test-count delta. Verify what the tests actually do.
- When fixing a "tests pass but it's wrong" bug, audit every other test
  that uses the same pattern.

### 2. Read existing docs before investigating

Before "investigating" any bug, read:

1. This file's "Known Bugs" and "Investigation Notes" sections.
2. `CHANGELOG.md` `[Unreleased]` and the most recent shipped version.
3. `git log --since="3 days ago" --oneline -30`.
4. `git log --grep="<error fragment>" --all`.
5. The relevant backend's `PORTING.md` if applicable.

If the bug is already documented, continue the existing investigation —
do not start a new note. Update in place.

### 3. Test the test infrastructure

When you add a test helper (corpus harness, fake fixtures, diff utility),
exercise it against known-good and known-bad inputs before relying on it.
A test helper bug is worse than the bug it's trying to catch — it
poisons every test built on top.

### 4. Documentation reflects what's verified, not what's intended

Before claiming a feature "works" in CLAUDE.md, README, CHANGELOG, or a
commit message:

- Decode a real HEIC end-to-end via the path being claimed.
- Compare against libheif / the rust backend / dec265 — at least two
  external sources of truth.
- For native backends, the bar is `compare_backends_via_zensim` reporting
  zero failed files in the bundled corpus.

Status markers (used in CLAUDE.md + CHANGELOG):

| Marker | Meaning |
|---|---|
| ✓ Complete | Works end-to-end, ≥ 2 cross-checks, runtime CI green. |
| ⚠ Partial | Some inputs work, others fail; failure mode documented. |
| ⚙ In Progress | Implementation exists, not yet exercised against real input. |
| ✗ Broken | Implementation exists, known-failing test pinned. |
| ❌ Not Started | No implementation. |

### 5. One commit, one complete fix

Multiple `fix: correct X` commits for the same `X` within a day means the
first fix was shipped without understanding. Before fixing a bug:

- Trace every consumer of the wrong data.
- Write a failing test that reproduces it.
- Understand **why** the bug exists, not just where.

After: verify with a different code path (e.g. the MF backend fix
verified via both `mediafoundation_alone_decodes_when_required` and the
corpus zensim diff).

### 6. Investigation lives in ONE place

CLAUDE.md "Investigation Notes" is the single source of truth. Do NOT
create `STATUS.md`, `NOTES.md`, `INVESTIGATION-of-foo.md` files — use
dated entries in CLAUDE.md instead. Multiple symptoms (UnexpectedEof,
InvalidEnum, byte corruption) may share a root cause; link related
findings instead of duplicating them.

### 7. Read code before claiming you understand it

Before committing an implementation:

- Read it line by line. Verify variable names match semantics
  (`crop_y` vs `coded_y`, `width` vs `coded_width`).
- Check that doc comments match what the code does.
- Verify every computed value is actually used (don't compute a value
  and then not consume it).

For ports (chromium → Rust):

- Read the reference implementation completely before writing the port.
- Don't assume "similar" Rust code does the same thing as the C++.
- Verify matching inputs produce matching outputs (parity tests).

### 8. Add tracing FIRST when writing bitstream / FFI code

For new bitstream paths or native FFI: add diagnostic logging *before*
shipping the first `unsafe` block. The `mf_diff` example was added
*after* example.heic was discovered to be broken — should have been
written the moment we started writing the MF unpacker. Future native
backends (D3D11VA, VA-API real decode): write the equivalent of
`mf_diff` for that backend before declaring decode "complete".

### 9. Proof-by-Tests investigation (layered invariants)

When debugging, build a stack of invariant tests from coarsest to finest
and commit each one as it passes:

- **Layer 0**: Does it compile? Do existing tests pass?
- **Layer 1**: Does the new component roundtrip in isolation?
- **Layer 2**: Does the byte-level serialization match the reference?
- **Layer 3**: Does the full pipeline produce output a reference decoder
  accepts? (libheif, dec265, the rust backend)
- **Layer 4**: Is the output perceptually correct on real photos?
  (zensim, SSIM2)

When a layer passes, record it in this file with the test name and the
commit hash. Don't re-investigate passed layers. Focus on the first
failing layer — that's where the bug lives.

### 10. Pre-commit checklist

Run before every commit:

```bash
cargo fmt --all
cargo clippy --workspace --features backend-rust -- -D warnings
cargo test --features backend-rust,std --lib
```

For changes to MF / VT / MediaCodec / VA-API / D3D11VA backends, also
verify the per-backend cross-compile path that matches CI:

```bash
cargo clippy -p heic-backend-mediafoundation --target x86_64-pc-windows-gnu -- -D warnings
cargo clippy -p heic-backend-mediacodec --target aarch64-linux-android -- -D warnings
```

For MF-specific changes, run the Windows host tests via
`pwsh.exe -File V:\heic-win-test.ps1` and confirm 5/5 dispatch tests pass.

### 11. Bitstream / FFI tracing is permanent

`examples/mf_diff.rs`, `examples/probe_backends.rs`, the zensim corpus
diff harness — these stay in tree forever. They're the regression gate
that catches the next "chroma offset of 4 pixels" bug. Do not remove
them once they catch their first bug; they catch the next one too.

## Reference: chromium HEVC source

The chromium tree is sparse-checked-out at `~/work/chromium` with only
the HEVC-relevant files materialized:

* `media/gpu/h265_decoder.{cc,h}` — generic Accelerator-delegated
  decoder loop.
* `media/gpu/windows/d3d11_h265_accelerator.{cc,h}` — DXVA decode FFI.
* `media/gpu/vaapi/h265_vaapi_video_decoder_delegate.{cc,h}` — libva
  decode FFI.
* `media/parsers/h265_parser.{cc,h}` — SPS/PPS/slice header parsing.

Use these as reference when porting decoder paths. The
`heic-backend-d3d11va/PORTING.md` and `heic-backend-vaapi/PORTING.md`
guides cross-reference these files with line numbers.

## Project Overview

Pure Rust HEIC/HEIF image decoder. No C/C++ dependencies.

## Build Commands

```bash
cargo build
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --test compare_reference -- --nocapture  # SSIM2 comparison
cargo test --test compare_reference write_comparison_images -- --nocapture --ignored  # Write PPMs
```

## Test Files

- `/home/lilith/work/heic/libheif/examples/example.heic` (1280x854)
- `/home/lilith/work/heic/test-images/classic-car-iphone12pro.heic` (3024x4032)

## Reference Implementations

- libde265 (C++): `/home/lilith/work/heic/libde265-src/`
- OpenHEVC (C): `/home/lilith/work/heic/openhevc-src/`

## HEVC Specification

**ITU-T H.265 (08/2021)** organized by decoder component:
- `/home/lilith/work/heic/spec/sections/README.md` - Index
- `/home/lilith/work/heic/spec/sections/09-decoding/03-slice-decoding.md` - Slice/CTU/CU decoding
- `/home/lilith/work/heic/spec/sections/10-parsing/cabac/` - CABAC context derivation
- Key sections for coefficient decode: 9.3.4.2.5 (sig_coeff_flag ctx), 9.3.4.2.6 (greater1_flag ctx)

Do NOT use web searches for HEVC spec details - read the spec sections or reference implementations directly.

## API Design

Follows the zen codec three-layer pattern from `/home/lilith/work/codec-design/README.md`:

```rust
// Simple one-shot
let output = DecoderConfig::new().decode(&data, PixelLayout::Rgba8)?;

// Full control with limits and cancellation
let output = DecoderConfig::new()
    .decode_request(&data)
    .with_output_layout(PixelLayout::Rgba8)
    .with_limits(&limits)
    .with_stop(&cancel)
    .decode()?;

// Zero-copy into pre-allocated buffer
let info = ImageInfo::from_bytes(&data)?;
let mut buf = vec![0u8; info.output_buffer_size(PixelLayout::Rgba8).unwrap()];
let (w, h) = DecoderConfig::new()
    .decode_request(&data)
    .with_output_layout(PixelLayout::Rgba8)
    .decode_into(&mut buf)?; // returns (width, height)

// Probe without decoding
let info = ImageInfo::from_bytes(&data)?;

// Raw YCbCr access
let frame = DecoderConfig::new().decode_to_frame(&data)?;

// HDR gain map
let gainmap = DecoderConfig::new().decode_gain_map(&data)?;

// EXIF/XMP extraction (zero-copy for single-extent, owned for multi-extent)
let exif: Option<Cow<'_, [u8]>> = DecoderConfig::new().extract_exif(&data)?;
let xmp: Option<Cow<'_, [u8]>> = DecoderConfig::new().extract_xmp(&data)?;

// Thumbnail decode (smaller embedded preview image)
let thumb: Option<DecodeOutput> = DecoderConfig::new().decode_thumbnail(&data, PixelLayout::Rgb8)?;
```

### Key Types
- `DecoderConfig` — HOW to decode (reusable, Clone)
- `DecodeRequest<'a>` — WHAT to decode (data + layout + limits + stop)
- `DecodeOutput` — decoded pixels (data + width + height + layout)
- `PixelLayout` — Rgb8, Rgba8, Bgr8, Bgra8
- `Limits` — max_width, max_height, max_pixels, max_memory_bytes
- `ImageInfo` — probe result (width, height, has_alpha, bit_depth, chroma_format, has_exif, has_xmp, has_thumbnail)
- `enough::Stop` — cooperative cancellation (re-exported)

### Dependencies
- `enough` — cooperative cancellation (Stop trait)
- `whereat` — error location tracking (At<E> wrapper)
- `archmage` — SIMD dispatch via CPU feature tokens
- `safe_unaligned_simd` — safe wrappers over std::arch intrinsics

## Code Style

- Use `div_ceil()` instead of `(x + n - 1) / n`
- Use `is_multiple_of()` instead of `x % n == 0`
- Collapse nested `if` with `&&` when possible
- Use iterators with `.enumerate()` instead of manual counters

## Current Implementation Status

### Completed
- Zen codec API (DecoderConfig → DecodeRequest → decode)
- PixelLayout (Rgb8, Rgba8, Bgr8, Bgra8), Limits, Stop cancellation
- decode_into zero-copy, ImageInfo::from_bytes probing
- HEIF container parsing (boxes.rs, parser.rs)
- NAL unit parsing (bitstream.rs)
- VPS/SPS/PPS parsing (params.rs)
- Slice header parsing (slice.rs)
- CTU/CU quad-tree decoding structure (ctu.rs)
- Intra prediction modes with TU-level ordering (intra.rs)
- Reference sample filtering (H.265 8.4.4.2.3)
- Reference sample substitution with forward propagation (H.265 8.4.4.2.2)
- Transform matrices and inverse DCT/DST (transform.rs)
- Transform skip mode (H.265 8.6.4.1) — proper bypass of inverse transform
- CABAC tables and decoder framework (cabac.rs) — bit-exact with libde265
- Frame buffer with YCbCr→RGB conversion (picture.rs)
- Transform coefficient parsing via CABAC (residual.rs)
- Adaptive Golomb-Rice coefficient decoding
- DC coefficient inference for coded sub-blocks
- Sign data hiding (all 280 CTUs decode)
- Debug infrastructure (debug.rs) with CABAC tracker
- sig_coeff_flag proper H.265 context derivation
- Conformance window cropping (to_rgb/to_rgba apply SPS conf_win_offset)
- Deblocking filter (deblock.rs) — H.265 8.7.2, strong/weak luma + chroma, inter-aware bS with TB/PB edge distinction
- SAO filter (sao.rs) — H.265 8.7.3, band offset + edge offset
- Grid-based HEIC decoding (idat, iref/dimg, tile assembly)
- Alpha plane decoding from auxiliary images (auxl/auxC)
- HDR gain map extraction (Apple HDR aux format) — incl. the monochrome (4:0:0)
  pixel decode (fixed 2026-05-31; was ~95% UNINIT, see Known Bugs FIXED note)
- Identity-derived (iden) and overlay (iovl) image types
- Image mirror (imir) with ordered transform application (ipma order)
- VUI color info parsing (video_full_range_flag, matrix_coefficients, color_primaries, transfer_characteristics)
- YCbCr→RGB with BT.601, BT.709, BT.2020 matrices (full + limited range)
- colr nclx box color info override from HEIF container (all 4 CICP fields)
- CICP propagation through zencodec adapter (TF/primaries on PixelDescriptor, with_cicp on ImageInfo)
- ICC profile extraction (extract_icc API)
- RowSink streaming decode (decode_rows API, grid-to-sink streaming)
- HEVC scaling list support (custom dequantization matrices from SPS/PPS)
- `#![forbid(unsafe_code)]` — zero unsafe blocks in codebase
- `no_std + alloc` support (compiles for wasm32-unknown-unknown)
- Integer overflow protection for dimension calculations
- Memory estimation before decode (DecoderConfig::estimate_memory)
- Hardened parser: checked arithmetic, resource limits, fallible allocation, Stop cancellation
- Multi-extent item support (get_item_data returns Cow: borrow single, concat multi)
- Parser defensive validation (clap zero-denom, ispe bounds, hvcc length_size, string/NAL/ICC caps)
- cargo-fuzz targets: decode, decode_limits, probe
- whereat error location tracking (At<HeicError> Result type)
- EXIF extraction (zero-copy, strips 4-byte HEIF prefix, returns raw TIFF)
- XMP extraction (zero-copy, returns raw XML from mime items)
- ImageInfo::from_bytes grid/iden/iovl probing (reads ispe + first tile hvcC)
- Thumbnail decode support (thmb references, decode_thumbnail API)
- Zero compiler warnings (clippy clean, all doc comments present)
- Criterion benchmarks (57ms RGB, 1.3µs probe, 4.4µs EXIF, 4.2ms thumbnail)
- 10-bit HEVC support (u16 planes, transparent downconvert to 8-bit output)
- SIMD-accelerated color conversion via archmage (AVX2 with scalar fallback)
- SIMD-accelerated IDCT 8x8/16x16/32x32 via archmage AVX2 (madd_epi16 butterfly)
- SIMD-accelerated IDST 4x4 via archmage SSE4.1 (3.77x vs scalar)
- SIMD residual add (u16+i16→clamped u16) and dequantize via archmage AVX2
- Tile-parallel grid decoding via rayon (optional `parallel` feature)
- PCM mode support (H.265 7.3.8.8) — raw sample read + CABAC reinit
- Tile-aware CABAC context derivation (split_cu_flag, cu_skip_flag, SAO merge, intra MPM)
- Tile boundary QP prediction reset (H.265 8.6.1) and context/StatCoeff reinit

### Current Quality (RGB comparison vs libheif)
- 103/162 test files decode successfully
- Best: example_q95 65.7dB (98% pixel-exact), classic-car 77.3dB (BT.709)
- Nokia C001-C052: 50.5dB (77% pixel-exact)
- Grid images: image1 50.4dB, classic-car 77.3dB
- Scaling list files: iphone_rotated 55.3dB (91% exact), iphone_telephoto 50.9dB
- All CABAC SEs match libde265 perfectly
- YUV-level: pixel-perfect for q50+ (76.1dB for q10, 128 Y-plane diffs vs dec265)
- Color conversion: ×8192 fixed-point for limited-range, ×256 for full-range
- example.heic: 73.0% pixel-exact, SSIM2 91.86, avg diff 0.45, max diff 12

### Known Edge Cases
- MIAF003 (4:4:4 chroma, RExt profile): 61.9dB (97.8% exact, max diff 4)
- overlay_1000x680: 13.1dB — remaining diff from color conversion on fill regions
- example_q10: 36.1dB RGB — low-QP amplifies color conversion rounding

### Performance
- Release profile: thin LTO + codegen-units=1
- Criterion benchmarks: 54ms example.heic (1280x854), 451ms iPhone sequential (3024x4032), 180ms parallel
- Callgrind (iPhone, scalar under valgrind): 5090M instructions
- Key optimizations applied:
  - Plane-direct writes, in-place dequant, border fill inlining (731M→653M)
  - Partial butterfly IDCT for 8/16/32 (decode_and_apply_residual -14%)
  - SAO edge interior/border split + lazy plane cloning (SAO -26%)
  - Color conversion: 4:2:0 specialization, ×8192 fixed-point (to_rgb -38%)
  - Row-slice bounds-check elimination in intra prediction and residual add
  - SIMD color conversion via archmage AVX2 (81M → 9.2M, -88%)
  - SIMD IDCT 8x8/16x16/32x32 via archmage AVX2 (madd_epi16 butterfly)
  - SIMD IDST 4x4 via archmage SSE4.1 (14.1M → 3.7M, -73%)
  - SIMD residual add via archmage AVX2 (u16+i16→clamped u16)
  - SIMD dequantize via archmage AVX2 (flat scale only; scaled uses scalar)
  - Intra prediction: early-exit substitution, hoisted bounds, halved arrays
  - Deblocking: direct plane access with step_along/step_across
  - Residual buffer reuse across TU decode calls
  - Tile-parallel grid decode via rayon: 451ms → 180ms (2.5x) for 48-tile iPhone image
  - Streaming decode_into for grids: bypasses full-frame YCbCr, color-converts tiles directly to output
- Remaining hotspots: decode_and_apply_residual (32%), predict_intra (17%), CABAC (10%), memcpy/memset (7%)

## Known Limitations

- Inter prediction (P/B slices) on `inter-prediction` branch:
  - Full pipeline: syntax parsing, merge/AMVP/TMVP, scalar MC, DPB, VideoDecoder
  - Conformance: 48/48 vectors decode without crash, 1 pixel-exact (I-only)
  - CABAC verified BIT-EXACT vs dec265 (all 28 CTU byte positions match for MERGE_A)
  - MERGE_A: frames 1-7 small deblocking diffs (29-349 pixels, max_abs 2-6), frame 0 exact
  - SAO_B (3x1 tiles): 12.9dB (all frames decode, no UNINIT)
  - Fixed bugs:
    - Tile boundary CABAC context: split_cu_flag, cu_skip_flag, SAO merge left/up now check same-tile availability (matching libde265 6.4.1)
    - Tile QP reset: QP prediction, is_cu_qp_delta_coded, StatCoeff reset at tile boundaries
    - Intra MPM tile awareness: get_neighbor_intra_mode_left returns DC for cross-tile neighbors
    - PCM mode: pcm_flag decode via decode_terminate, raw sample read, CABAC reinit (H.265 7.3.8.8)
    - SAO merge slice check: uses actual slice_segment_address instead of hardcoded 0
    - interSplitFlag: missing forced TU split when max_transform_hierarchy_depth_inter==0 and PartMode!=2Nx2N (H.265 7.3.8.7). Caused CABAC desync in RQT_A B-frame.
    - Temporal MVP fallback: only tried one collocated position (bottom-right OR center), but H.265 8.5.3.2.8 requires trying bottom-right first, then falling back to center when collocated block is intra
    - Small PU L1 restriction: nPbW+nPbH==12 rule unconditionally disabled L1, but H.265 8.5.3.2.2 step 10 only disables L1 when both L0 and L1 are active (bi-prediction)
    - ref_idx decode: truncated unary consumed extra CABAC bin when num_active>=2 (CABAC desync)
    - AMVP: isScaledFlagLX, B-candidate always-run, same-list-first ordering
    - Temporal MVP: collocated MV selection (NoBackwardPredFlag, col_from_l0_flag), per-list derivation, collocated frame ref_poc stored in DPB
    - Earlier: CBF tracking for deblocking bS=2, DST/DCT for inter 4x4 TUs, chroma MC shifts (4→6), skip/no-residual CU boundary marking, WPP entry point offsets + CABAC reinit, cu_skip_flag cross-CTB-row context derivation, deblocking TB/PB edge distinction with separate bS derivation, bi-pred cross-list bS comparison, bi-pred H+V MC rounding offset
  - IMPORTANT: dec265 reference YUV is in DISPLAY ORDER (not decode order)
  - Deblock trace infrastructure: `enable_deblock_trace()` dumps all edge parameters to /tmp/our_deblock_trace.txt
  - Multi-slice: PictureMaps persistence across slices (ct_depth, intra modes, pred/mv, cbf, qp, sao)
  - Loop filters deferred to picture completion (prevents intra ref corruption in multi-slice)
  - Tile scan order: boundary detection, CABAC reinit at tile boundaries with entry point offsets
  - Remaining UNINIT vectors: TILES_A/B (complex tile CABAC desync within tiles), DELTAQP_A (PCM + complex content), DBLK_A/B (multi-slice inter desync), SDH_A (cu_qp_delta desync), MVDL1ZERO (scattered inter desync)
  - Deferred: SIMD MC (Phase 7), weighted prediction application
- 4:4:4 color conversion dispatches through `[v3, neon, scalar]` in
  `heic-core/src/color_convert.rs`. The NEON implementation uses 16-pixel
  interleaved stores; the earlier claim that ARM stays scalar is stale.
  On Apple M4 Pro, the 2026-09-06 paired audit favors NEON for 64² through
  4096² and ties at 17². These are limited-range BT.709 strided-input
  measurements, not claims about all machines or matrix/range combinations.
  Full results and exact-pixel checks:
  [ARM audit](benchmarks/arm_audit_2026-09-06/README.md).
- Dependent slice segments: not supported (2 vectors fail)
- **Cancellation reaches the primary-image decode only; the auxiliary decodes
  are uncancellable** (found 2026-08-29 while closing #22). `decode_gain_map`,
  `decode_thumbnail` and the depth-map path hardcode `&Unstoppable` at their
  `decode_item` calls (`src/decode.rs` ~2189, ~2359, ~2496) — these are full
  HEVC decodes of a real image, so a hostile or merely large auxiliary runs to
  completion no matter what the caller does. Unlike the grid-tile hole fixed in
  the same pass, this one **cannot** be fixed without a public API change:
  `decode_gain_map` / `decode_thumbnail` hang off `DecoderConfig`, not
  `DecodeRequest`, so there is no `with_stop` on the call at all. Same shape as
  the metadata/probe paths (`extract_exif` / `extract_xmp` / `extract_icc` /
  `ImageInfo::from_bytes`), which the 2026-05-31 audit already flagged as
  uncancellable via the `stsc` resolver. Fixing it means either moving these
  onto `DecodeRequest` or adding `*_with_stop` siblings — an additive but real
  API decision, so it needs the user's sign-off before anyone builds it.

## Known Bugs

### FIXED 2026-09-06: SIMD residual addition wrapped before clamping

`transform_simd::{add_residual_block_v3, add_residual_block_wasm128}` and
`transform_simd_neon::add_residual_block_neon` added signed 16-bit lanes with
wrapping arithmetic. Prediction 2 plus residual 32766 produced 0 instead of
255 in the 8-bit NEON path; unsigned predictions above 32767 were also
misinterpreted. Biasing the unsigned prediction into the signed domain,
saturating the addition, then unbiasing and applying an unsigned maximum
clamp preserves widened scalar arithmetic through 16-bit predictions.
The `_dev` residual-range regression checks every prediction value at five
bit depths, seven residual extremes, five block sizes and padded strides.


### OPEN 2026-08-29: `conformance::merge_a_deblock_debug` panics (not skips) without ffmpeg

`tests/conformance.rs` reaches for `ffprobe` in `get_dimensions()` /
`count_frame_types()`. On a box with no ffmpeg installed, `Command::output()`
fails, `.ok()?` yields `None`, and line 648's

```rust
let (width, height) = get_dimensions(&bitstream).unwrap();
```

panics. Its two siblings in the same function guard the same class of missing
input (`ensure_vector` → `SKIP`, `ensure_reference_yuv` → `SKIP: no ref`), so
the test reads as skip-capable and then hard-fails on the one dependency nobody
guarded. Observed on aarch64-macOS, 48/49 `conformance` tests passing.

**Not a decoder defect** and not visible in CI — the coverage job explicitly
excludes the `conformance.rs` target ("it downloads ITU vectors at runtime"), so
no CI leg runs it. The vector itself is fine (`conformance/vectors/MERGE_A_TI_3/
MERGE_A_TI_3/MERGE_A_TI_3.bit`, 16,374 B).

The fix is **not** to add another silent skip — per the workspace rule against
graceful skips, the ffmpeg requirement should be a caller-visible switch (env
var or feature threaded through justfile → test invocation), the way
`HEIC_SKIP_WIN_HOST_TESTS` already gates the WSL→Windows bridge. Until then,
install ffmpeg to run this target locally.

### FIXED 2026-08-27: 4:2:2 chroma decodes sample-exact (#48); RExt `chroma_qp_offset_list` still open

4:2:2 (`chroma_format_idc = 2`) fell through the 4:2:0 branch of `src/hevc/ctu.rs`
(one cbf_cb/cbf_cr per node instead of two, chroma TB at `y0/2` instead of `y0`, no
Table 8-3 remap) → CABAC desync; 10-bit streams returned Ok with chroma ~17x out of
range. Fixed: `ChromaCbf` two-bit cbf pairs, stacked chroma TBs (cb0, cb1, cr0, cr1),
`CHROMA_422_MODE_MAP` (copied from libde265, which passes the official md5s), and
`QpC = Min(qPi, 51)` for ChromaArrayType != 1 at the `cu_qp_delta` re-derivation and
in `deblock.rs` (the slice-init site was already right). Layers passed: x265
`--lossless` 8/10-bit raw 4:2:2 == source (bit-exact); heif-enc `chroma=422` HEICs ==
dec265 (sample-exact, q=35 exercises the high-QP chroma table + deblock);
`ADJUST_IPRED_ANGLE_A_RExt_Mitsubishi_2` (1080p 10-bit, all 35 chroma modes × all
sizes) and `GENERAL_10b_422_RExt_Sony_1` seq 0 == dec265. Tests:
`tests/chroma422.rs` (every fix mutation-verified); fixtures from
`scripts/gen_422_fixtures.py`; tool: `examples/annexb_vs_yuv.rs`.

STILL OPEN (not 4:2:2-specific): `Main_422_10_A/B_RExt_Sony_2` and GENERAL seq 1 fail
with "cu_qp_delta out of range" because they set the PPS range-extension
`chroma_qp_offset_list_enabled_flag` (`cu_chroma_qp_offset_flag/idx` bins in
`transform_unit`, `cu_chroma_qp_offset_enabled_flag` in the slice header,
`Log2MinCuChromaQpOffsetSize` reset in `coding_quadtree`, `CuQpOffsetCb/Cr` in 8.6.1).
`params.rs` does not parse `pps_range_extension` at all. RExt vectors live under
`https://www.itu.int/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/RExt/`
(not covered by `conformance/fetch.sh`); dec265 built from
`https://github.com/strukturag/libde265` reproduces their md5s and is the reference.

### FIXED 2026-06-01: Lossless 4:4:4 transquant-bypass now decodes (bit-exact vs libheif)

`testdata/features/lossless.heic` (10-bit 4:4:4 `cu_transquant_bypass` intra-NxN,
matrix_coefficients=0 GBR) decodes bit-exact vs libheif. Found by diffing our
SE# trace (set `SE_TRACE_LIMIT` in ctu.rs > 0) against dec265's until 304/304
matched. Five fixes, in `src/hevc/ctu.rs` unless noted:
1. **4:4:4 cbf carve-out** — `cbf_cb`/`cbf_cr` read at the 4x4 leaf when
   `ChromaArrayType==3` (`log2_size > 2 || chroma_array_type == 3`).
2. **4:4:4 NxN chroma modes** — one `intra_chroma_pred_mode` per luma PB (4 for
   NxN) when `ChromaArrayType==3`, stored per-quadrant.
3. **transquant_bypass transform tree** — the intra residual tree is parsed even
   under bypass (bypass skips reconstruction, not the residual_coding syntax).
4. **chroma scan order (the final desync)** — the leaf looks up the per-quadrant
   chroma mode via `get_intra_chroma_mode_at`, not the single threaded mode;
   Angular 10/26 select vertical/horizontal scan, so the wrong mode picked
   diagonal and desynced CABAC inside the residual_coding.
5. **bypass reconstruction** — residual = parsed coeff levels directly (no
   dequant/transform); implicit RDPCM gated on the SPS `implicit_rdpcm` flag.
Supporting: `params.rs` now parses the full VUI tail + `hrd_parameters` +
`sps_range_extension` (the 9 RExt flags were never read; all false here, no-op
for non-RExt streams); `heic-core/frame.rs` handles `matrix_coefficients==0`
(R=Cr, G=Y, B=Cb) in both `to_rgb` and `ycbcr_to_rgb`. Pinned by
`tests/cov_features.rs::lossless_decodes_gradient`. No regression: example.heic,
nokia_444 (4:4:4 MAE 1.4), corpus, and 179 lib/integration tests unchanged.

### Negative-offset overlay: we are MORE spec-compliant than libheif (not a bug)

For an `iovl` tile placed at a negative offset (e.g. `testdata/features/iovl_negoff.heic`,
offset (-16,-16)), our decoder composites the tile's clipped on-canvas region per
ISO 23008-12 §6.6.2.3, and `tests/cov_features.rs::iovl_negative_offset_clips_tile`
asserts the spec-derived result. libheif 1.21.2 instead renders such overlays as
all-fill (skips the tile) and **fails Nokia conformance C021 entirely** (offset
(-640,-360), `heif-dec` rc=1). So the golden here is spec-derived, not
libheif-derived — libheif is the incomplete reference for negative overlays.

### FIXED 2026-05-31: Apple HDR gain map decoded ~95% UNINIT (monochrome CABAC desync)

The Apple HDR gain map (`apple-hdr/hdr-sample.heic`, a 768×432 **monochrome /
4:0:0** HEVC image) decoded ~95% UNINIT garbage (downconverting to 255) because
the 4:0:0 path read chroma syntax that isn't in the bitstream — each phantom
CABAC bin desynced the decode after a few CTBs. H.265 gates those syntax
elements on `ChromaArrayType != 0`; the decoder didn't. Fixed two sites in
`src/hevc/ctu.rs`:
- `cbf_cb`/`cbf_cr` (transform_tree, H.265 7.3.8.8): short-circuit to
  `(false,false)` when `ChromaArrayType == 0`.
- `intra_chroma_pred_mode` (`decode_intra_chroma_mode`, H.265 7.3.8.5): return
  the luma mode without reading when `ChromaArrayType == 0`.
Result: the gain map now decodes 0/331776 UNINIT. This unblocked the deferred
decode-completeness guard (re-added in `decode_nal_units`: error if any luma
sample stays UNINIT — verified safe against the full corpus). A stronger
`decode_apple_hdr_gain_map` test now asserts the 255-saturation fraction < 50%.
This was the first HEIC test asset with a monochrome stream, so the 4:0:0 path
had been unexercised. KNOWN GAP: H.265's `ChromaArrayType == 3` carve-out (4:4:4
reads cbf at the smallest TU) is still not implemented — 4:4:4 decodes at
61.9dB so it's masked/rare; fix if a 4:4:4 regression appears.

### Deep safety+code review — 2026-05-31 (43-agent workflow, adversarially verified)

Full fan-out review of all crates + native backends + CI. 81 findings, 29
verified-real, 2 refuted. Confirmed bugs below (file:line traced + an
independent verifier re-derived each from source). Pure-Rust items are
untrusted-input safety bugs reachable from a crafted .heic; native-backend
items are FFI/correctness bugs not runtime-verifiable on this Linux host.
Findings also logged to `~/work/feedback/heic.md`.

**FIX STATUS (2026-05-31) — all 29 confirmed findings addressed except 2 deferred:**
- ✅ FIXED + pushed to main: HEVC param/slice overflows (95d6eab); 10-bit
  limited-range color + NAL 32-bit wrap + crop saturating_sub (5be0203, 5f23346);
  stsc DoS + clap crop + iovl offset (4bf4f5e); zencodec grid limits (be7533e);
  VA-API ABI field order + offset asserts + unpack OOB + probe clamps (274db91);
  D3D11VA u32-trunc OOB (c840a02); native-FFI batch — MF chroma/leak/linear-unpack,
  VT chroma/crop/null, MediaCodec chroma/NV21/crop, D3D11VA multi-slice/0xFF-refpic
  (19a998c). Verification: pure-Rust by tests (heic-core color regression, corpus
  gate, overflow-checked corpus); native by cross-compile (windows-gnu / android /
  apple) — NOT runtime-verified (no Win/Android/Apple/GPU HW here). Runtime
  validation lands with the GPU/emulator CI (docs/CI.md).
- ⏳ DEFERRED (measure-first, not skipped):
  1. **UNINIT-sentinel leak** (cabac.rs:266 + frame.rs:444) — error-vs-clamp choice
     needs corpus-regression measurement (erroring on truncated streams could
     regress currently-passing corpus files). Sequence behind the conformance/corpus
     CI gate. NOTE: the color fix's `y_val.clamp(0, max_val)` already stops the raw
     0xFFFF leak in the 16-bit path.
  2. **dequant i32 overflow** (transform.rs:689,704; transform_simd.rs:1435,1449)
     for high-QP 10/12/16-bit — a correct fix widens the SIMD dequant kernel to i64;
     needs zenbench to confirm no 8/10-bit hot-path regression before landing. 8-bit
     (common case) does NOT overflow. Sequence behind a zenbench i64-vs-i32 dequant
     measurement.

**CI ADDED (2026-05-31):** corpus-decode gate over 95 bundled files (all 6 OSes +
i686 32-bit); overflow-checks job; semver/doc/av1/unci gates; 7 fuzz targets; MF
runtime fail-loud; MediaCodec emulator gate; VA-API/D3D11VA self-hosted GPU gates +
`examples/backend_decode`. Strategy + bounded-cost GPU plan: `docs/CI.md`.

**HIGH — pure-Rust untrusted-input (fixable + verifiable on Linux):**
- `src/codec.rs:1027,1112` — zencodec streaming grid decoder skips dim/memory
  limits when `Limits` not provided (parent's `try_decode_grid_streaming`
  uses `unwrap_or(&NO_LIMITS)`; the adapter does not). Crafted 32-bit-dims
  grid → uncapped alloc (64-bit OOM) / usize-overflow undersized buffer +
  OOB-index panic (wasm32). `strip_bytes` mul is unchecked. Fix: mirror
  `NO_LIMITS` default + `checked_mul`.
- `heic-core/src/frame.rs:1098-1122` — 10/12-bit **limited-range** YCbCr→RGB is
  ~4× too dark. Y-scale const `9576` is the 8-bit fixed-point scale but `shift`
  grows with bit depth without rescaling the coefficient. 10-bit white
  Y=940→RGB 256 not 1023. Corrupts every 10-bit limited-range image (iPhone
  HDR/BT.2020) through the 16-bit path. 8-bit + full-range branches correct,
  so 8-bit RGB tests never caught it. Fix: scale `9576` by `2^(bit_depth-8)`.
- `src/hevc/params.rs:1070-1072` — scaling-list DC coef `dc_coef_minus8 + 8`
  uses plain `+`; `read_se()` can return i32::MAX → overflow panic (debug /
  cargo-fuzz) or wrapped garbage DC (release wrong-pixels). Adjacent delta
  path already hardened with `wrapping_add`. Fix: reject per H.265 range
  [-7,247] or `wrapping_add`+`rem_euclid`.
- `src/hevc/params.rs:629-646` — SPS validates min/diff TB-size fields
  individually but not derived `log2_max_tb_size`. Craft min_tb=5 + diff=3 →
  log2_max_tb=8; an **inter** CU leaf at log2_size=6 bypasses intra's
  log2>5 guard → `coeffs[..4096]` slices the fixed `[i16;1024]` CoeffBuffer
  (`ctu.rs:2253/2258`) → OOB panic. Fix: reject `log2_max_tb_size > 5`.
- `src/decode.rs:2149-2188` + `src/hevc/transforms.rs` + `heic-core/src/frame.rs:416-418`
  — crafted `clap` clean-aperture inflates crop offsets unboundedly; a later
  `irot` swaps a huge offset into crop_right/bottom; `width - crop_right`
  u32 subtraction underflows → panic (debug) / OOB (release). Fix: clamp
  crop ≤ dims after clap+each transform AND `saturating_sub` the ~10
  `to_rgb*`/`to_bgr*` y_end/x_end sites. (verifier downgraded high→medium)
- `src/heif/parser.rs:2484-2491` — `resolve_sample_offset` inner loop upper
  bound `next_first_chunk` (attacker stsc field, up to u32::MAX) only
  validated `>= first_chunk`, not `<= num_chunks`. With `samples_per_chunk==0`
  the match never fires → up to ~4.3e9 useless iterations (measured 6.7s at
  5e8). Probe paths (`ImageInfo::from_bytes`, `extract_exif/xmp/icc`) pass
  `Unstoppable` so it's **uncancellable**. ~650-byte msf1 file. Fix: clamp
  `next_first_chunk.min(num_chunks+1)` + reject `samples_per_chunk==0`.

**HIGH — native FFI (fix by code-reading; runtime-verify needs the target OS):**
- `heic-backend-d3d11va/src/decoder.rs:508-520` — `(data.len() as u32) > buf_size`
  guard truncates on 64-bit; a >4 GiB bitstream whose low 32 bits < buf_size
  passes, then `copy_nonoverlapping(..,data.len())` overflows the GPU buffer
  (OOB write/UB). Same file uses safe `u32::try_from(..).unwrap_or(MAX)`
  elsewhere (lines 409,443). Fix: compare in usize before the cast.
- `heic-backend-vaapi/src/va_hevc.rs:84-134` — `VaPictureParameterBufferHevc`
  field order diverges from libva ABI: `slice_parsing_fields` placed right
  after `pic_fields` but libva puts it AFTER `row_height_minus1[21]`. Struct
  is `bytes_of`-memcpy'd into the driver buffer → every field from
  `sps_max_dec_pic_buffering_minus1` on is read 4 bytes off → garbage decode
  + plausible driver-side tile OOB. Total size still 604 so no size error;
  unit tests check field *values* not *offsets* (CLAUDE.md §1 anti-pattern).
  Confirmed via offset_of harness. Fix: reorder to match va_dec_hevc.h +
  add `const offset_of` layout asserts.
- `heic-backend-vaapi/src/decode.rs:593-672` — `unpack_planes` walks the mapped
  VAImage with loop bounds `config.width/height` + crop from `ispe`/SPS-conf
  (attacker, independent boxes) but the surface is sized from SPS coded dims.
  No check that `cy+h<=image.height` / offsets `<=data_size` → OOB read/UB
  when ispe > coded. Fix: validate against VAImage geometry, clamp.
- `heic-backend-mediafoundation/src/pixels.rs:57-74` — linear (non-IMF2DBuffer)
  fallback unpack does ZERO buffer-length validation (cur_len/max_len read but
  unused) before `from_raw_parts` over `coded_w*coded_h*3/2` → OOB if the MFT
  buffer is shorter. The 2D path guards this (pixels.rs:118-129); linear
  doesn't. (verifier: high→medium) Fix: share the 2D bounds check.

**Systemic native-backend bug — chroma mislabel (3 backends, all confirmed):**
- MF `imp.rs:408`, VT `imp.rs:594-663`, MediaCodec `imp.rs:457-463` — each
  hardcodes a 4:2:0 output buffer but tags `DecodedFrame.chroma_format` with
  the *source* `chroma_format_idc`. For 4:2:2/4:4:4 sources the parent
  (`src/decode.rs:634-664`, `frame.c_stride()`) reads chroma with the wrong
  stride/row-count → silent wrong pixels (guards turn would-be OOB into
  partial garbage reads). Fix: set `chroma_format=1` after decode, or reject
  non-4:2:0.
- MediaCodec `imp.rs:487-503` also unconditionally unpacks
  COLOR_FormatYUV420Flexible as NV12 → Cb/Cr swap (red/blue) on NV21/planar
  devices.

**MEDIUM (confirmed, abbreviated — see ~/work/feedback/heic.md for full):**
- `src/hevc/slice.rs:369-374` — long-term RPS `num_long_term_sps +
  num_long_term_pics` u32 sum overflow (two unbounded ue loops).
- `heic-core/src/nal.rs:40-44` — `hvcc_to_annexb` length-prefix bound check
  wraps on 32-bit → slice-index panic.
- `src/hevc/params.rs:883-895` — PPS num_tile_columns/rows_minus1 `as u16`
  truncation, no spec-bound validation; then `src/backend.rs:528-529` clamps
  to 255 (`unwrap_or(u8::MAX)`) into FFI pic-params → tile desync / OOB tile
  index in native backends.
- `src/decode.rs:609-631` — iovl negative per-image offsets clamped to 0
  instead of clipping off-canvas → wrong pixels (overlay_1000x680 13.1dB).
- `src/hevc/transform.rs:704` / `transform_simd.rs:1449` — i32 overflow in flat
  dequantize scalar path for 12/16-bit at high QP (partial).
- `src/hevc/cabac.rs:266-273` + `heic-core/src/frame.rs:444-447,718-723` —
  CABAC EOF tolerance lets a truncated bitstream "succeed" with
  `UNINIT_SAMPLE`(u16::MAX) CTBs; the sentinel passes through to_rgb/to_rgba
  (only final RGB clamp) → garbage pixels instead of a decode error.
- VT `imp.rs:583-667` — conformance-window crop / coded size ignored →
  spatially shifted pixels (top/left crop).
- MF `imp.rs:509-560` — pre-allocated IMFSample never Released on
  STREAM_CHANGE / NEED_MORE_INPUT / error retry → COM ref leak.
- MediaCodec `imp.rs:362-503` — crafted SPS-conf crop + mismatched ispe drives
  unpack index past the validated output-buffer size → panic.
- D3D11VA `imp.rs:130-165`+`decoder.rs:386-424` — single slice-control entry
  emitted for multi-slice/multi-NAL bitstream → wrong pixels.

**Refuted (NOT bugs — recorded so they're not re-investigated):**
- `heic-core/src/frame.rs:413` degenerate bit_depth<8 panic — native backends
  never produce bit_depth<8; not reachable from untrusted input.
- `src/decode.rs:1303,1948,2111,2404` unguarded luma/alpha/gainmap blits —
  dimensions validated upstream; the native-backend plane-size invariant holds.

### D3D11VA real-decode: example.heic midgray (not scaling lists; 2026-05-22)

Status: ⚠ Partial — 5 of 6 bundled corpus files (now including apple-hdr
near-bit-exact at max_delta=1) + 9/9 iPhone HDR fixtures decode on the
RTX 5070 via D3D11VA. One known failure:

- `tests/testdata/libheif-examples/example.heic` (1280×854, grid of
  6× 512×512 tiles, BT.709 limited): all 6 tiles return Y=128 /
  Cb=128 / Cr=128 midgray.

Root cause: NOT scaling lists. After adding scaling-list propagation
through `ParsedSps`/`ParsedPps` + `qmatrix_from_parsed()`,
apple-hdr/hdr-sample.heic dropped from max_delta=49 to max_delta=1
(near bit-exact). example.heic remains stuck at max_delta=255
midgray — i.e. the GPU isn't writing decoded pixels at all,
unrelated to scaling-list math.

Triage update (after commit `b16dcd28`): individual 512×512 tile
decodes for example.heic produce REAL CONTENT (debug dump:
y0=[180, 180, 180, ...] for tile 1, y0=[94, 93, ...] for tile 2,
y0=[153, 153, ...] for tile 3 — clearly varied photo content). So
the GRID-TILE decode path through D3D11VA is fully working.

The corpus_diff failure (max_delta=255 against rust output) comes
from a DIFFERENT decode call the parent makes for example.heic:
debug shows a 1280×854 "full image" decode with spsW=1280
spsH=856, nals=2, first_nal_type=39 (SEI prefix), and Y plane
showing y0=[16, 16, 16, ...] (BT.709 limited-range black).

That suggests example.heic contains BOTH a grid layout (6× 512×512
tiles) AND a separate single-image item (1280×856 coded → 1280×854
visible). The parent picks the single-image path for the actual
decode, and D3D11VA produces all-black on that specific bitstream.

Next session: hex-dump the FULL-IMAGE pic_params for the
example.heic 1280×854 decode (not the tile decodes) and find why
this bitstream specifically goes black. Different fix path than
the scaling-list one.

Reproduce: `HEIC_D3D11VA_HW=1 HEIC_D3D11VA_DEBUG=1 cargo test ...
d3d11va_vs_rust_corpus_diff` to see the full corpus output;
`d3d11va_vs_rust_synthetic_corpus` is the bit-exact subset gate.

## Investigation Notes

### UNINIT pixel triage (conformance vectors with negative PSNR)

10 vectors have UNINIT pixels (max_diff=65535 = UNINIT_SAMPLE sentinel).

**Vectors analyzed:**
- TILES_A/B: tiles_enabled=1 (5x5 non-uniform tiles), I-slices only. Tile CABAC reinit implemented.
- DELTAQP_A: cu_qp_delta_enabled=1, multi-slice (5 slices per I-frame). All frames UNINIT including frame 0 (I-frame).
- SDH_A: single-slice I+B. I-frame has 34% UNINIT, B-frame has 99% UNINIT.
- SAO_B: tiles=1 (3x1), B-slices
- RQT_A: FIXED — was missing interSplitFlag (H.265 7.3.8.7). Now 23.9dB (CABAC bit-exact, remaining diffs from deblock/SAO).
- CONFWIN_A: single-slice P/B. I-frame ~12dB (no UNINIT), later P/B frames get UNINIT.
- DBLK_A/B: multi-slice (4 slices per frame). ~12dB all frames, some have UNINIT.
- MVDL1ZERO_A: multi-slice, 500-frame sequence. Most frames ~12dB, 3 have UNINIT.

**Root cause analysis:**
- The dec265 CTU-CK checksum trace is ONLY printed for non-I slices (gated by `slice_type != I`), so earlier byte-position comparisons were between different frame types.
- MERGE_A (pixel-exact) has `cu_qp_delta_enabled=0`, `transform_skip_enabled=1`, QP=32.
- Failing I-frames (DELTAQP_A, SDH_A) have `cu_qp_delta_enabled=1` or different QP.
- The CABAC init formula and context tables match libde265 for the contexts checked (split_cu_flag, SAO, CBF).
- The CABAC engine init (range=510, value from first 2 bytes) matches libde265.
- Data offset computation verified correct for DELTAQP_A IDR slice (1 byte header → data_offset=1).
- First bin (split_cu_flag) verified manually: state=0,mps=0 at QP=26, LPS range=240, MPS threshold=34560, value=28334 → MPS (no split). This matches expected behavior for a flat-content first CTB.
- The UNINIT pixels are genuine (u16::MAX in Y plane), meaning some CTBs' pixels are never written. The decoder doesn't error out.

**Hypothesis:** The UNINIT appears to be from B/P frames referencing corrupted reference frames, not from I-frame decode failures. The I-frame first CTU decodes correctly (verified via SE trace). Need to compare more CTUs to find where decode diverges.

**Infrastructure built:**
- `PictureMaps` for cross-slice map persistence (ctu.rs)
- Deferred loop filter application in `finish_current_picture` (mod.rs)
- Tile boundary detection with CABAC reinit (ctu.rs)
- `compute_tile_boundaries`, `build_tile_scan_order`, `get_tile_id` helpers

## Module Structure

```
src/
├── lib.rs           # Public API types (DecoderConfig, DecodeRequest, Limits, etc.)
├── decode.rs        # Internal decode pipeline (grid, overlay, alpha, gain map, metadata)
├── error.rs         # Error types
├── auxiliary.rs     # Auxiliary image handling
├── codec.rs         # zencodec integration adapter
├── zennode_defs.rs  # zennode decode node definitions
├── heif/
│   ├── mod.rs
│   ├── boxes.rs     # ISOBMFF box definitions
│   └── parser.rs    # Container parsing
└── hevc/
    ├── mod.rs       # Main decode entry point
    ├── bitstream.rs # NAL unit parsing, BitstreamReader
    ├── params.rs    # VPS, SPS, PPS
    ├── slice.rs     # Slice header parsing (I/P/B)
    ├── ctu.rs       # CTU/CU decoding, SliceContext (intra + inter syntax)
    ├── intra.rs     # Intra prediction (35 modes)
    ├── inter.rs     # Inter prediction types, merge/AMVP candidate derivation
    ├── mc.rs        # Motion compensation (quarter-pel luma, eighth-pel chroma)
    ├── refpic.rs    # Reference picture set parsing, POC derivation, list construction
    ├── dpb.rs       # Decoded picture buffer management
    ├── cabac.rs     # CABAC decoder, context tables
    ├── residual.rs  # Transform coefficient parsing
    ├── transform.rs # Inverse DCT/DST (scalar + incant! dispatch)
    ├── transform_simd.rs # SIMD transforms: IDST 4x4, IDCT 8/16/32, residual add, dequantize
    ├── transform_simd_neon.rs # NEON SIMD transforms
    ├── color_convert.rs # YCbCr→RGB SIMD color conversion
    ├── color_convert_neon.rs # NEON color conversion
    ├── deblock.rs   # Deblocking filter (H.265 8.7.2, inter-aware bS)
    ├── sao.rs       # Sample Adaptive Offset (H.265 8.7.3)
    ├── debug.rs     # CABAC tracker, invariant checks
    ├── picture.rs   # Frame buffer, YCbCr→RGB conversion, deblock metadata
    └── transforms.rs # Spatial transforms: rotation (90/180/270) and mirror (H/V)
```

## FEEDBACK.md

See `/home/lilith/.claude/CLAUDE.md` for global instructions including feedback logging.

## aarch64 / NEON: decode is NOT SIMD-bound (measured 2026-07-28)

`benches/tier_isolation.rs` (`--features "_dev,backend-rust"`) compares whole
decode with the NEON token enabled vs forced off:

| image  | neon      | scalar    | delta  |
|--------|-----------|-----------|--------|
| image1 | 449.76 ms | 448.48 ms | -0.3%  |
| image2 | 331.64 ms | 337.20 ms | +1.7%  |
| C012   |  23.70 ms |  23.79 ms | +0.4%  |

Those three historical comparisons measured less than 2% change from toggling
the existing dispatch paths. They do not establish a ceiling for future work:
NEON IDCT16 and IDCT32 still call scalar implementations in
`src/hevc/transform_simd_neon.rs`. The 2026-09-06 Apple M4 Pro audit also found
no reliable whole-decode gain on its three fixtures; see
[the native audit](benchmarks/arm_audit_2026-09-06/README.md). Profile the actual
hot paths before choosing an optimization; tier timing alone does not identify
the source of decode cost.

Two caveats on reading that table:
- NEON is BASELINE on aarch64, so the "scalar" arm is still autovectorized by
  LLVM. A ~1.00x result means "hand-written NEON is no better than what LLVM
  emits", NOT "SIMD is unused".
- These are end-to-end numbers; they cannot show an individual kernel that is
  slower than its own fallback. If per-kernel work is ever wanted here, a
  per-kernel bench is needed first (see zenresize's `benches/kernel_tiers.rs`,
  where exactly that gap was hiding a kernel running at 0.94x).
