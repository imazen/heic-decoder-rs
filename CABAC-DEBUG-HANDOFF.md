# CABAC Debug Handoff Document

**Created:** 2026-01-22
**Sessions:** ~12 debugging sessions over 2 days
**Status:** CABAC primitives verified correct, but coefficient decoding produces wrong values
**SSIM2 Score:** -1082 (catastrophic - target is >70)

## TL;DR

The HEVC CABAC decoder's primitive operations (bypass decode, context-coded bins, coeff_abs_level_remaining) are **verified identical** to libde265. Yet the full coefficient decoding produces 24 large coefficients (>500) that corrupt the image. The bug is in **how we orchestrate these primitives** - context selection, scan order, or state machine logic.

## The Problem

```
Input:  example.heic (1280x854, 718 KB)
Output: SSIM2 = -1082, avg pixel diff = 87, max diff = 255
        Only 32.9% of pixels within 50 of reference
```

First corruption appears at **byte 124** (call #23), producing coefficient value **723** instead of a small value. This cascades through intra prediction, corrupting the entire image.

## What We Know Works (Verified Against libde265)

1. **CABAC bypass decode** - bit-for-bit identical (hevc-compare test passes)
2. **CABAC bypass bits** - identical for all num_bits values
3. **coeff_abs_level_remaining** - Golomb-Rice decode identical for all rice_param values
4. **Context initialization** - state/mps values match for all init_values and QPs
5. **Context-coded bin decode** - single bin decode matches perfectly
6. **Mixed context/bypass** - interleaved decoding matches

See `crates/hevc-compare/` for the comparison infrastructure.

## What We Know Is Broken

1. **Full coefficient decoding** produces wrong values
2. **First large coefficient at call #23, byte 124** - value 723 should be small
3. **24 total large coefficients (>500)** scattered throughout decode
4. **Using simplified contexts** - some proper H.265 derivations cause worse desync

## Debugging History

### Session 1-3: Sign Data Hiding
- Fixed DC coefficient inference for coded sub-blocks
- Fixed sig_coeff_flag decoding for position 15
- Fixed sign decoding order (high scan pos to low)
- Progress: CTU 49 → 161 → 272 → 269 → 280 (all decode)

### Session 4-6: Context Derivation
- Fixed sig_coeff_flag with H.265 9.3.4.2.5 derivation
- Fixed prev_csbf bit ordering (bit0=below, bit1=right)
- Fixed greater1_flag: ctxSet*4 + greater1Ctx formula
- Fixed greater2_flag: uses ctxSet not just chroma offset

### Session 7-9: Tracing Large Coefficients
- Added CabacTracker for debugging
- Traced first large coeff to call #23 at byte 124
- Found CABAC state at boundary: value=34552, scaled_range=34560
- Pattern: value close to scaled_range causes bypass to return many 1s

### Session 10-12: Context Simplification Investigation
- Tried proper coded_sub_block_flag neighbor context - **made things worse**
- Tried proper last_significant_coeff_prefix context - **causes early termination**
- Conclusion: Simpler contexts mask some bug; proper contexts expose it

## Key Files

```
src/hevc/
├── cabac.rs      # CABAC primitives (verified correct)
├── residual.rs   # Coefficient decoding (BUG IS HERE)
├── ctu.rs        # CTU/CU decode orchestration
├── intra.rs      # Intra prediction (works if given correct residuals)
└── debug.rs      # CabacTracker for debugging

crates/hevc-compare/
├── src/lib.rs    # Rust/C++ comparison infrastructure
└── cpp/          # Extracted libde265 CABAC functions
```

## The Smoking Gun

At call #23, decoding a 4x4 TU's coefficients:
- CABAC state before: range=some, value=some
- After decoding a few coefficients, state becomes: value ≈ scaled_range
- Next bypass decode: value*2 >= scaled_range → returns 1
- This continues, producing prefix=12+ in Golomb-Rice
- Result: coefficient value 723 instead of ~0-10

## What Has NOT Been Tried

### 1. Byte-for-Byte CABAC State Comparison
**Approach:** Run libde265 and our decoder in parallel, comparing CABAC state after every single operation.

**Implementation:**
```rust
// In residual.rs, after EVERY cabac call:
let (our_range, our_value, our_bits) = cabac.get_state();
let (ref_range, ref_value, ref_bits) = reference_cabac.get_state();
assert_eq!((our_range, our_value, our_bits), (ref_range, ref_value, ref_bits),
    "Divergence at operation #{} in TU at ({},{})", op_count, tu_x, tu_y);
```

**Why not done:** Requires instrumenting libde265 to expose state after every operation.

### 2. Fuzzing with Minimized Test Cases
**Approach:** Use cargo-fuzz to find minimal bitstreams that trigger divergence.

**Implementation:**
```rust
#[fuzz_target]
fn fuzz_coefficient_decode(data: &[u8]) {
    if data.len() < 10 { return; }
    let our_result = our_decode_residual(data);
    let ref_result = ref_decode_residual(data);
    assert_eq!(our_result, ref_result);
}
```

**Why not done:** Time investment to set up, may find edge cases not present in real files.

### 3. Differential Execution with libde265
**Approach:** Link libde265 directly and call its decode_residual_block for comparison.

**Implementation:**
```c
// Extract from libde265 slice.cc:
int16_t coefficients[32*32];
decode_residual_block(ctx, log2Size, cIdx, xT, yT, coefficients);
```

Compare coefficient arrays after each TU decode.

**Why not done:** Requires significant C++ FFI work.

### 4. Record/Replay CABAC Operations
**Approach:** Record every CABAC operation from libde265, replay in our decoder.

**Implementation:**
```rust
enum CabacOp {
    Bypass { result: u32 },
    BypassBits { num: u8, result: u32 },
    ContextBin { ctx_idx: u32, result: u32 },
    CoeffAbsLevelRemaining { rice: u8, result: i32 },
}

// Record from libde265, then replay:
for op in recorded_ops {
    match op {
        CabacOp::Bypass { result } => {
            let our_result = cabac.decode_bypass();
            assert_eq!(our_result, result, "Bypass mismatch at op #{}", i);
        }
        // ...
    }
}
```

**Why not done:** Requires instrumenting libde265 to emit trace.

### 5. Binary Search for Divergence Point
**Approach:** Decode N TUs, compare, bisect to find exact divergence point.

**Implementation:**
```rust
fn find_divergence(our_decoder: &mut Decoder, ref_decoder: &mut RefDecoder) -> usize {
    let mut lo = 0;
    let mut hi = total_tus;
    while lo < hi {
        let mid = (lo + hi) / 2;
        reset_both(our_decoder, ref_decoder);
        decode_n_tus(our_decoder, mid);
        decode_n_tus(ref_decoder, mid);
        if states_match(our_decoder, ref_decoder) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
```

**Why not done:** Requires ability to reset/snapshot decoder state.

## Proposed Novel Debugging Approaches

### A. Symbolic Execution of CABAC
Use a symbolic execution engine (e.g., haybale for LLVM, or manual implementation) to explore all paths through coefficient decoding. Find constraints that lead to large coefficients.

### B. Mutation Testing
Systematically mutate each context selection, scan order lookup, and state variable. For each mutation, run the decoder and check if output improves. This could identify which specific line is wrong.

### C. Formal Verification
Encode the CABAC state machine in a theorem prover (e.g., Z3, Coq) and prove equivalence with the spec. This would catch subtle off-by-one errors.

### D. Visual Debugging with Frame Diff
Generate per-CTU images showing:
1. Reference decoder output
2. Our decoder output
3. Coefficient difference map
4. CABAC state divergence map

This could reveal spatial patterns in the corruption.

### E. Canary Values
Insert known "canary" patterns in the coefficient buffer before decode. After decode, check which canaries were overwritten and which weren't. This could reveal if we're writing to wrong positions.

### F. Statistical Analysis
For each CABAC call type (bypass, context, remaining), collect:
- Frequency of each result value
- Distribution compared to reference

A statistical anomaly (e.g., too many 1s from bypass) would point to the bug.

### G. Execution Trace Diffing
Generate detailed execution traces from both decoders:
```
[our] bypass_decode -> 1 (state: range=356, value=45440)
[ref] bypass_decode -> 0 (state: range=356, value=45312)
                          ^^^^^ DIVERGENCE
```

The first divergence line is the bug location.

### H. Coefficient Value Histogram
For each scan position (0-15 in 4x4), collect histogram of decoded coefficient values. Compare against reference. Anomalies at specific positions indicate scan order bugs.

### I. Context State Dumping
After initializing all contexts, dump the full context table (state, mps for each context index). Compare with libde265. Any mismatch is a bug.

### J. Inverse Engineering
Instead of forward debugging:
1. Take the reference output coefficients
2. Run them through our inverse transform (known good)
3. Compare residuals
4. If residuals match reference, the bug is in coefficient decode
5. If residuals don't match, the bug might be in transform

## Recommended Next Steps

### Step 1: Instrument libde265 for Trace Export
Add logging to libde265 that emits:
```
CABAC_OP: type=bypass result=1 state=(510,12345,-8)
CABAC_OP: type=context ctx=42 result=0 state=(450,9876,-7)
...
COEFF: tu=(0,0) size=4 c_idx=0 values=[1,-3,0,2,...]
```

This creates a ground truth trace.

### Step 2: Record-Replay Test
Read the libde265 trace and replay in our decoder:
```rust
for (i, op) in trace.iter().enumerate() {
    let our_result = match op.op_type {
        "bypass" => cabac.decode_bypass(),
        "context" => cabac.decode_bin(op.ctx_idx),
        ...
    };
    if our_result != op.result {
        panic!("Divergence at op {}: expected {}, got {}", i, op.result, our_result);
    }
}
```

### Step 3: Binary Search within First TU
Once divergence TU is found (call #23 = first 4x4 in second 8x8 block?), add logging for every operation within that TU. Find the exact CABAC call that diverges.

### Step 4: Root Cause Analysis
The diverging call will be one of:
- Wrong context index (context derivation bug)
- Wrong scan position (scan table bug)
- Wrong number of calls (loop logic bug)
- Wrong order of operations (state machine bug)

Fix it, and the 24 large coefficients should become 0.

## Files to Focus On

1. **residual.rs:180-400** - Main coefficient decode loop
2. **residual.rs:450-550** - sig_coeff_flag context derivation
3. **residual.rs:600-700** - greater1/greater2 flag handling
4. **residual.rs:750-850** - coeff_abs_level_remaining calls
5. **ctu.rs:400-500** - decode_transform_unit orchestration

## Test Commands

```bash
# Run SSIM2 comparison
cargo test --test compare_reference test_ssim2_against_reference -- --nocapture

# Run pixel diff stats
cargo test --test compare_reference test_pixel_difference_stats -- --nocapture

# Generate comparison images
cargo test --test compare_reference write_comparison_images -- --nocapture --ignored

# Run hevc-compare tests
cd crates/hevc-compare && cargo test -- --nocapture

# Full decode with debug output
cargo test --test decode_heic test_decode -- --nocapture --ignored
```

## Reference Decoder Access

```rust
// In tests, using heic-wasm-rs (libheif/libde265 via WASM):
let ref_decoder = heic_wasm_rs::HeicDecoder::from_file(
    Path::new("/home/lilith/work/heic/wasm-module/heic_decoder.wasm")
)?;
let ref_image = ref_decoder.decode(&data)?;

// Or using the local libde265-src:
// /home/lilith/work/heic/libde265-src/
```

## Appendix: CABAC State Machine

```
CABAC State = (range: u32, value: u32, bits_needed: i32)

Initial: range=510, value=first_two_bytes, bits_needed=-8

Bypass decode:
  value <<= 1
  bits_needed += 1
  if bits_needed >= 0: read_byte()
  scaled_range = range << 7
  if value >= scaled_range:
    value -= scaled_range
    return 1
  else:
    return 0

Context decode:
  lps = LPS_TABLE[ctx.state][(range >> 6) - 4]
  range -= lps
  scaled_range = range << 7
  if value < scaled_range:
    // MPS path
    ctx.state = TRANS_MPS[ctx.state]
    bin = ctx.mps
  else:
    // LPS path
    value -= scaled_range
    range = lps
    bin = 1 - ctx.mps
    if ctx.state == 0: ctx.mps ^= 1
    ctx.state = TRANS_LPS[ctx.state]
  renormalize()
  return bin
```

## Conclusion

The bug is NOT in CABAC primitives. It's in coefficient decode orchestration - likely a context index calculation or scan order issue. The fix requires finding the exact divergence point using trace comparison with libde265.

---

*Good luck, future session. The answer is in the traces.*
