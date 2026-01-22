//! HEVC function comparison crate
//!
//! Compares pure functions between libde265 (C++) and our Rust implementation
//! to find divergence points.

#![allow(non_camel_case_types)]

use std::ffi::c_int;

// FFI bindings to C++ functions
#[repr(C)]
pub struct CabacState {
    range: u32,
    value: u32,
    bits_needed: c_int,
    bitstream_curr: *const u8,
    bitstream_end: *const u8,
}

#[repr(C)]
pub struct CppContextModel {
    state: u8,
    mps: u8,
}

unsafe extern "C" {
    fn cabac_init(state: *mut CabacState, data: *const u8, length: c_int);
    fn cabac_decode_bypass(state: *mut CabacState) -> c_int;
    fn cabac_decode_bypass_bits(state: *mut CabacState, num_bits: c_int) -> u32;
    fn cabac_decode_coeff_abs_level_remaining(state: *mut CabacState, rice_param: c_int) -> c_int;
    fn cabac_get_state(state: *const CabacState, range: *mut u32, value: *mut u32, bits_needed: *mut c_int);
    fn context_init(ctx: *mut CppContextModel, init_value: u8, slice_qp: c_int);
    fn context_get_state(ctx: *const CppContextModel, state: *mut u8, mps: *mut u8);
    fn cabac_decode_bin(decoder: *mut CabacState, model: *mut CppContextModel) -> c_int;
}

/// C++ CABAC decoder wrapper
pub struct CppCabac {
    state: CabacState,
    // Keep the data alive - leaked to ensure stable address
    _data: &'static [u8],
}

impl CppCabac {
    pub fn new(data: &[u8]) -> Self {
        // Leak the data to get a stable address (for testing only)
        let data_leaked: &'static [u8] = Box::leak(data.to_vec().into_boxed_slice());

        let mut state = CabacState {
            range: 0,
            value: 0,
            bits_needed: 0,
            bitstream_curr: std::ptr::null(),
            bitstream_end: std::ptr::null(),
        };

        unsafe {
            cabac_init(&mut state, data_leaked.as_ptr(), data_leaked.len() as c_int);
        }

        Self {
            state,
            _data: data_leaked,
        }
    }

    pub fn decode_bypass(&mut self) -> u32 {
        unsafe { cabac_decode_bypass(&mut self.state) as u32 }
    }

    pub fn decode_bypass_bits(&mut self, num_bits: u8) -> u32 {
        unsafe { cabac_decode_bypass_bits(&mut self.state, num_bits as c_int) }
    }

    pub fn decode_coeff_abs_level_remaining(&mut self, rice_param: u8) -> i32 {
        unsafe { cabac_decode_coeff_abs_level_remaining(&mut self.state, rice_param as c_int) }
    }

    pub fn get_state(&self) -> (u32, u32, i32) {
        let mut range = 0u32;
        let mut value = 0u32;
        let mut bits_needed = 0i32;
        unsafe {
            cabac_get_state(&self.state, &mut range, &mut value, &mut bits_needed);
        }
        (range, value, bits_needed)
    }

    pub fn decode_bin(&mut self, ctx: &mut CppContextModel) -> u32 {
        unsafe { cabac_decode_bin(&mut self.state, ctx) as u32 }
    }
}

/// C++ context model wrapper
pub struct CppContext {
    model: CppContextModel,
}

impl CppContext {
    pub fn new(init_value: u8, slice_qp: i32) -> Self {
        let mut model = CppContextModel { state: 0, mps: 0 };
        unsafe {
            context_init(&mut model, init_value, slice_qp as c_int);
        }
        Self { model }
    }

    pub fn get_state(&self) -> (u8, u8) {
        let mut state = 0u8;
        let mut mps = 0u8;
        unsafe {
            context_get_state(&self.model, &mut state, &mut mps);
        }
        (state, mps)
    }

    pub fn model_mut(&mut self) -> &mut CppContextModel {
        &mut self.model
    }
}

/// Rust CABAC decoder (our implementation)
pub struct RustCabac<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    value: u32,
    bits_needed: i32,
}

impl<'a> RustCabac<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut cabac = Self {
            data,
            pos: 0,
            range: 510,
            value: 0,
            bits_needed: 8,
        };

        // Initialize value (matching C++ init)
        cabac.bits_needed = -8;
        if cabac.pos < cabac.data.len() {
            cabac.value = cabac.data[cabac.pos] as u32;
            cabac.pos += 1;
        }
        cabac.value <<= 8;
        cabac.bits_needed = 0;
        if cabac.pos < cabac.data.len() {
            cabac.value |= cabac.data[cabac.pos] as u32;
            cabac.pos += 1;
            cabac.bits_needed = -8;
        }

        cabac
    }

    pub fn decode_bypass(&mut self) -> u32 {
        self.value <<= 1;
        self.bits_needed += 1;

        if self.bits_needed >= 0 {
            if self.pos < self.data.len() {
                self.bits_needed = -8;
                self.value |= self.data[self.pos] as u32;
                self.pos += 1;
            } else {
                self.bits_needed = -8;
            }
        }

        let scaled_range = self.range << 7;
        if self.value >= scaled_range {
            self.value -= scaled_range;
            1
        } else {
            0
        }
    }

    pub fn decode_bypass_bits(&mut self, num_bits: u8) -> u32 {
        let mut value = 0u32;
        for _ in 0..num_bits {
            value = (value << 1) | self.decode_bypass();
        }
        value
    }

    pub fn decode_coeff_abs_level_remaining(&mut self, rice_param: u8) -> i32 {
        // Count prefix (unary 1s terminated by 0)
        let mut prefix = 0u32;
        while self.decode_bypass() != 0 && prefix < 32 {
            prefix += 1;
        }

        let value = if prefix <= 3 {
            // TR part only
            let suffix = self.decode_bypass_bits(rice_param);
            ((prefix << rice_param) + suffix) as i32
        } else {
            // EGk part
            let suffix_bits = (prefix - 3 + rice_param as u32) as u8;
            let suffix = self.decode_bypass_bits(suffix_bits);
            let base = ((1u32 << (prefix - 3)) + 2) << rice_param;
            (base + suffix) as i32
        };

        value
    }

    pub fn get_state(&self) -> (u32, u32, i32) {
        (self.range, self.value, self.bits_needed)
    }

    /// Read a single bit (for renormalization)
    fn read_bit(&mut self) {
        self.value <<= 1;
        self.bits_needed += 1;

        if self.bits_needed >= 0 {
            if self.pos < self.data.len() {
                self.bits_needed = -8;
                self.value |= self.data[self.pos] as u32;
                self.pos += 1;
            } else {
                self.bits_needed = -8;
            }
        }
    }

    /// Renormalize the decoder state
    fn renormalize(&mut self) {
        while self.range < 256 {
            self.range <<= 1;
            self.read_bit();
        }
    }

    /// Decode a context-coded bin
    pub fn decode_bin(&mut self, ctx: &mut RustContext) -> u32 {
        let q_range_idx = (self.range >> 6) & 3;
        let lps_range = LPS_TABLE[ctx.state as usize][q_range_idx as usize] as u32;

        self.range -= lps_range;

        let scaled_range = self.range << 7;

        let bin_val;
        if self.value < scaled_range {
            // MPS path
            bin_val = ctx.mps as u32;
            ctx.state = STATE_TRANS_MPS[ctx.state as usize];
        } else {
            // LPS path
            bin_val = (1 - ctx.mps) as u32;
            self.value -= scaled_range;
            self.range = lps_range;

            if ctx.state == 0 {
                ctx.mps = 1 - ctx.mps;
            }
            ctx.state = STATE_TRANS_LPS[ctx.state as usize];
        }

        self.renormalize();
        bin_val
    }
}

/// CABAC tables from H.265 spec
static LPS_TABLE: [[u8; 4]; 64] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
    [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 122, 144, 166],
    [95, 116, 137, 158], [90, 110, 130, 150], [85, 104, 123, 142], [81, 99, 117, 135],
    [77, 94, 111, 128], [73, 89, 105, 122], [69, 85, 100, 116], [66, 80, 95, 110],
    [62, 76, 90, 104], [59, 72, 86, 99], [56, 69, 81, 94], [53, 65, 77, 89],
    [51, 62, 73, 85], [48, 59, 69, 80], [46, 56, 66, 76], [43, 53, 63, 72],
    [41, 50, 59, 69], [39, 48, 56, 65], [37, 45, 54, 62], [35, 43, 51, 59],
    [33, 41, 48, 56], [32, 39, 46, 53], [30, 37, 43, 50], [29, 35, 41, 48],
    [27, 33, 39, 45], [26, 31, 37, 43], [24, 30, 35, 41], [23, 28, 33, 39],
    [22, 27, 32, 37], [21, 26, 30, 35], [20, 24, 29, 33], [19, 23, 27, 31],
    [18, 22, 26, 30], [17, 21, 25, 28], [16, 20, 23, 27], [15, 19, 22, 25],
    [14, 18, 21, 24], [14, 17, 20, 23], [13, 16, 19, 22], [12, 15, 18, 21],
    [12, 14, 17, 20], [11, 14, 16, 19], [11, 13, 15, 18], [10, 12, 15, 17],
    [10, 12, 14, 16], [9, 11, 13, 15], [9, 11, 12, 14], [8, 10, 12, 14],
    [8, 9, 11, 13], [7, 9, 11, 12], [7, 9, 10, 12], [7, 8, 10, 11],
    [6, 8, 9, 11], [6, 7, 9, 10], [6, 7, 8, 9], [2, 2, 2, 2],
];

static STATE_TRANS_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

static STATE_TRANS_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// Rust context model
pub struct RustContext {
    pub state: u8,
    pub mps: u8,
}

impl RustContext {
    pub fn new(init_value: u8, slice_qp: i32) -> Self {
        let slope = (init_value >> 4) as i32 * 5 - 45;
        let offset = ((init_value & 15) << 3) as i32 - 16;

        let init_state = ((slope * (slice_qp - 16)) >> 4) + offset;
        let init_state = init_state.clamp(1, 126);

        if init_state >= 64 {
            Self {
                state: (init_state - 64) as u8,
                mps: 1,
            }
        } else {
            Self {
                state: (63 - init_state) as u8,
                mps: 0,
            }
        }
    }

    pub fn get_state(&self) -> (u8, u8) {
        (self.state, self.mps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test data - random bytes for CABAC testing
    const TEST_DATA: &[u8] = &[
        0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x01,
        0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
    ];

    #[test]
    fn test_bypass_decode_matches() {
        let mut cpp = CppCabac::new(TEST_DATA);
        let mut rust = RustCabac::new(TEST_DATA);

        for i in 0..100 {
            let cpp_bit = cpp.decode_bypass();
            let rust_bit = rust.decode_bypass();

            let (cpp_r, cpp_v, cpp_b) = cpp.get_state();
            let (rust_r, rust_v, rust_b) = rust.get_state();

            assert_eq!(cpp_bit, rust_bit,
                "Bit {} mismatch: C++={} Rust={}", i, cpp_bit, rust_bit);
            assert_eq!(cpp_r, rust_r,
                "Range mismatch at bit {}: C++={} Rust={}", i, cpp_r, rust_r);
            assert_eq!(cpp_v, rust_v,
                "Value mismatch at bit {}: C++={} Rust={}", i, cpp_v, rust_v);
            assert_eq!(cpp_b, rust_b,
                "Bits_needed mismatch at bit {}: C++={} Rust={}", i, cpp_b, rust_b);
        }
    }

    #[test]
    fn test_bypass_bits_matches() {
        for num_bits in 1..=8 {
            let mut cpp = CppCabac::new(TEST_DATA);
            let mut rust = RustCabac::new(TEST_DATA);

            for i in 0..10 {
                let cpp_val = cpp.decode_bypass_bits(num_bits);
                let rust_val = rust.decode_bypass_bits(num_bits);

                assert_eq!(cpp_val, rust_val,
                    "Bypass bits mismatch at iteration {}, num_bits={}: C++={} Rust={}",
                    i, num_bits, cpp_val, rust_val);
            }
        }
    }

    /// Simulate sign decoding for a sub-block
    /// Returns the signs decoded and the final state
    fn simulate_sign_decode(cabac: &mut impl CabacLike, num_coeffs: usize, skip_last: bool) -> Vec<u32> {
        let to_decode = if skip_last { num_coeffs.saturating_sub(1) } else { num_coeffs };
        let mut signs = Vec::with_capacity(to_decode);
        for _ in 0..to_decode {
            signs.push(cabac.decode_bypass());
        }
        signs
    }

    trait CabacLike {
        fn decode_bypass(&mut self) -> u32;
        fn get_state(&self) -> (u32, u32, i32);
    }

    impl CabacLike for CppCabac {
        fn decode_bypass(&mut self) -> u32 { CppCabac::decode_bypass(self) }
        fn get_state(&self) -> (u32, u32, i32) { CppCabac::get_state(self) }
    }

    impl<'a> CabacLike for RustCabac<'a> {
        fn decode_bypass(&mut self) -> u32 { RustCabac::decode_bypass(self) }
        fn get_state(&self) -> (u32, u32, i32) { RustCabac::get_state(self) }
    }

    #[test]
    fn test_sign_decode_with_hiding() {
        // Test that skipping the last sign bit causes divergence
        // This simulates what happens with sign_data_hiding

        // Decode signs for 8 coefficients WITHOUT hiding
        let mut cpp_no_hide = CppCabac::new(TEST_DATA);
        let mut rust_no_hide = RustCabac::new(TEST_DATA);

        let cpp_signs = simulate_sign_decode(&mut cpp_no_hide, 8, false);
        let rust_signs = simulate_sign_decode(&mut rust_no_hide, 8, false);

        println!("Without hiding (8 signs): C++={:?} Rust={:?}", cpp_signs, rust_signs);
        assert_eq!(cpp_signs, rust_signs, "Signs should match without hiding");

        let (cpp_r, cpp_v, cpp_b) = cpp_no_hide.get_state();
        let (rust_r, rust_v, rust_b) = rust_no_hide.get_state();
        println!("State after 8 signs: C++=({},{},{}) Rust=({},{},{})", cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);
        assert_eq!((cpp_r, cpp_v, cpp_b), (rust_r, rust_v, rust_b));

        // Now decode signs for 8 coefficients WITH hiding (skip last)
        let mut cpp_hide = CppCabac::new(TEST_DATA);
        let mut rust_hide = RustCabac::new(TEST_DATA);

        let cpp_signs_hide = simulate_sign_decode(&mut cpp_hide, 8, true);
        let rust_signs_hide = simulate_sign_decode(&mut rust_hide, 8, true);

        println!("With hiding (7 signs): C++={:?} Rust={:?}", cpp_signs_hide, rust_signs_hide);
        assert_eq!(cpp_signs_hide, rust_signs_hide, "Signs should match with hiding");

        let (cpp_r, cpp_v, cpp_b) = cpp_hide.get_state();
        let (rust_r, rust_v, rust_b) = rust_hide.get_state();
        println!("State after 7 signs: C++=({},{},{}) Rust=({},{},{})", cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);
        assert_eq!((cpp_r, cpp_v, cpp_b), (rust_r, rust_v, rust_b));

        // The state after hiding should be DIFFERENT from without hiding
        // (one less bit consumed)
        let (no_hide_r, no_hide_v, _) = cpp_no_hide.get_state();
        let (hide_r, hide_v, _) = cpp_hide.get_state();
        println!("\nState comparison:");
        println!("  After 8 signs (no hiding): range={}, value={}", no_hide_r, no_hide_v);
        println!("  After 7 signs (with hiding): range={}, value={}", hide_r, hide_v);
        assert_ne!(no_hide_v, hide_v, "States should differ when hiding one sign");
    }

    #[test]
    fn test_coeff_abs_level_remaining_matches() {
        for rice_param in 0..=4 {
            let mut cpp = CppCabac::new(TEST_DATA);
            let mut rust = RustCabac::new(TEST_DATA);

            for i in 0..5 {
                let (cpp_r, cpp_v, cpp_b) = cpp.get_state();
                let (rust_r, rust_v, rust_b) = rust.get_state();

                println!("Before decode {}, rice={}: C++ state=({},{},{}) Rust state=({},{},{})",
                    i, rice_param, cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);

                let cpp_val = cpp.decode_coeff_abs_level_remaining(rice_param);
                let rust_val = rust.decode_coeff_abs_level_remaining(rice_param);

                println!("  C++ result={}, Rust result={}", cpp_val, rust_val);

                assert_eq!(cpp_val, rust_val,
                    "coeff_abs_level_remaining mismatch at iteration {}, rice_param={}: C++={} Rust={}",
                    i, rice_param, cpp_val, rust_val);

                let (cpp_r, cpp_v, cpp_b) = cpp.get_state();
                let (rust_r, rust_v, rust_b) = rust.get_state();
                assert_eq!((cpp_r, cpp_v, cpp_b), (rust_r, rust_v, rust_b),
                    "State mismatch after decode {}: C++=({},{},{}) Rust=({},{},{})",
                    i, cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);
            }
        }
    }

    #[test]
    fn test_context_init_matches() {
        // Test context initialization for various init values and QPs
        for init_value in [111u8, 125, 140, 153, 179, 154, 139] {
            for slice_qp in [20, 26, 30, 40] {
                let cpp_ctx = CppContext::new(init_value, slice_qp);
                let rust_ctx = RustContext::new(init_value, slice_qp);

                let (cpp_state, cpp_mps) = cpp_ctx.get_state();
                let (rust_state, rust_mps) = rust_ctx.get_state();

                assert_eq!((cpp_state, cpp_mps), (rust_state, rust_mps),
                    "Context init mismatch for init_value={}, qp={}: C++=({},{}) Rust=({},{})",
                    init_value, slice_qp, cpp_state, cpp_mps, rust_state, rust_mps);
            }
        }
    }

    #[test]
    fn test_context_coded_bin_matches() {
        // Test context-coded bin decoding
        // Use init_value 154 (a common default) and QP 26
        let init_value = 154u8;
        let slice_qp = 26;

        let mut cpp_cabac = CppCabac::new(TEST_DATA);
        let mut rust_cabac = RustCabac::new(TEST_DATA);

        let mut cpp_ctx = CppContext::new(init_value, slice_qp);
        let mut rust_ctx = RustContext::new(init_value, slice_qp);

        // Decode 50 context-coded bins
        for i in 0..50 {
            let (cpp_r, cpp_v, cpp_b) = cpp_cabac.get_state();
            let (rust_r, rust_v, rust_b) = rust_cabac.get_state();
            let (cpp_state, cpp_mps) = cpp_ctx.get_state();
            let (rust_state, rust_mps) = rust_ctx.get_state();

            let cpp_bin = cpp_cabac.decode_bin(cpp_ctx.model_mut());
            let rust_bin = rust_cabac.decode_bin(&mut rust_ctx);

            assert_eq!(cpp_bin, rust_bin,
                "Bin {} mismatch: C++={} Rust={}\n\
                 Before decode: C++ cabac=({},{},{}) ctx=({},{}) | Rust cabac=({},{},{}) ctx=({},{})",
                i, cpp_bin, rust_bin,
                cpp_r, cpp_v, cpp_b, cpp_state, cpp_mps,
                rust_r, rust_v, rust_b, rust_state, rust_mps);

            // Also verify state after decode
            let (cpp_r2, cpp_v2, cpp_b2) = cpp_cabac.get_state();
            let (rust_r2, rust_v2, rust_b2) = rust_cabac.get_state();
            let (cpp_state2, cpp_mps2) = cpp_ctx.get_state();
            let (rust_state2, rust_mps2) = rust_ctx.get_state();

            assert_eq!((cpp_r2, cpp_v2, cpp_b2), (rust_r2, rust_v2, rust_b2),
                "CABAC state mismatch after bin {}: C++=({},{},{}) Rust=({},{},{})",
                i, cpp_r2, cpp_v2, cpp_b2, rust_r2, rust_v2, rust_b2);

            assert_eq!((cpp_state2, cpp_mps2), (rust_state2, rust_mps2),
                "Context state mismatch after bin {}: C++=({},{}) Rust=({},{})",
                i, cpp_state2, cpp_mps2, rust_state2, rust_mps2);
        }
    }

    #[test]
    fn test_multiple_contexts() {
        // Test with multiple different contexts being used
        // This simulates real coefficient decoding where different contexts are used
        let init_values = [111u8, 125, 140, 153, 154];
        let slice_qp = 26;

        let mut cpp_cabac = CppCabac::new(TEST_DATA);
        let mut rust_cabac = RustCabac::new(TEST_DATA);

        let mut cpp_ctxs: Vec<_> = init_values.iter()
            .map(|&iv| CppContext::new(iv, slice_qp))
            .collect();
        let mut rust_ctxs: Vec<_> = init_values.iter()
            .map(|&iv| RustContext::new(iv, slice_qp))
            .collect();

        // Decode 100 bins, cycling through contexts
        for i in 0..100 {
            let ctx_idx = i % init_values.len();

            let cpp_bin = cpp_cabac.decode_bin(cpp_ctxs[ctx_idx].model_mut());
            let rust_bin = rust_cabac.decode_bin(&mut rust_ctxs[ctx_idx]);

            assert_eq!(cpp_bin, rust_bin,
                "Bin {} (ctx {}) mismatch: C++={} Rust={}",
                i, ctx_idx, cpp_bin, rust_bin);

            // Verify states match
            let (cpp_r, cpp_v, cpp_b) = cpp_cabac.get_state();
            let (rust_r, rust_v, rust_b) = rust_cabac.get_state();

            assert_eq!((cpp_r, cpp_v, cpp_b), (rust_r, rust_v, rust_b),
                "CABAC state mismatch after bin {}: C++=({},{},{}) Rust=({},{},{})",
                i, cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);
        }
    }

    /// Test with actual slice data from example.heic
    /// The slice data starts at the specified bytes
    #[test]
    fn test_real_slice_data() {
        // First 32 bytes of slice data from example.heic
        // (after slice header, at CABAC data start)
        let slice_data: &[u8] = &[
            0x49, 0xc0, 0xc2, 0xc4, 0x92, 0x61, 0x0c, 0x00,
            0x16, 0xcc, 0xbe, 0x77, 0x82, 0x8c, 0xcb, 0xfa,
            0x93, 0x5f, 0xb2, 0x6a, 0x65, 0x34, 0xe6, 0xf8,
            0xd3, 0xa0, 0x76, 0xcc, 0x39, 0xe8, 0xe0, 0xac,
        ];

        let mut cpp = CppCabac::new(slice_data);
        let mut rust = RustCabac::new(slice_data);

        // Verify initial state matches
        let (cpp_r, cpp_v, cpp_b) = cpp.get_state();
        let (rust_r, rust_v, rust_b) = rust.get_state();
        println!("Initial state: C++=({},{},{}) Rust=({},{},{})",
            cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);
        assert_eq!((cpp_r, cpp_v, cpp_b), (rust_r, rust_v, rust_b),
            "Initial state mismatch");

        // Decode 200 bypass bits and compare
        for i in 0..200 {
            let (cpp_r, cpp_v, cpp_b) = cpp.get_state();
            let (rust_r, rust_v, rust_b) = rust.get_state();

            let cpp_bit = cpp.decode_bypass();
            let rust_bit = rust.decode_bypass();

            if cpp_bit != rust_bit {
                panic!("Bypass {} mismatch: C++={} Rust={}\n\
                        Before: C++=({},{},{}) Rust=({},{},{})",
                    i, cpp_bit, rust_bit, cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);
            }

            let (cpp_r2, cpp_v2, cpp_b2) = cpp.get_state();
            let (rust_r2, rust_v2, rust_b2) = rust.get_state();

            if (cpp_r2, cpp_v2, cpp_b2) != (rust_r2, rust_v2, rust_b2) {
                panic!("State after bypass {} mismatch:\n\
                        C++=({},{},{}) Rust=({},{},{})",
                    i, cpp_r2, cpp_v2, cpp_b2, rust_r2, rust_v2, rust_b2);
            }
        }
        println!("All 200 bypass bits match!");
    }

    /// Test a realistic coefficient decode sequence
    /// Uses context indices and operations similar to actual TU decode
    #[test]
    fn test_realistic_coeff_decode_sequence() {
        // Slice data from example.heic
        let slice_data: &[u8] = &[
            0x49, 0xc0, 0xc2, 0xc4, 0x92, 0x61, 0x0c, 0x00,
            0x16, 0xcc, 0xbe, 0x77, 0x82, 0x8c, 0xcb, 0xfa,
            0x93, 0x5f, 0xb2, 0x6a, 0x65, 0x34, 0xe6, 0xf8,
            0xd3, 0xa0, 0x76, 0xcc, 0x39, 0xe8, 0xe0, 0xac,
            0x4d, 0x7e, 0xc9, 0xa9, 0x95, 0xd3, 0x9b, 0xe3,
            0x4e, 0x81, 0xdb, 0x30, 0xe7, 0xa3, 0x82, 0xb1,
        ];

        let slice_qp = 17;

        let mut cpp = CppCabac::new(slice_data);
        let mut rust = RustCabac::new(slice_data);

        // Initialize contexts for sig_coeff_flag (init_values for SIG_COEFF_FLAG)
        // Using first 16 contexts with their actual init values
        let sig_coeff_init: [u8; 16] = [
            111, 111, 125, 110, 110, 94, 124, 108,
            124, 107, 125, 141, 179, 153, 125, 107,
        ];

        let mut cpp_ctxs: Vec<_> = sig_coeff_init.iter()
            .map(|&iv| CppContext::new(iv, slice_qp))
            .collect();
        let mut rust_ctxs: Vec<_> = sig_coeff_init.iter()
            .map(|&iv| RustContext::new(iv, slice_qp))
            .collect();

        // Simulate coefficient decoding pattern:
        // - Decode context-coded sig_coeff_flag
        // - Decode context-coded greater1_flag
        // - Decode bypass sign bits
        // - Decode bypass coeff_abs_level_remaining

        println!("Simulating coefficient decode sequence...");

        for iteration in 0..20 {
            // Decode 4 sig_coeff_flags using different contexts
            for ctx_offset in 0..4 {
                let ctx_idx = (iteration * 3 + ctx_offset) % 16;

                let (cpp_r, cpp_v, cpp_b) = cpp.get_state();
                let (rust_r, rust_v, rust_b) = rust.get_state();

                let cpp_bin = cpp.decode_bin(cpp_ctxs[ctx_idx].model_mut());
                let rust_bin = rust.decode_bin(&mut rust_ctxs[ctx_idx]);

                if cpp_bin != rust_bin {
                    let (cpp_state, cpp_mps) = cpp_ctxs[ctx_idx].get_state();
                    let (rust_state, rust_mps) = rust_ctxs[ctx_idx].get_state();
                    panic!("Iteration {}, sig_coeff ctx {}: C++={} Rust={}\n\
                            Before: C++=({},{},{}) Rust=({},{},{})\n\
                            Context after: C++=({},{}) Rust=({},{})",
                        iteration, ctx_idx, cpp_bin, rust_bin,
                        cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b,
                        cpp_state, cpp_mps, rust_state, rust_mps);
                }
            }

            // Decode 2 bypass sign bits
            for _ in 0..2 {
                let cpp_bit = cpp.decode_bypass();
                let rust_bit = rust.decode_bypass();
                assert_eq!(cpp_bit, rust_bit, "Sign bypass mismatch at iteration {}", iteration);
            }

            // Decode coeff_abs_level_remaining
            let rice_param = (iteration % 5) as u8;
            let cpp_remaining = cpp.decode_coeff_abs_level_remaining(rice_param);
            let rust_remaining = rust.decode_coeff_abs_level_remaining(rice_param);

            if cpp_remaining != rust_remaining {
                let (cpp_r, cpp_v, cpp_b) = cpp.get_state();
                let (rust_r, rust_v, rust_b) = rust.get_state();
                panic!("coeff_remaining mismatch at iteration {}, rice={}:\n\
                        C++={} Rust={}\n\
                        State after: C++=({},{},{}) Rust=({},{},{})",
                    iteration, rice_param, cpp_remaining, rust_remaining,
                    cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);
            }

            // Verify state still matches
            let (cpp_r, cpp_v, cpp_b) = cpp.get_state();
            let (rust_r, rust_v, rust_b) = rust.get_state();
            if (cpp_r, cpp_v, cpp_b) != (rust_r, rust_v, rust_b) {
                panic!("State diverged at iteration {}: C++=({},{},{}) Rust=({},{},{})",
                    iteration, cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);
            }
        }

        println!("All 20 iterations matched!");
    }

    #[test]
    fn test_mixed_context_and_bypass() {
        // Test interleaved context-coded and bypass bins
        // This is how real coefficient decoding works
        let init_value = 154u8;
        let slice_qp = 26;

        let mut cpp_cabac = CppCabac::new(TEST_DATA);
        let mut rust_cabac = RustCabac::new(TEST_DATA);

        let mut cpp_ctx = CppContext::new(init_value, slice_qp);
        let mut rust_ctx = RustContext::new(init_value, slice_qp);

        for i in 0..20 {
            // Decode 3 context-coded bins
            for j in 0..3 {
                let cpp_bin = cpp_cabac.decode_bin(cpp_ctx.model_mut());
                let rust_bin = rust_cabac.decode_bin(&mut rust_ctx);
                assert_eq!(cpp_bin, rust_bin,
                    "Iteration {}, context bin {}: C++={} Rust={}", i, j, cpp_bin, rust_bin);
            }

            // Decode 2 bypass bins
            for j in 0..2 {
                let cpp_bit = cpp_cabac.decode_bypass();
                let rust_bit = rust_cabac.decode_bypass();
                assert_eq!(cpp_bit, rust_bit,
                    "Iteration {}, bypass bin {}: C++={} Rust={}", i, j, cpp_bit, rust_bit);
            }

            // Verify states match after mixed decoding
            let (cpp_r, cpp_v, cpp_b) = cpp_cabac.get_state();
            let (rust_r, rust_v, rust_b) = rust_cabac.get_state();
            assert_eq!((cpp_r, cpp_v, cpp_b), (rust_r, rust_v, rust_b),
                "State mismatch after iteration {}: C++=({},{},{}) Rust=({},{},{})",
                i, cpp_r, cpp_v, cpp_b, rust_r, rust_v, rust_b);
        }
    }
}
