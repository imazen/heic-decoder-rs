//! Residual (transform coefficient) decoding
//!
//! This module handles parsing of transform coefficients via CABAC
//! and applying inverse transforms to residuals.

use super::cabac::{CabacDecoder, ContextModel, context};
use super::debug;
use super::transform::MAX_COEFF;
use crate::error::HevcError;

type Result<T> = core::result::Result<T, HevcError>;

/// Scan order types for coefficient scanning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOrder {
    /// Diagonal scan (default)
    Diagonal = 0,
    /// Horizontal scan (for horizontal intra modes)
    Horizontal = 1,
    /// Vertical scan (for vertical intra modes)
    Vertical = 2,
}

/// Get scan order based on intra prediction mode
pub fn get_scan_order(log2_size: u8, intra_mode: u8) -> ScanOrder {
    // For 4x4 and 8x8 TUs, use scan order based on intra mode
    if log2_size == 2 || log2_size == 3 {
        if (6..=14).contains(&intra_mode) {
            ScanOrder::Vertical
        } else if (22..=30).contains(&intra_mode) {
            ScanOrder::Horizontal
        } else {
            ScanOrder::Diagonal
        }
    } else {
        ScanOrder::Diagonal
    }
}

/// 4x4 diagonal scan order (indexed by position 0-15)
pub static SCAN_ORDER_4X4_DIAG: [(u8, u8); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (1, 1),
    (0, 2),
    (3, 0),
    (2, 1),
    (1, 2),
    (0, 3),
    (3, 1),
    (2, 2),
    (1, 3),
    (3, 2),
    (2, 3),
    (3, 3),
];

/// 4x4 horizontal scan order
pub static SCAN_ORDER_4X4_HORIZ: [(u8, u8); 16] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
];

/// 4x4 vertical scan order
pub static SCAN_ORDER_4X4_VERT: [(u8, u8); 16] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
];

/// Get scan order table for 4x4 sub-blocks
pub fn get_scan_4x4(order: ScanOrder) -> &'static [(u8, u8); 16] {
    match order {
        ScanOrder::Diagonal => &SCAN_ORDER_4X4_DIAG,
        ScanOrder::Horizontal => &SCAN_ORDER_4X4_HORIZ,
        ScanOrder::Vertical => &SCAN_ORDER_4X4_VERT,
    }
}

/// Coefficient buffer for a transform unit
#[derive(Clone)]
pub struct CoeffBuffer {
    /// Coefficients for this TU
    pub coeffs: [i16; MAX_COEFF],
    /// Transform size (log2)
    pub log2_size: u8,
    /// Number of non-zero coefficients
    pub num_nonzero: u16,
}

impl Default for CoeffBuffer {
    fn default() -> Self {
        Self {
            coeffs: [0; MAX_COEFF],
            log2_size: 2,
            num_nonzero: 0,
        }
    }
}

impl CoeffBuffer {
    /// Create a new coefficient buffer
    pub fn new(log2_size: u8) -> Self {
        Self {
            coeffs: [0; MAX_COEFF],
            log2_size,
            num_nonzero: 0,
        }
    }

    /// Get the transform size
    pub fn size(&self) -> usize {
        1 << self.log2_size
    }

    /// Get coefficient at position
    pub fn get(&self, x: usize, y: usize) -> i16 {
        let stride = self.size();
        self.coeffs[y * stride + x]
    }

    /// Set coefficient at position
    pub fn set(&mut self, x: usize, y: usize, value: i16) {
        let stride = self.size();
        self.coeffs[y * stride + x] = value;
        if value != 0 {
            self.num_nonzero = self.num_nonzero.saturating_add(1);
        }
    }

    /// Check if all coefficients are zero
    pub fn is_zero(&self) -> bool {
        self.num_nonzero == 0
    }
}

/// Decode residual coefficients for a transform unit
/// Debug counter to identify specific TU calls
pub static DEBUG_RESIDUAL_COUNTER: core::sync::atomic::AtomicU32 =
    core::sync::atomic::AtomicU32::new(0);

pub fn decode_residual(
    cabac: &mut CabacDecoder<'_>,
    ctx: &mut [ContextModel],
    log2_size: u8,
    c_idx: u8, // 0=Y, 1=Cb, 2=Cr
    scan_order: ScanOrder,
    sign_data_hiding_enabled: bool,
    cu_transquant_bypass: bool,
) -> Result<CoeffBuffer> {
    // Track initial CABAC state for debugging
    let (init_range, init_offset) = cabac.get_state();

    // Increment and capture counter for debugging
    let residual_call_num =
        DEBUG_RESIDUAL_COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

    // Verbose debug for specific call (disabled by default)
    #[allow(unused_variables)]
    let debug_call = false;
    if debug_call {
        let (byte_pos, _, _) = cabac.get_position();
        eprintln!(
            "DEBUG call#{}: START log2={} c_idx={} scan={:?} byte={} cabac=({},{})",
            residual_call_num, log2_size, c_idx, scan_order, byte_pos, init_range, init_offset
        );
    }

    let mut buffer = CoeffBuffer::new(log2_size);
    let size = 1u32 << log2_size;

    // Decode last significant coefficient position
    let (last_x, last_y) = decode_last_sig_coeff_pos(cabac, ctx, log2_size, c_idx)?;

    // Swap coordinates for vertical scan
    let (last_x, last_y) = if scan_order == ScanOrder::Vertical {
        (last_y, last_x)
    } else {
        (last_x, last_y)
    };

    if last_x >= size || last_y >= size {
        return Err(HevcError::InvalidBitstream("invalid last coeff position"));
    }

    // Get scan tables
    let scan_sub = get_scan_sub_block(log2_size, scan_order);
    let scan_pos = get_scan_4x4(scan_order);

    // Convert ScanOrder to scan_idx for context derivation
    let scan_idx = match scan_order {
        ScanOrder::Diagonal => 0,
        ScanOrder::Horizontal => 1,
        ScanOrder::Vertical => 2,
    };

    // Find last sub-block
    let sb_width = (size / 4) as usize;
    let last_sb_x = last_x / 4;
    let last_sb_y = last_y / 4;
    let last_sb_idx = find_scan_pos(scan_sub, last_sb_x, last_sb_y, sb_width as u32);

    // Find last position within sub-block
    let local_x = (last_x % 4) as u8;
    let local_y = (last_y % 4) as u8;
    let last_pos_in_sb = find_scan_pos_4x4(scan_pos, local_x, local_y);

    // Track coded_sub_block_flag for prevCsbf calculation
    // Max sub-block grid is 8x8 for 32x32 TU
    let mut coded_sb_flags = [[false; 8]; 8];

    // Per libde265 HM algorithm: ctxSet = c1, which is reset to 1 each subblock
    // and becomes 0 if any greater1_flag is 1. Cross-subblock state not needed.

    // Decode coefficients
    for sb_idx in (0..=last_sb_idx).rev() {
        let (sb_x, sb_y) = scan_sub[sb_idx as usize];

        // Calculate neighbor flags BEFORE decoding coded_sub_block_flag
        // These are used for both coded_sub_block_flag and sig_coeff_flag contexts
        // Per H.265: bit 0 = right neighbor coded, bit 1 = below neighbor coded
        let csbf_neighbors = {
            let right_coded = if (sb_x as usize + 1) < sb_width {
                coded_sb_flags[sb_y as usize][sb_x as usize + 1]
            } else {
                false
            };
            let below_coded = if (sb_y as usize + 1) < sb_width {
                coded_sb_flags[sb_y as usize + 1][sb_x as usize]
            } else {
                false
            };
            (if right_coded { 1 } else { 0 }) | (if below_coded { 2 } else { 0 })
        };

        // Check if sub-block is coded
        // Middle sub-blocks need coded_sub_block_flag decoded
        // First (i=0) and last sub-blocks are always considered coded
        let (sb_coded, infer_sb_dc_sig) = if sb_idx > 0 && sb_idx < last_sb_idx {
            // Use simplified context (without neighbor info) for now
            let coded = decode_coded_sub_block_flag_simple(cabac, ctx, c_idx)?;
            // If sub-block is coded, we may need to infer DC later
            (coded, coded)
        } else {
            (true, false) // First and last sub-blocks don't use DC inference
        };

        // Track coded sub-block flag
        if sb_coded {
            coded_sb_flags[sb_y as usize][sb_x as usize] = true;
        }

        // prevCsbf for sig_coeff_flag context
        // Per H.265 spec: bit 0 = below neighbor, bit 1 = right neighbor
        // csbf_neighbors has: bit 0 = right, bit 1 = below (from our calculation)
        // So we swap the bits to match spec semantics
        let prev_csbf = {
            let right_coded = (csbf_neighbors & 1) != 0;
            let below_coded = (csbf_neighbors & 2) != 0;
            (if below_coded { 1 } else { 0 }) | (if right_coded { 2 } else { 0 })
        };

        if !sb_coded {
            continue;
        }

        // Decode coefficients in this sub-block
        let start_pos = if sb_idx == last_sb_idx {
            last_pos_in_sb
        } else {
            15
        };

        let mut coeff_values = [0i16; 16];
        let mut coeff_flags = [false; 16];
        let mut num_coeffs = 0u8;
        let mut can_infer_dc = infer_sb_dc_sig;

        // Determine the last position to check
        // For last sub-block: start from last_pos_in_sb (the known last significant coeff)
        // For other sub-blocks: start from position 15
        let last_coeff = if sb_idx == last_sb_idx {
            // Set the known last significant coefficient (no need to decode sig_coeff_flag)
            coeff_flags[start_pos as usize] = true;
            coeff_values[start_pos as usize] = 1;
            num_coeffs = 1;
            can_infer_dc = false; // Can't infer DC if we have other coeffs
            // Then check positions from start_pos-1 down to 1
            start_pos.saturating_sub(1)
        } else {
            // For non-last sub-blocks, check all positions from 15 down to 1
            15
        };

        // Decode significant_coeff_flags for positions last_coeff down to 1
        // (DC at position 0 is handled separately for inference)
        for n in (1..=last_coeff).rev() {
            let sig = decode_sig_coeff_flag(
                cabac, ctx, c_idx, n, log2_size, scan_idx, sb_x, sb_y, prev_csbf, scan_pos,
            )?;
            if sig {
                coeff_flags[n as usize] = true;
                coeff_values[n as usize] = 1;
                num_coeffs += 1;
                can_infer_dc = false; // Found a coefficient, can't infer DC
            }
        }

        // Handle DC coefficient (position 0)
        // Per H.265: if sub-block is coded but no other coefficients found,
        // DC is inferred to be significant (otherwise the sub-block would be all zeros)
        if start_pos > 0 {
            if can_infer_dc {
                // Infer DC is present - don't decode, just set it
                coeff_flags[0] = true;
                coeff_values[0] = 1;
                num_coeffs += 1;
            } else {
                // Decode sig_coeff_flag for DC
                let sig = decode_sig_coeff_flag(
                    cabac, ctx, c_idx, 0, log2_size, scan_idx, sb_x, sb_y, prev_csbf, scan_pos,
                )?;
                if sig {
                    coeff_flags[0] = true;
                    coeff_values[0] = 1;
                    num_coeffs += 1;
                }
            }
        }

        if num_coeffs == 0 {
            continue;
        }

        // c1 tracks if any greater1_flag is 1 (for next subblock's ctxSet)
        // Reset to 1 at start of each subblock
        // Per libde265: ctxSet = c1 directly (HM algorithm)
        let mut c1 = 1u8;

        // greater1Ctx: context index component, reset to 1 each subblock
        // Updated BEFORE decoding each coefficient (except first) based on PREVIOUS flag
        let mut greater1_ctx = 1u8;
        let mut last_greater1_flag = false;

        // Decode greater-1 flags (up to 8)
        let mut first_g1_idx: Option<u8> = None;
        let mut g1_positions = [false; 16]; // Track which positions have g1=1
        let max_g1 = (num_coeffs as usize).min(8);
        let mut g1_count = 0;

        // Track which coefficients need remaining level decoding
        let mut needs_remaining = [false; 16];

        for n in (0..=start_pos).rev() {
            if !coeff_flags[n as usize] {
                continue;
            }

            if g1_count >= max_g1 {
                // Beyond first 8: base=1, always needs remaining for values > 1
                needs_remaining[n as usize] = true;
                continue;
            }

            // Update greater1Ctx BEFORE decoding, using PREVIOUS flag
            // (skip for first coefficient in subblock)
            // Per libde265: greater1Ctx increments without clamping (clamped in ctx calc)
            if g1_count > 0 && greater1_ctx > 0 {
                if last_greater1_flag {
                    greater1_ctx = 0;
                } else {
                    greater1_ctx += 1;
                }
            }

            // Per libde265: ctxSet = c1 (HM algorithm)
            let g1 = decode_coeff_greater1_flag(cabac, ctx, c_idx, c1, greater1_ctx)?;
            last_greater1_flag = g1;

            if g1 {
                coeff_values[n as usize] = 2;
                g1_positions[n as usize] = true;
                c1 = 0; // Track that a greater1_flag was 1
                if first_g1_idx.is_none() {
                    first_g1_idx = Some(n);
                } else {
                    // Non-first g1=1: base=2, needs remaining for values > 2
                    needs_remaining[n as usize] = true;
                }
            }
            // Note: c1 stays 1 if no g1=1 yet, becomes 0 when any g1=1
            g1_count += 1;
        }

        // Decode greater-2 flag (only for first coefficient with greater-1)
        // Per libde265: uses ctxSet from last greater1 decode, which is c1
        if let Some(g1_idx) = first_g1_idx {
            let g2 = decode_coeff_greater2_flag(cabac, ctx, c_idx, c1)?;
            if g2 {
                coeff_values[g1_idx as usize] = 3;
                needs_remaining[g1_idx as usize] = true;
            }
        }

        // Find first and last significant positions in this sub-block
        let mut first_sig_pos = None;
        let mut last_sig_pos = None;
        for n in 0..=start_pos {
            if coeff_flags[n as usize] {
                if first_sig_pos.is_none() {
                    first_sig_pos = Some(n);
                }
                last_sig_pos = Some(n);
            }
        }
        let first_sig_pos = first_sig_pos.unwrap_or(0);
        let last_sig_pos = last_sig_pos.unwrap_or(start_pos);

        // Determine if sign is hidden for this sub-block
        // Per H.265 9.3.4.3: sign is hidden if:
        // - sign_data_hiding_enabled_flag is true
        // - cu_transquant_bypass_flag is false
        // - lastScanPos - firstScanPos > 3
        // Note: also disabled for RDPCM modes (not implemented here)
        let sign_hidden =
            sign_data_hiding_enabled && !cu_transquant_bypass && (last_sig_pos - first_sig_pos) > 3;

        // Decode signs (bypass mode)
        // Following libde265's approach: decode signs in coefficient order (high scan pos to low)
        // The LAST coefficient (at first_sig_pos) has its sign hidden when sign_hidden=true
        //
        // Build list of significant positions in reverse scan order (high to low)
        let mut sig_positions = [0u8; 16];
        let mut n_sig = 0usize;
        for n in (0..=start_pos).rev() {
            if coeff_flags[n as usize] {
                sig_positions[n_sig] = n;
                n_sig += 1;
            }
        }

        // Decode signs following libde265 order:
        // - Decode signs for coefficients 0 to n_sig-2 (high scan pos to low)
        // - For coefficient at n_sig-1 (lowest scan pos, first in scan order):
        //   - If sign_hidden: skip decoding, will infer from parity later
        //   - Otherwise: decode the sign
        //
        // This matches the H.265 spec and libde265: the FIRST coefficient in
        // scanning order (lowest scan position) has its sign hidden.
        let mut coeff_signs = [0u8; 16];
        for (i, sign) in coeff_signs[..n_sig.saturating_sub(1)]
            .iter_mut()
            .enumerate()
        {
            *sign = cabac.decode_bypass()?;
            let _ = i; // Silence unused warning if needed
        }
        // Last coefficient (lowest scan pos, first in scan order)
        if n_sig > 0 && !sign_hidden {
            coeff_signs[n_sig - 1] = cabac.decode_bypass()?;
        }
        // If sign_hidden, coeff_signs[n_sig-1] stays 0 (will be inferred later)

        // Decode remaining levels for all coefficients that need it
        // Rice parameter starts at 0 and is updated adaptively
        let mut rice_param = 0u8;

        // Decode remaining levels for coefficients that need it
        for n in (0..=start_pos).rev() {
            if coeff_flags[n as usize] && needs_remaining[n as usize] {
                let base = coeff_values[n as usize];

                let (remaining, new_rice) =
                    decode_coeff_abs_level_remaining(cabac, rice_param, base)?;

                rice_param = new_rice;
                let final_value = base + remaining;
                coeff_values[n as usize] = final_value;
            }
        }

        // Apply signs and compute sum for parity inference
        // At this point coeff_values[] are all positive (absolute values)
        let mut sum_abs_level = 0i32;

        for (i, &pos) in sig_positions[..n_sig].iter().enumerate() {
            let pos = pos as usize;
            if coeff_signs[i] != 0 {
                coeff_values[pos] = -coeff_values[pos];
            }
            sum_abs_level += coeff_values[pos] as i32;

            // Infer hidden sign at the last coefficient (first in scan order)
            // Per H.265: if sum of signed coefficients is odd, flip the hidden sign
            if i == n_sig - 1 && sign_hidden && (sum_abs_level & 1) != 0 {
                coeff_values[pos] = -coeff_values[pos];
            }
        }

        // Store coefficients in buffer
        for (n, &(px, py)) in scan_pos.iter().enumerate() {
            if coeff_flags[n] {
                let x = sb_x as usize * 4 + px as usize;
                let y = sb_y as usize * 4 + py as usize;

                buffer.set(x, y, coeff_values[n]);

                // Invariant check: track large coefficients (indicates CABAC desync)
                if coeff_values[n].abs() > 500 {
                    let (byte_pos, _, _) = cabac.get_position();
                    eprintln!(
                        "LARGE COEFF: call#{} c_idx={} pos=({},{}) sb=({},{}) n={} val={} byte={}",
                        residual_call_num, c_idx, x, y, sb_x, sb_y, n, coeff_values[n], byte_pos
                    );
                    debug::track_large_coeff(byte_pos);
                }
            }
        }
    }

    let _ = (init_range, init_offset); // Used for future debugging if needed
    Ok(buffer)
}

/// Get sub-block scan order
/// For TU of size 2^log2_size, sub-blocks are arranged in a grid of size 2^(log2_size-2)
/// This function returns the diagonal scan order for sub-blocks
fn get_scan_sub_block(log2_size: u8, order: ScanOrder) -> &'static [(u8, u8)] {
    // Sub-block scan tables
    // Note: The order here must match how coefficients are accessed in the decoder
    static SCAN_1X1: [(u8, u8); 1] = [(0, 0)];
    static SCAN_2X2_DIAG: [(u8, u8); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
    static SCAN_4X4_DIAG: [(u8, u8); 16] = [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
        (3, 1),
        (2, 2),
        (1, 3),
        (3, 2),
        (2, 3),
        (3, 3),
    ];
    static SCAN_8X8_DIAG: [(u8, u8); 64] = [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
        (4, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 4),
        (5, 0),
        (4, 1),
        (3, 2),
        (2, 3),
        (1, 4),
        (0, 5),
        (6, 0),
        (5, 1),
        (4, 2),
        (3, 3),
        (2, 4),
        (1, 5),
        (0, 6),
        (7, 0),
        (6, 1),
        (5, 2),
        (4, 3),
        (3, 4),
        (2, 5),
        (1, 6),
        (0, 7),
        (7, 1),
        (6, 2),
        (5, 3),
        (4, 4),
        (3, 5),
        (2, 6),
        (1, 7),
        (7, 2),
        (6, 3),
        (5, 4),
        (4, 5),
        (3, 6),
        (2, 7),
        (7, 3),
        (6, 4),
        (5, 5),
        (4, 6),
        (3, 7),
        (7, 4),
        (6, 5),
        (5, 6),
        (4, 7),
        (7, 5),
        (6, 6),
        (5, 7),
        (7, 6),
        (6, 7),
        (7, 7),
    ];

    let _ = order; // TODO: Use horizontal/vertical scan when appropriate
    match log2_size {
        2 => &SCAN_1X1,
        3 => &SCAN_2X2_DIAG,
        4 => &SCAN_4X4_DIAG,
        5 => &SCAN_8X8_DIAG,
        _ => &SCAN_1X1,
    }
}

/// Find position in scan order
fn find_scan_pos(scan: &[(u8, u8)], x: u32, y: u32, _width: u32) -> u32 {
    for (i, &(sx, sy)) in scan.iter().enumerate() {
        if sx as u32 == x && sy as u32 == y {
            return i as u32;
        }
    }
    0
}

/// Find position in 4x4 scan order
fn find_scan_pos_4x4(scan: &[(u8, u8); 16], x: u8, y: u8) -> u8 {
    for (i, &(sx, sy)) in scan.iter().enumerate() {
        if sx == x && sy == y {
            return i as u8;
        }
    }
    0
}

/// Decode last significant coefficient position
fn decode_last_sig_coeff_pos(
    cabac: &mut CabacDecoder<'_>,
    ctx: &mut [ContextModel],
    log2_size: u8,
    c_idx: u8,
) -> Result<(u32, u32)> {
    // Decode prefix
    let x_prefix = decode_last_sig_coeff_prefix(cabac, ctx, log2_size, c_idx, true)?;
    let y_prefix = decode_last_sig_coeff_prefix(cabac, ctx, log2_size, c_idx, false)?;

    // Decode suffix if needed
    let x = if x_prefix > 3 {
        let n_bits = (x_prefix >> 1) - 1;
        let suffix = cabac.decode_bypass_bits(n_bits as u8)?;
        ((2 + (x_prefix & 1)) << n_bits) + suffix
    } else {
        x_prefix
    };

    let y = if y_prefix > 3 {
        let n_bits = (y_prefix >> 1) - 1;
        let suffix = cabac.decode_bypass_bits(n_bits as u8)?;
        ((2 + (y_prefix & 1)) << n_bits) + suffix
    } else {
        y_prefix
    };

    Ok((x, y))
}

/// Decode last_significant_coeff prefix
fn decode_last_sig_coeff_prefix(
    cabac: &mut CabacDecoder<'_>,
    ctx: &mut [ContextModel],
    log2_size: u8,
    c_idx: u8,
    is_x: bool,
) -> Result<u32> {
    let ctx_base = if is_x {
        context::LAST_SIG_COEFF_X_PREFIX
    } else {
        context::LAST_SIG_COEFF_Y_PREFIX
    };

    // Context offset and shift based on component and size
    // Per H.265 spec (matches libde265):
    // Luma: ctxOffset = 3*(log2Size-2) + ((log2Size-1)>>2), ctxShift = (log2Size+1)>>2
    // Chroma: ctxOffset = 15, ctxShift = log2Size-2
    let (ctx_offset, ctx_shift) = if c_idx == 0 {
        // Luma
        let offset = 3 * (log2_size as usize - 2) + ((log2_size as usize - 1) >> 2);
        let shift = (log2_size + 1) >> 2;
        (offset, shift)
    } else {
        // Chroma
        (15, log2_size - 2)
    };

    let max_prefix = (log2_size << 1) - 1;

    let mut prefix = 0u32;
    while prefix < max_prefix as u32 {
        let ctx_idx = ctx_base + ctx_offset + ((prefix as usize) >> ctx_shift as usize).min(3);
        let bin = cabac.decode_bin(&mut ctx[ctx_idx])?;
        if bin == 0 {
            break;
        }
        prefix += 1;
    }

    Ok(prefix)
}

/// Decode coded_sub_block_flag (simplified - without neighbor context)
/// This always uses context 0 or 2 (for chroma), which is a simplification
fn decode_coded_sub_block_flag_simple(
    cabac: &mut CabacDecoder<'_>,
    ctx: &mut [ContextModel],
    c_idx: u8,
) -> Result<bool> {
    // Simplified: always use ctx 0 or 2
    let ctx_idx = context::CODED_SUB_BLOCK_FLAG + if c_idx > 0 { 2 } else { 0 };
    Ok(cabac.decode_bin(&mut ctx[ctx_idx])? != 0)
}

/// Decode coded_sub_block_flag
/// Per H.265 section 9.3.4.2.4, context depends on neighbor coded_sub_block_flags:
/// - csbfCtx = (csbf_right | csbf_below) ? 1 : 0
/// - ctx_idx = base + csbfCtx + c_idx_offset
fn decode_coded_sub_block_flag(
    cabac: &mut CabacDecoder<'_>,
    ctx: &mut [ContextModel],
    c_idx: u8,
    csbf_neighbors: u8, // bit 0 = right, bit 1 = below
) -> Result<bool> {
    // Context offset: 0-1 for luma, 2-3 for chroma
    // csbfCtx = 1 if either neighbor is coded, else 0
    let csbf_ctx = if csbf_neighbors != 0 { 1 } else { 0 };
    let ctx_idx = context::CODED_SUB_BLOCK_FLAG + csbf_ctx + if c_idx > 0 { 2 } else { 0 };
    Ok(cabac.decode_bin(&mut ctx[ctx_idx])? != 0)
}

/// Context index map for 4x4 TU sig_coeff_flag (H.265 Table 9-41)
/// Maps position (y*4 + x) to context index
static CTX_IDX_MAP_4X4: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];

/// Calculate sig_coeff_flag context index
///
/// Per H.265 section 9.3.4.2.5, the context depends on:
/// - Position within sub-block (xP, yP)
/// - Sub-block position within TU (xS, yS)
/// - coded_sub_block_flag of neighbors (prevCsbf)
/// - Component (luma/chroma)
/// - TU size and scan order
///
/// Parameters:
/// - x_c, y_c: coefficient position within TU
/// - log2_size: TU size (2=4x4, 3=8x8, 4=16x16, 5=32x32)
/// - c_idx: component (0=Y, 1=Cb, 2=Cr)
/// - scan_idx: scan order (0=diagonal, 1=horizontal, 2=vertical)
/// - prev_csbf: coded_sub_block_flag of neighbors (bit0=below, bit1=right - our convention)
fn calc_sig_coeff_flag_ctx(
    x_c: u8,
    y_c: u8,
    log2_size: u8,
    c_idx: u8,
    scan_idx: u8,
    prev_csbf: u8,
) -> usize {
    let sb_width = 1u8 << (log2_size - 2);

    let sig_ctx = if sb_width == 1 {
        // 4x4 TU: use lookup table
        CTX_IDX_MAP_4X4[(y_c as usize * 4 + x_c as usize).min(15)]
    } else if x_c == 0 && y_c == 0 {
        // DC coefficient
        0
    } else {
        // Sub-block and position within sub-block
        let x_s = x_c >> 2;
        let y_s = y_c >> 2;
        let x_p = x_c & 3;
        let y_p = y_c & 3;

        // Base context from position and neighbor flags
        // Per H.265 spec Table 9-43: prevCsbf bit0=below, bit1=right
        let mut ctx = match prev_csbf {
            0 => {
                // No coded neighbors: context based on position sum
                if x_p + y_p >= 3 {
                    0
                } else if x_p + y_p > 0 {
                    1
                } else {
                    2
                }
            }
            1 => {
                // Below neighbor coded (bit0=1): context based on y position
                if y_p == 0 {
                    2
                } else if y_p == 1 {
                    1
                } else {
                    0
                }
            }
            2 => {
                // Right neighbor coded (bit1=1): context based on x position
                if x_p == 0 {
                    2
                } else if x_p == 1 {
                    1
                } else {
                    0
                }
            }
            _ => {
                // Both neighbors coded
                2
            }
        };

        if c_idx == 0 {
            // Luma
            if x_s + y_s > 0 {
                ctx += 3; // Not first sub-block
            }

            // Size-dependent offset
            if sb_width == 2 {
                // 8x8 TU
                ctx += if scan_idx == 0 { 9 } else { 15 };
            } else {
                // 16x16 or 32x32 TU
                ctx += 21;
            }
        } else {
            // Chroma
            if sb_width == 2 {
                // 8x8 TU
                ctx += 9;
            } else {
                // 16x16 or larger TU
                ctx += 12;
            }
        }

        ctx
    };

    // Final context index
    context::SIG_COEFF_FLAG + if c_idx > 0 { 27 } else { 0 } + sig_ctx as usize
}

/// Decode significant_coeff_flag with proper context derivation
#[allow(clippy::too_many_arguments)]
fn decode_sig_coeff_flag(
    cabac: &mut CabacDecoder<'_>,
    ctx: &mut [ContextModel],
    c_idx: u8,
    pos: u8,
    log2_size: u8,
    scan_idx: u8,
    sb_x: u8,
    sb_y: u8,
    prev_csbf: u8,
    scan_table: &[(u8, u8); 16],
) -> Result<bool> {
    // Get coefficient position within TU from scan position
    let (x_in_sb, y_in_sb) = scan_table[pos as usize];
    let x_c = sb_x * 4 + x_in_sb;
    let y_c = sb_y * 4 + y_in_sb;

    let ctx_idx = calc_sig_coeff_flag_ctx(x_c, y_c, log2_size, c_idx, scan_idx, prev_csbf);

    Ok(cabac.decode_bin(&mut ctx[ctx_idx])? != 0)
}

/// Decode coeff_abs_level_greater1_flag
/// Per H.265 9.3.4.2.6: context index = ctxSet * 4 + min(greater1Ctx, 3)
/// Plus 16 for chroma (c_idx > 0)
fn decode_coeff_greater1_flag(
    cabac: &mut CabacDecoder<'_>,
    ctx: &mut [ContextModel],
    c_idx: u8,
    ctx_set: u8,
    greater1_ctx: u8,
) -> Result<bool> {
    let ctx_idx = context::COEFF_ABS_LEVEL_GREATER1_FLAG
        + if c_idx > 0 { 16 } else { 0 }
        + (ctx_set as usize) * 4
        + (greater1_ctx as usize).min(3);
    Ok(cabac.decode_bin(&mut ctx[ctx_idx])? != 0)
}

/// Decode coeff_abs_level_greater2_flag
/// Per H.265: context index = ctxSet + (c_idx > 0 ? 4 : 0)
fn decode_coeff_greater2_flag(
    cabac: &mut CabacDecoder<'_>,
    ctx: &mut [ContextModel],
    c_idx: u8,
    ctx_set: u8,
) -> Result<bool> {
    let ctx_idx =
        context::COEFF_ABS_LEVEL_GREATER2_FLAG + if c_idx > 0 { 4 } else { 0 } + ctx_set as usize;
    Ok(cabac.decode_bin(&mut ctx[ctx_idx])? != 0)
}

/// Decode coeff_abs_level_remaining (Golomb-Rice with adaptive rice parameter)
/// Returns (value, updated_rice_param)
fn decode_coeff_abs_level_remaining(
    cabac: &mut CabacDecoder<'_>,
    rice_param: u8,
    base_level: i16,
) -> Result<(i16, u8)> {
    // Decode prefix (unary part)
    let mut prefix = 0u32;
    while cabac.decode_bypass()? != 0 && prefix < 32 {
        prefix += 1;
    }

    let value = if prefix <= 3 {
        // TR part only: value = (prefix << rice_param) + suffix
        let suffix = if rice_param > 0 {
            cabac.decode_bypass_bits(rice_param)?
        } else {
            0
        };
        ((prefix << rice_param) + suffix) as i16
    } else {
        // EGk part: suffix bits = prefix - 3 + rice_param
        let suffix_bits = (prefix - 3 + rice_param as u32) as u8;
        let suffix = cabac.decode_bypass_bits(suffix_bits)?;
        // value = (((1 << (prefix-3)) + 3 - 1) << rice_param) + suffix
        let base = ((1u32 << (prefix - 3)) + 2) << rice_param;
        (base + suffix) as i16
    };

    // Update rice parameter: if baseLevel + value > 3 * (1 << rice_param), increase
    let threshold = 3 * (1 << rice_param);
    let new_rice_param = if (base_level.unsigned_abs() as u32 + value as u32) > threshold {
        (rice_param + 1).min(4)
    } else {
        rice_param
    };

    Ok((value, new_rice_param))
}
