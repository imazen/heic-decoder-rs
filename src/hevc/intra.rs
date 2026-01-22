//! Intra prediction for HEVC
//!
//! Implements the 35 intra prediction modes:
//! - Mode 0: Planar (smooth bilinear interpolation)
//! - Mode 1: DC (average of reference samples)
//! - Modes 2-34: Angular (directional prediction)

use super::picture::DecodedFrame;
use super::slice::IntraPredMode;

/// Maximum block size for intra prediction
pub const MAX_INTRA_PRED_BLOCK_SIZE: usize = 64;

/// Intra prediction angle table (H.265 Table 8-4)
/// Index 0-1 are placeholders, modes 2-34 have actual angles
pub static INTRA_PRED_ANGLE: [i16; 35] = [
    0, 0, // modes 0, 1 (planar, DC)
    32, 26, 21, 17, 13, 9, 5, 2, // modes 2-9
    0, // mode 10 (horizontal)
    -2, -5, -9, -13, -17, -21, -26, // modes 11-17
    -32, // mode 18 (diagonal down-left)
    -26, -21, -17, -13, -9, -5, -2, // modes 19-25
    0, // mode 26 (vertical)
    2, 5, 9, 13, 17, 21, 26, // modes 27-33
    32, // mode 34 (diagonal down-right)
];

/// Inverse angle table for negative angles (modes 11-17 and 19-25)
/// Used to extend reference samples for negative angle prediction
pub static INV_ANGLE: [i32; 15] = [
    -4096, -1638, -910, -630, -482, -390, -315, // modes 11-17
    -256,  // mode 18
    -315, -390, -482, -630, -910, -1638, -4096, // modes 19-25
];

/// Get inverse angle for a mode (for negative angle modes only)
fn get_inv_angle(mode: u8) -> i32 {
    if (11..=25).contains(&mode) {
        INV_ANGLE[(mode - 11) as usize]
    } else {
        0
    }
}

/// Perform intra prediction for a block
pub fn predict_intra(
    frame: &mut DecodedFrame,
    x: u32,
    y: u32,
    log2_size: u8,
    mode: IntraPredMode,
    c_idx: u8, // 0=Y, 1=Cb, 2=Cr
) {
    let size = 1u32 << log2_size;

    // Get reference samples (border pixels)
    let mut border = [0i32; 4 * MAX_INTRA_PRED_BLOCK_SIZE + 1];
    let border_center = 2 * MAX_INTRA_PRED_BLOCK_SIZE;

    fill_border_samples(frame, x, y, size, c_idx, &mut border, border_center);

    // DEBUG: Track order and print border samples for chroma blocks near the problem area
    static PRED_SEQ: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    let debug_output = c_idx == 1 && y == 0 && (16..=36).contains(&x);
    let seq = if debug_output {
        PRED_SEQ.fetch_add(1, core::sync::atomic::Ordering::Relaxed)
    } else {
        0
    };

    // Print BEFORE prediction to see what border samples are read
    if c_idx == 1 && y == 0 && (20..=28).contains(&x) {
        let left_samples: Vec<i32> = (0..size as usize).map(|i| border[border_center - 1 - i]).collect();
        let actual_frame_val = frame.get_cb(x.saturating_sub(1), y);
        eprintln!("DEBUG Cb[{}]: at ({},{}) BEFORE predict: border_left[0]={} frame.get_cb({},0)={}",
            seq, x, y, left_samples[0], x.saturating_sub(1), actual_frame_val);
    }

    // DEBUG: Print detailed info for Cr around the corruption point (x=104-112)
    if c_idx == 2 && y == 0 && (100..=116).contains(&x) {
        let left_samples: Vec<i32> = (0..size.min(8) as usize).map(|i| border[border_center - 1 - i]).collect();
        let top_samples: Vec<i32> = (0..size.min(8) as usize).map(|i| border[border_center + 1 + i]).collect();
        let top_left = border[border_center];
        let left_frame_val = if x > 0 { frame.get_cr(x - 1, 0) } else { 0 };
        eprintln!("DEBUG Cr: at ({},{}) size={} mode={:?}", x, y, size, mode);
        eprintln!("  border: top_left={} left={:?} top={:?}", top_left, left_samples, top_samples);
        eprintln!("  frame.get_cr({},0)={}", x.saturating_sub(1), left_frame_val);
    }

    // Apply prediction based on mode
    match mode {
        IntraPredMode::Planar => {
            predict_planar(frame, x, y, size, c_idx, &border, border_center);
        }
        IntraPredMode::Dc => {
            predict_dc(frame, x, y, size, c_idx, &border, border_center);
        }
        _ => {
            let mode_val = mode.as_u8();
            predict_angular(frame, x, y, size, c_idx, mode_val, &border, border_center);
        }
    }

    // DEBUG: Print predicted values for first block
    if x == 0 && y == 0 && c_idx == 0 {
        eprintln!("DEBUG: predicted Y values at (0,0):");
        for py in 0..size.min(4) {
            let row: Vec<u16> = (0..size.min(4)).map(|px| frame.get_y(x + px, y + py)).collect();
            eprintln!("  {:?}", row);
        }
    }

    // DEBUG: Print Cb output after prediction for problem area
    if debug_output {
        // Print the right edge of the block (which will be read by the next block)
        let right_edge: Vec<u16> = (0..size).map(|py| frame.get_cb(x + size - 1, y + py)).collect();
        eprintln!("DEBUG Cb: predicted at ({},{}) size={} mode={:?} right_edge={:?}",
            x, y, size, mode, right_edge);
    }
}

/// Fill border samples from neighboring pixels
fn fill_border_samples(
    frame: &DecodedFrame,
    x: u32,
    y: u32,
    size: u32,
    c_idx: u8,
    border: &mut [i32],
    center: usize,
) {
    // Border layout (indexed from center):
    //   border[-2*size .. -1] = left samples (bottom to top)
    //   border[0] = top-left corner
    //   border[1 .. 2*size] = top samples (left to right)

    let (frame_w, frame_h) = if c_idx == 0 {
        (frame.width, frame.height)
    } else {
        // Chroma is half resolution for 4:2:0
        (frame.width / 2, frame.height / 2)
    };

    // Check availability of neighbors
    let avail_left = x > 0;
    let avail_top = y > 0;
    let avail_top_left = avail_left && avail_top;
    let avail_top_right = avail_top && (x + size * 2) <= frame_w;
    let avail_bottom_left = avail_left && (y + size * 2) <= frame_h;

    // Fill with default value if no neighbors available
    let default_val = 1i32 << (frame.bit_depth - 1);

    // Top-left corner
    if avail_top_left {
        border[center] = get_sample(frame, x - 1, y - 1, c_idx) as i32;
    } else if avail_top {
        border[center] = get_sample(frame, x, y - 1, c_idx) as i32;
    } else if avail_left {
        border[center] = get_sample(frame, x - 1, y, c_idx) as i32;
    } else {
        border[center] = default_val;
    }

    // Top samples (border[1..size] and border[size+1..2*size] for top-right)
    for i in 0..size {
        if avail_top {
            border[center + 1 + i as usize] = get_sample(frame, x + i, y - 1, c_idx) as i32;
        } else {
            border[center + 1 + i as usize] = border[center];
        }
    }

    // Top-right samples
    for i in size..(2 * size) {
        if avail_top_right && (x + i) < frame_w {
            border[center + 1 + i as usize] = get_sample(frame, x + i, y - 1, c_idx) as i32;
        } else if avail_top {
            // Replicate last available top sample
            border[center + 1 + i as usize] = border[center + size as usize];
        } else {
            border[center + 1 + i as usize] = border[center];
        }
    }

    // Left samples (border[-1..-size] and border[-size-1..-2*size] for bottom-left)
    for i in 0..size {
        if avail_left {
            border[center - 1 - i as usize] = get_sample(frame, x - 1, y + i, c_idx) as i32;
        } else {
            border[center - 1 - i as usize] = border[center];
        }
    }

    // Bottom-left samples
    for i in size..(2 * size) {
        if avail_bottom_left && (y + i) < frame_h {
            border[center - 1 - i as usize] = get_sample(frame, x - 1, y + i, c_idx) as i32;
        } else if avail_left {
            // Replicate last available left sample
            border[center - 1 - i as usize] = border[center - size as usize];
        } else {
            border[center - 1 - i as usize] = border[center];
        }
    }

    // Reference sample substitution (H.265 8.4.4.2.2)
    // If any sample is unavailable, substitute from available samples
    // NOTE: This uses 0 as a sentinel for "unavailable" which is technically a bug
    // since 0 is a valid sample value. However, removing this breaks the decoder
    // due to interactions with how the border array is initialized.
    reference_sample_substitution(border, center, size as usize);
}

/// Substitute unavailable reference samples (H.265 8.4.4.2.2)
fn reference_sample_substitution(border: &mut [i32], center: usize, size: usize) {
    // Find first available sample
    let mut first_avail = None;

    // Search from bottom-left to top-right
    for i in (0..(2 * size)).rev() {
        if border[center - 1 - i] != 0 {
            first_avail = Some(border[center - 1 - i]);
            break;
        }
    }

    if first_avail.is_none() && border[center] != 0 {
        first_avail = Some(border[center]);
    }

    if first_avail.is_none() {
        for i in 0..(2 * size) {
            if border[center + 1 + i] != 0 {
                first_avail = Some(border[center + 1 + i]);
                break;
            }
        }
    }

    // Substitute unavailable samples with first available
    let val = first_avail.unwrap_or(1 << 7); // Default to mid-gray if all unavailable

    for i in 0..(2 * size) {
        if border[center - 1 - i] == 0 {
            border[center - 1 - i] = val;
        }
    }
    if border[center] == 0 {
        border[center] = val;
    }
    for i in 0..(2 * size) {
        if border[center + 1 + i] == 0 {
            border[center + 1 + i] = val;
        }
    }
}

/// Get a sample from the frame
fn get_sample(frame: &DecodedFrame, x: u32, y: u32, c_idx: u8) -> u16 {
    match c_idx {
        0 => frame.get_y(x, y),
        1 => frame.get_cb(x, y),
        2 => frame.get_cr(x, y),
        _ => 0,
    }
}

/// Set a sample in the frame
fn set_sample(frame: &mut DecodedFrame, x: u32, y: u32, c_idx: u8, value: u16) {
    // DEBUG: Track Cr writes at y=0 for x=104-111 to find corruption
    if c_idx == 2 && y == 0 && (104..=111).contains(&x) {
        let old_val = frame.get_cr(x, y);
        eprintln!("DEBUG: set_cr({},{}) = {} (was {})", x, y, value, old_val);
    }

    match c_idx {
        0 => frame.set_y(x, y, value),
        1 => frame.set_cb(x, y, value),
        2 => frame.set_cr(x, y, value),
        _ => {}
    }
}

/// Planar prediction (mode 0) - H.265 8.4.4.2.4
fn predict_planar(
    frame: &mut DecodedFrame,
    x: u32,
    y: u32,
    size: u32,
    c_idx: u8,
    border: &[i32],
    center: usize,
) {
    let n = size as i32;
    let log2_size = (size as f32).log2() as u32;

    for py in 0..size {
        for px in 0..size {
            let px_i = px as i32;
            let py_i = py as i32;

            // Planar formula:
            // pred = ((nT-1-x)*border[-1-y] + (x+1)*border[nT+1] +
            //         (nT-1-y)*border[1+x] + (y+1)*border[-1-nT] + nT) >> (log2(nT)+1)
            let left = border[center - 1 - py as usize];
            let right = border[center + 1 + size as usize]; // border[nT+1]
            let top = border[center + 1 + px as usize];
            let bottom = border[center - 1 - size as usize]; // border[-1-nT]

            let pred = ((n - 1 - px_i) * left
                + (px_i + 1) * right
                + (n - 1 - py_i) * top
                + (py_i + 1) * bottom
                + n)
                >> (log2_size + 1);

            let value = pred.clamp(0, (1 << frame.bit_depth) - 1) as u16;
            set_sample(frame, x + px, y + py, c_idx, value);
        }
    }
}

/// DC prediction (mode 1) - H.265 8.4.4.2.5
fn predict_dc(
    frame: &mut DecodedFrame,
    x: u32,
    y: u32,
    size: u32,
    c_idx: u8,
    border: &[i32],
    center: usize,
) {
    let n = size as i32;
    let log2_size = (size as f32).log2() as u32;

    // Calculate DC value as average of top and left samples
    let mut dc_val = 0i32;
    for i in 0..size {
        dc_val += border[center + 1 + i as usize]; // top
        dc_val += border[center - 1 - i as usize]; // left
    }
    dc_val = (dc_val + n) >> (log2_size + 1);

    let max_val = (1 << frame.bit_depth) - 1;

    // Apply DC filtering for luma and small blocks
    if c_idx == 0 && size < 32 {
        // Corner pixel: average of corner neighbors and 2*DC
        let corner = (border[center - 1] + 2 * dc_val + border[center + 1] + 2) >> 2;
        set_sample(frame, x, y, c_idx, corner.clamp(0, max_val) as u16);

        // Top edge: blend top border with DC
        for px in 1..size {
            let pred = (border[center + 1 + px as usize] + 3 * dc_val + 2) >> 2;
            set_sample(frame, x + px, y, c_idx, pred.clamp(0, max_val) as u16);
        }

        // Left edge: blend left border with DC
        for py in 1..size {
            let pred = (border[center - 1 - py as usize] + 3 * dc_val + 2) >> 2;
            set_sample(frame, x, y + py, c_idx, pred.clamp(0, max_val) as u16);
        }

        // Interior: pure DC
        let dc_u16 = dc_val.clamp(0, max_val) as u16;
        for py in 1..size {
            for px in 1..size {
                set_sample(frame, x + px, y + py, c_idx, dc_u16);
            }
        }
    } else {
        // No filtering: fill entire block with DC value
        let dc_u16 = dc_val.clamp(0, max_val) as u16;
        for py in 0..size {
            for px in 0..size {
                set_sample(frame, x + px, y + py, c_idx, dc_u16);
            }
        }
    }
}

/// Angular prediction (modes 2-34) - H.265 8.4.4.2.6
#[allow(clippy::too_many_arguments)]
fn predict_angular(
    frame: &mut DecodedFrame,
    x: u32,
    y: u32,
    size: u32,
    c_idx: u8,
    mode: u8,
    border: &[i32],
    center: usize,
) {
    let n = size as i32;
    let intra_pred_angle = INTRA_PRED_ANGLE[mode as usize] as i32;

    // Build reference array
    let mut ref_arr = [0i32; 4 * MAX_INTRA_PRED_BLOCK_SIZE + 1];
    let ref_center = 2 * MAX_INTRA_PRED_BLOCK_SIZE;

    let max_val = (1 << frame.bit_depth) - 1;

    if mode >= 18 {
        // Horizontal-ish modes (18-34)
        // Reference is top samples

        // Copy top samples to ref[0..nT]
        for i in 0..=n {
            ref_arr[ref_center + i as usize] = border[center + i as usize];
        }

        if intra_pred_angle < 0 {
            // Negative angle: need to extend reference to the left
            let inv_angle = get_inv_angle(mode);
            let ext = (n * intra_pred_angle) >> 5;

            if ext < -1 {
                for xx in ext..=-1 {
                    // Note: xx is negative, inv_angle is negative for modes 19-25
                    // So xx * inv_angle is positive, giving a positive idx
                    let idx = (xx * inv_angle + 128) >> 8;
                    if idx >= 0 && idx <= (2 * n) {
                        ref_arr[(ref_center as i32 + xx) as usize] = border[(center as i32 - idx) as usize];
                    }
                }
            }
        } else {
            // Positive angle: extend reference to the right
            for xx in (n + 1)..=(2 * n) {
                ref_arr[ref_center + xx as usize] = border[center + xx as usize];
            }
        }

        // Generate prediction
        for py in 0..n {
            for px in 0..n {
                let i_idx = ((py + 1) * intra_pred_angle) >> 5;
                let i_fact = ((py + 1) * intra_pred_angle) & 31;

                let pred = if i_fact != 0 {
                    let idx = (ref_center as i32 + px + i_idx + 1) as usize;
                    ((32 - i_fact) * ref_arr[idx] + i_fact * ref_arr[idx + 1] + 16) >> 5
                } else {
                    let idx = (ref_center as i32 + px + i_idx + 1) as usize;
                    ref_arr[idx]
                };

                set_sample(
                    frame,
                    x + px as u32,
                    y + py as u32,
                    c_idx,
                    pred.clamp(0, max_val) as u16,
                );
            }
        }

        // Boundary filter for mode 26 (vertical)
        if mode == 26 && c_idx == 0 && size < 32 {
            for py in 0..n {
                let pred = border[center + 1]
                    + ((border[center - 1 - py as usize] - border[center]) >> 1);
                set_sample(frame, x, y + py as u32, c_idx, pred.clamp(0, max_val) as u16);
            }
        }
    } else {
        // Vertical-ish modes (2-17)
        // Reference is left samples (mirrored)

        // Copy left samples (negated indices) to ref[0..nT]
        for i in 0..=n {
            ref_arr[ref_center + i as usize] = border[center - i as usize];
        }

        if intra_pred_angle < 0 {
            // Negative angle: extend reference
            let inv_angle = get_inv_angle(mode);
            let ext = (n * intra_pred_angle) >> 5;

            if ext < -1 {
                for xx in ext..=-1 {
                    let idx = (xx * inv_angle + 128) >> 8;
                    if idx >= 0 && idx <= (2 * n) {
                        ref_arr[(ref_center as i32 + xx) as usize] = border[(center as i32 + idx) as usize];
                    }
                }
            }
        } else {
            // Positive angle: extend reference
            for xx in (n + 1)..=(2 * n) {
                ref_arr[ref_center + xx as usize] = border[center - xx as usize];
            }
        }

        // Generate prediction (transposed compared to mode >= 18)
        for py in 0..n {
            for px in 0..n {
                let i_idx = ((px + 1) * intra_pred_angle) >> 5;
                let i_fact = ((px + 1) * intra_pred_angle) & 31;

                let pred = if i_fact != 0 {
                    let idx = (ref_center as i32 + py + i_idx + 1) as usize;
                    ((32 - i_fact) * ref_arr[idx] + i_fact * ref_arr[idx + 1] + 16) >> 5
                } else {
                    let idx = (ref_center as i32 + py + i_idx + 1) as usize;
                    ref_arr[idx]
                };

                set_sample(
                    frame,
                    x + px as u32,
                    y + py as u32,
                    c_idx,
                    pred.clamp(0, max_val) as u16,
                );
            }
        }

        // Boundary filter for mode 10 (horizontal)
        if mode == 10 && c_idx == 0 && size < 32 {
            for px in 0..n {
                let pred = border[center - 1]
                    + ((border[center + 1 + px as usize] - border[center]) >> 1);
                set_sample(frame, x + px as u32, y, c_idx, pred.clamp(0, max_val) as u16);
            }
        }
    }
}

/// Fill MPM (Most Probable Mode) candidate list
pub fn fill_mpm_candidates(
    cand_a: IntraPredMode, // left neighbor mode
    cand_b: IntraPredMode, // above neighbor mode
) -> [IntraPredMode; 3] {
    if cand_a == cand_b {
        if cand_a.as_u8() < 2 {
            // DC or Planar
            [
                IntraPredMode::Planar,
                IntraPredMode::Dc,
                IntraPredMode::Angular26, // Vertical
            ]
        } else {
            // Angular mode
            let mode = cand_a.as_u8();
            let left = 2 + ((mode - 2).wrapping_sub(1) % 32);
            let right = 2 + ((mode - 2) + 1) % 32;
            [
                cand_a,
                IntraPredMode::from_u8(left).unwrap_or(IntraPredMode::Dc),
                IntraPredMode::from_u8(right).unwrap_or(IntraPredMode::Dc),
            ]
        }
    } else {
        // Different modes
        let third = if cand_a != IntraPredMode::Planar && cand_b != IntraPredMode::Planar {
            IntraPredMode::Planar
        } else if cand_a != IntraPredMode::Dc && cand_b != IntraPredMode::Dc {
            IntraPredMode::Dc
        } else {
            IntraPredMode::Angular26
        };
        [cand_a, cand_b, third]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpm_candidates_same_dc() {
        let mpm = fill_mpm_candidates(IntraPredMode::Dc, IntraPredMode::Dc);
        assert_eq!(mpm[0], IntraPredMode::Planar);
        assert_eq!(mpm[1], IntraPredMode::Dc);
        assert_eq!(mpm[2], IntraPredMode::Angular26);
    }

    #[test]
    fn test_mpm_candidates_different() {
        let mpm = fill_mpm_candidates(IntraPredMode::Dc, IntraPredMode::Planar);
        assert_eq!(mpm[0], IntraPredMode::Dc);
        assert_eq!(mpm[1], IntraPredMode::Planar);
        assert_eq!(mpm[2], IntraPredMode::Angular26);
    }

    #[test]
    fn test_intra_angles() {
        // Mode 10 should be horizontal (angle 0)
        assert_eq!(INTRA_PRED_ANGLE[10], 0);
        // Mode 26 should be vertical (angle 0)
        assert_eq!(INTRA_PRED_ANGLE[26], 0);
        // Mode 2 should have positive angle
        assert_eq!(INTRA_PRED_ANGLE[2], 32);
        // Mode 34 should have positive angle
        assert_eq!(INTRA_PRED_ANGLE[34], 32);
    }
}
