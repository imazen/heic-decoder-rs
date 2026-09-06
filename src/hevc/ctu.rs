//! CTU (Coding Tree Unit) and CU (Coding Unit) decoding
//!
//! This module handles the hierarchical quad-tree structure of HEVC:
//! - CTU: Coding Tree Unit (largest block, typically 64x64)
//! - CU: Coding Unit (result of quad-tree split, 8x8 to 64x64)
//! - PU: Prediction Unit (for motion/intra prediction)
//! - TU: Transform Unit (for residual coding)

use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

/// Set to true to enable verbose debug tracing to stderr
const DEBUG_TRACE: bool = false;

/// Set to true to enable WPP-specific debug tracing
#[allow(dead_code)]
const WPP_TRACE: bool = false;

/// How often the per-CTU decode loop polls the cooperative [`enough::Stop`]
/// token for cancellation.
///
/// The check is hoisted to CTU-block granularity (every `N` CTUs, masked with
/// a power-of-two `& (N - 1)`) so it stays out of the innermost per-sample
/// work. At a 64×64 CTB size, 256 CTUs is ~1 megapixel of luma per poll —
/// on a 16384×16384 single-tile frame (65 536 CTUs) that is 256 polls,
/// responsive within single-digit milliseconds while adding a single masked
/// compare per CTU (negligible vs the CABAC + transform + prediction cost of
/// decoding one CTU).
pub(crate) const STOP_CHECK_CTU_INTERVAL: u32 = 256;

/// Debug print macro gated behind DEBUG_TRACE const
macro_rules! debug_trace {
    ($($arg:tt)*) => {
        #[cfg(feature = "std")]
        if DEBUG_TRACE {
            eprintln!($($arg)*);
        }
    };
}

use super::cabac::{CabacDecoder, ContextModel, INIT_VALUES, context};
use super::debug;
use super::inter::{
    self, CollocatedFrame, MergePuParams, MotionVector, MvContext, PbMotion, PbMotionCoding,
    RefPicLists,
};
use super::intra;
use super::mc::{self, ChromaRef, McBlock};
use super::params::{Pps, Sps};
use super::picture::DecodedFrame;
use super::residual::{self, ScanOrder};
use super::sao::SaoMap;
use super::slice::{IntraPredMode, PartMode, PredMode, SliceHeader, SliceType};
use super::transform;
use super::transform_simd::add_residual_block_scalar;
#[cfg(target_arch = "x86_64")]
use super::transform_simd::add_residual_block_v3;
#[cfg(target_arch = "wasm32")]
use super::transform_simd::add_residual_block_wasm128;
#[cfg(target_arch = "aarch64")]
use super::transform_simd_neon::add_residual_block_neon;
use crate::error::HevcError;
use archmage::incant;

use super::Result;
use whereat::at;

/// Global SE counter for syntax element tracing
pub static SE_COUNTER: AtomicU32 = AtomicU32::new(0);
pub const SE_TRACE_LIMIT: u32 = 0;

/// Log a syntax element decode for differential testing.
/// Set SE_TRACE_LIMIT > 0 to enable tracing.
#[allow(clippy::absurd_extreme_comparisons)]
fn se_trace(name: &str, val: i64, cabac: &CabacDecoder) {
    let num = SE_COUNTER.fetch_add(1, Ordering::Relaxed);
    if num < SE_TRACE_LIMIT {
        #[cfg(feature = "std")]
        {
            let (range, _, _) = cabac.get_state_extended();
            let (byte_pos, _, _) = cabac.get_position();
            eprintln!(
                "SE#{} {} val={} range={} byte={}",
                num, name, val, range, byte_pos
            );
        }
    }
    let _ = (name, val, cabac);
}

/// Chroma QP mapping table (H.265 Table 8-10)
/// Maps qPi (0-57) to QpC for 8-bit video
/// Map partition mode to PU rectangles: (x, y, width, height)
/// Returns up to 4 PUs for NxN, 2 for split modes, 1 for 2Nx2N
fn partition_to_pu_list(
    part_mode: PartMode,
    x0: u32,
    y0: u32,
    cb_size: u32,
) -> alloc::vec::Vec<(u32, u32, u32, u32)> {
    let n = cb_size;
    match part_mode {
        PartMode::Part2Nx2N => alloc::vec![(x0, y0, n, n)],
        PartMode::Part2NxN => alloc::vec![(x0, y0, n, n / 2), (x0, y0 + n / 2, n, n / 2)],
        PartMode::PartNx2N => alloc::vec![(x0, y0, n / 2, n), (x0 + n / 2, y0, n / 2, n)],
        PartMode::PartNxN => alloc::vec![
            (x0, y0, n / 2, n / 2),
            (x0 + n / 2, y0, n / 2, n / 2),
            (x0, y0 + n / 2, n / 2, n / 2),
            (x0 + n / 2, y0 + n / 2, n / 2, n / 2),
        ],
        PartMode::Part2NxnU => alloc::vec![(x0, y0, n, n / 4), (x0, y0 + n / 4, n, 3 * n / 4)],
        PartMode::Part2NxnD => {
            alloc::vec![(x0, y0, n, 3 * n / 4), (x0, y0 + 3 * n / 4, n, n / 4)]
        }
        PartMode::PartnLx2N => alloc::vec![(x0, y0, n / 4, n), (x0 + n / 4, y0, 3 * n / 4, n)],
        PartMode::PartnRx2N => {
            alloc::vec![(x0, y0, 3 * n / 4, n), (x0 + 3 * n / 4, y0, n / 4, n)]
        }
    }
}

/// Mark prediction block boundaries for deblocking (H.265 8.7.2.3)
///
/// For inter CUs with non-2Nx2N partitioning, the internal PB boundary must be
/// marked so the deblocking filter can derive boundary strength. These are marked
/// separately from transform block boundaries because the CBF check in bS derivation
/// (bS=1 for non-zero coefficients) only applies at transform block edges.
fn mark_pb_boundaries(
    frame: &mut DecodedFrame,
    part_mode: PartMode,
    x0: u32,
    y0: u32,
    cb_size: u32,
) {
    let half = cb_size / 2;
    let quarter = cb_size / 4;
    match part_mode {
        PartMode::Part2Nx2N => {
            // No internal PB boundary
        }
        PartMode::PartNx2N => {
            // Vertical PB edge at x0 + half
            frame.mark_pb_boundary(x0 + half, y0, cb_size, cb_size, true);
        }
        PartMode::Part2NxN => {
            // Horizontal PB edge at y0 + half
            frame.mark_pb_boundary(x0, y0 + half, cb_size, cb_size, false);
        }
        PartMode::PartNxN => {
            // Both vertical and horizontal PB edges at center
            frame.mark_pb_boundary(x0 + half, y0, cb_size, cb_size, true);
            frame.mark_pb_boundary(x0, y0 + half, cb_size, cb_size, false);
        }
        PartMode::PartnLx2N => {
            // Vertical PB edge at x0 + quarter
            frame.mark_pb_boundary(x0 + quarter, y0, cb_size, cb_size, true);
        }
        PartMode::PartnRx2N => {
            // Vertical PB edge at x0 + 3*quarter
            frame.mark_pb_boundary(x0 + half + quarter, y0, cb_size, cb_size, true);
        }
        PartMode::Part2NxnU => {
            // Horizontal PB edge at y0 + quarter
            frame.mark_pb_boundary(x0, y0 + quarter, cb_size, cb_size, false);
        }
        PartMode::Part2NxnD => {
            // Horizontal PB edge at y0 + 3*quarter
            frame.mark_pb_boundary(x0, y0 + half + quarter, cb_size, cb_size, false);
        }
    }
}

/// Coded block flags for the chroma transform blocks of one transform-tree node.
///
/// Bit 0 is the (only, for 4:2:0 / 4:4:4) chroma TB; bit 1 is the second,
/// vertically stacked TB that `ChromaArrayType == 2` (4:2:2) codes per
/// H.265 7.3.8.8 (`cbf_cb[x0][y0 + (1 << (log2TrafoSize - 1))]`). Same packing
/// as libde265's `cbf_cb`/`cbf_cr`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ChromaCbf {
    cb: u8,
    cr: u8,
}

impl ChromaCbf {
    const NONE: Self = Self { cb: 0, cr: 0 };
    /// Root of the transform tree: 7.3.8.8 reads the depth-0 flags
    /// unconditionally, which the recursion expresses as "parent had residual".
    const ROOT: Self = Self { cb: 1, cr: 1 };

    fn any(self) -> bool {
        (self.cb | self.cr) != 0
    }

    fn bits(self, c_idx: u8) -> u8 {
        if c_idx == 1 { self.cb } else { self.cr }
    }
}

/// H.265 (V2+) Table 8-3: 4:2:2 chroma intra prediction mode.
///
/// With chroma at half width but full height, a luma direction (dx, dy)
/// becomes (dx/2, dy) in the chroma plane, so `IntraPredModeC` is remapped to
/// the angular mode nearest that halved direction. Indexed by the mode
/// derived from `intra_chroma_pred_mode` (Table 8-2). Matches libde265's
/// `map_chroma_422`; pinned by the ADJUST_IPRED_ANGLE_A_RExt_Mitsubishi
/// conformance stream.
const CHROMA_422_MODE_MAP: [u8; 35] = [
    0, 1, 2, 2, 2, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 23, 24, 24, 25, 25,
    26, 27, 27, 28, 28, 29, 29, 30, 31,
];

fn chroma_qp_mapping(qp_i: i32) -> i32 {
    // Table 8-10: qPi to QpC mapping
    // For qPi 0-29, QpC = qPi
    // For qPi 30-57, QpC follows the table
    static CHROMA_QP_TABLE: [i32; 58] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    ];
    CHROMA_QP_TABLE[qp_i.clamp(0, 57) as usize]
}

/// Decoding context for a slice
pub struct SliceContext<'a> {
    /// Sequence parameter set
    pub sps: &'a Sps,
    /// Picture parameter set
    pub pps: &'a Pps,
    /// Slice header
    pub header: &'a SliceHeader,
    /// CABAC decoder
    pub cabac: CabacDecoder<'a>,
    /// Context models
    pub ctx: [ContextModel; context::NUM_CONTEXTS],
    /// Current CTB X position (in CTB units)
    pub ctb_x: u32,
    /// Current CTB Y position (in CTB units)
    pub ctb_y: u32,
    /// Current luma QP value
    pub qp_y: i32,
    /// Current Cb QP value
    pub qp_cb: i32,
    /// Current Cr QP value
    pub qp_cr: i32,
    /// Is CU QP delta coded flag
    pub is_cu_qp_delta_coded: bool,
    /// CU QP delta value
    pub cu_qp_delta: i32,
    /// CU transquant bypass flag
    pub cu_transquant_bypass_flag: bool,
    /// Debug flag for current CTU
    debug_ctu: bool,
    /// MV trace flag (dump all inter PU motion vectors)
    pub mv_trace: bool,
    /// Debug: track chroma prediction calls
    #[allow(dead_code)]
    chroma_pred_count: u32,
    /// CT depth map for split_cu_flag context derivation (indexed by min_cb_size grid)
    ct_depth_map: Vec<u8>,
    /// Width of ct_depth_map in min_cb_size units
    ct_depth_map_stride: u32,
    /// Intra luma mode map (indexed by min_pu_size grid, stores IntraPredMode as u8)
    /// min_pu_size = min_cb_size / 2, to support NxN PU resolution
    intra_mode_map: Vec<u8>,
    /// Width of intra_mode_map in min_pu_size units (also used as PU map stride)
    pub intra_mode_map_stride: u32,
    /// Intra chroma mode map (indexed by min_pu_size grid, stores IntraPredMode as u8)
    intra_chroma_mode_map: Vec<u8>,
    /// Current CU base position (set at decode_coding_unit start)
    cu_base_x: u32,
    cu_base_y: u32,
    /// Current CU log2 size (set at decode_coding_unit start)
    cu_log2_size: u8,
    /// QP map: stores per-CU QPY values (indexed by min_tb_size grid)
    qp_map: Vec<i8>,
    /// Width of QP map in min_tb_size units
    qp_map_stride: u32,
    /// Current QPY for the current quantization group
    current_qpy: i32,
    /// Last QPY from previous quantization group (for prediction)
    last_qpy_in_prev_qg: i32,
    /// Current quantization group position
    current_qg_x: i32,
    current_qg_y: i32,
    /// SAO parameters per CTB
    pub sao_map: SaoMap,
    /// Reusable residual buffer (inverse transform writes all elements, no re-zeroing needed)
    residual_buf: [i16; 1024],
    /// Reusable scaling matrix buffer
    scaling_buf: [u8; 1024],
    /// Rice parameter initialization states (H.265 StatCoeff[0..3])
    /// Persists across TUs within a substream; saved/restored for WPP
    pub stat_coeff: [u8; 4],
    /// Emulation prevention byte positions in slice_data EBSP space.
    /// Used to convert WPP/tile entry point offsets (EBSP-relative per spec
    /// 7.4.7.1) to positions within `slice_data` (RBSP) before seeking.
    ep_byte_positions: Vec<u32>,
    /// Reusable scratch buffer for motion compensation two-pass filtering
    mc_scratch: mc::McScratch,

    // -- Inter prediction state --
    /// Prediction mode map at min_pu_size granularity (Intra/Inter/Skip)
    pub pred_mode_map: Vec<PredMode>,
    /// Motion vector info at min_pu_size granularity
    pub mv_info: Vec<PbMotion>,
    /// CBF (coded block flag) map at 4x4 granularity for deblocking boundary strength
    pub cbf_map: Vec<bool>,
    /// Stride of cbf_map (width / 4)
    pub cbf_map_stride: u32,
    /// Current picture POC
    pub curr_poc: i32,
    /// Reference picture lists (constructed from slice header + DPB)
    pub ref_pic_lists: RefPicLists,
    /// Reference frames from DPB (indexed by DPB slot, not ref list index).
    /// Empty for I-slices. For P/B, populated by VideoDecoder before decode.
    pub ref_frames: Vec<Option<DecodedFrame>>,
    /// Collocated frame data for temporal MVP (owned copy from DPB)
    pub collocated_data: Option<OwnedCollocatedFrame>,

    // -- Tile boundary info (for neighbor availability) --
    /// Tile column boundaries: tile_col_bd[i] = first CTB column of tile column i.
    /// Length = num_tile_cols + 1 (sentinel at end = pic_width_in_ctbs).
    tile_col_bd: Vec<u32>,
    /// Tile row boundaries: tile_row_bd[i] = first CTB row of tile row i.
    /// Length = num_tile_rows + 1 (sentinel at end = pic_height_in_ctbs).
    tile_row_bd: Vec<u32>,
}

/// Owned collocated frame data for temporal MVP (copied from DPB entry)
pub struct OwnedCollocatedFrame {
    /// Motion vectors
    pub mv_info: Vec<PbMotion>,
    /// Prediction modes
    pub pred_mode: Vec<PredMode>,
    /// PU stride
    pub pu_stride: u32,
    /// Min PU size
    pub min_pu_size: u32,
    /// POC
    pub poc: i32,
    /// Reference POCs from this frame's slice
    pub ref_poc: [[i32; inter::MAX_NUM_REF_PICS]; 2],
}

/// Maps that persist across slices within the same picture.
///
/// When multiple slices share a picture, CABAC context derivation for
/// `split_cu_flag`, intra prediction, QP derivation, and deblocking all
/// need data from previously-decoded slices. These maps are extracted
/// from the SliceContext after each slice and injected into the next.
pub struct PictureMaps {
    /// CT depth map for split_cu_flag context (min_cb_size grid)
    pub ct_depth_map: Vec<u8>,
    pub ct_depth_map_stride: u32,
    /// Intra luma prediction mode map (min_pu_size grid)
    pub intra_mode_map: Vec<u8>,
    pub intra_mode_map_stride: u32,
    /// Intra chroma prediction mode map (min_pu_size grid)
    pub intra_chroma_mode_map: Vec<u8>,
    /// Prediction mode map (Intra/Inter/Skip) at min_pu_size grid
    pub pred_mode_map: Vec<PredMode>,
    /// Motion vector info at min_pu_size grid
    pub mv_info: Vec<PbMotion>,
    /// CBF map at 4x4 granularity for deblocking
    pub cbf_map: Vec<bool>,
    pub cbf_map_stride: u32,
    /// QP map at min_tb_size granularity
    pub qp_map: Vec<i8>,
    pub qp_map_stride: u32,
    /// SAO parameters per CTB
    pub sao_map: SaoMap,
}

/// Compute tile column and row boundaries from PPS tile info.
///
/// Returns (col_bd, row_bd) where col_bd[i] is the first CTB column of tile column i,
/// and col_bd[num_tile_cols] = pic_width_in_ctbs (sentinel). Same for rows.
fn compute_tile_boundaries(
    pps: &Pps,
    pic_width_in_ctbs: u32,
    pic_height_in_ctbs: u32,
) -> (Vec<u32>, Vec<u32>) {
    let tile_info = match &pps.tile_info {
        Some(ti) => ti,
        None => return (vec![0, pic_width_in_ctbs], vec![0, pic_height_in_ctbs]),
    };

    let num_cols = tile_info.num_tile_columns_minus1 as u32 + 1;
    let num_rows = tile_info.num_tile_rows_minus1 as u32 + 1;

    let col_bd = if tile_info.uniform_spacing_flag {
        let mut bd = Vec::with_capacity(num_cols as usize + 1);
        for i in 0..=num_cols {
            bd.push((i * pic_width_in_ctbs) / num_cols);
        }
        bd
    } else {
        // Explicit per-column widths (column_widths[i] = width_minus1, each up to
        // u16::MAX). A conformant stream's widths partition the picture exactly, so
        // every boundary is <= pic_width_in_ctbs. A malformed stream can make the
        // running sum explode (20 columns * 65536), which would otherwise drive
        // `build_tile_scan_order`'s inner range `col_bd[tc]..col_bd[tc+1]` into a
        // multi-billion-element push and OOM. Saturate each boundary to
        // pic_width_in_ctbs so the returned boundaries stay monotonic and bounded;
        // the scan-order push count is then capped at pic_width_in_ctbs *
        // pic_height_in_ctbs and out-of-range tiles collapse to empty ranges.
        let mut bd = vec![0u32];
        let mut pos = 0u32;
        for &w in &tile_info.column_widths {
            pos = pos.saturating_add(w as u32 + 1).min(pic_width_in_ctbs);
            bd.push(pos);
        }
        bd.push(pic_width_in_ctbs);
        bd
    };

    let row_bd = if tile_info.uniform_spacing_flag {
        let mut bd = Vec::with_capacity(num_rows as usize + 1);
        for i in 0..=num_rows {
            bd.push((i * pic_height_in_ctbs) / num_rows);
        }
        bd
    } else {
        // See the column_widths note above: clamp each running boundary to
        // pic_height_in_ctbs so a malformed row_heights sum cannot blow up the
        // tile-scan allocation.
        let mut bd = vec![0u32];
        let mut pos = 0u32;
        for &h in &tile_info.row_heights {
            pos = pos.saturating_add(h as u32 + 1).min(pic_height_in_ctbs);
            bd.push(pos);
        }
        bd.push(pic_height_in_ctbs);
        bd
    };

    (col_bd, row_bd)
}

/// Build the tile scan order: returns a vec of (ctb_x, ctb_y) in tile-scan order.
///
/// H.265 6.5.1: tiles are scanned left→right, top→bottom.
/// Within each tile, CTBs are scanned in raster order.
fn build_tile_scan_order(col_bd: &[u32], row_bd: &[u32]) -> Vec<(u32, u32)> {
    let num_cols = col_bd.len() - 1;
    let num_rows = row_bd.len() - 1;
    let mut scan = Vec::new();

    for tr in 0..num_rows {
        for tc in 0..num_cols {
            // Raster scan within tile (tr, tc)
            for cy in row_bd[tr]..row_bd[tr + 1] {
                for cx in col_bd[tc]..col_bd[tc + 1] {
                    scan.push((cx, cy));
                }
            }
        }
    }

    scan
}

/// Get the tile ID for a given CTB position.
/// Tile ID = tile_row * num_tile_cols + tile_col.
fn get_tile_id(col_bd: &[u32], row_bd: &[u32], ctb_x: u32, ctb_y: u32) -> u32 {
    let num_cols = (col_bd.len() - 1) as u32;
    let tc = col_bd
        .windows(2)
        .position(|w| ctb_x >= w[0] && ctb_x < w[1])
        .unwrap_or(0) as u32;
    let tr = row_bd
        .windows(2)
        .position(|w| ctb_y >= w[0] && ctb_y < w[1])
        .unwrap_or(0) as u32;
    tr * num_cols + tc
}

impl<'a> SliceContext<'a> {
    /// Create a new slice context
    pub fn new(
        sps: &'a Sps,
        pps: &'a Pps,
        header: &'a SliceHeader,
        slice_data: &'a [u8],
    ) -> Result<Self> {
        // DEBUG: Print first few bytes of slice data
        debug_trace!(
            "DEBUG: Slice data first 16 bytes: {:02x?}",
            &slice_data[..16.min(slice_data.len())]
        );
        debug_trace!(
            "DEBUG: SPS: {}x{}, ctb_size={}, min_cb_size={}, scaling_list={}",
            sps.pic_width_in_luma_samples,
            sps.pic_height_in_luma_samples,
            sps.ctb_size(),
            1 << sps.log2_min_cb_size(),
            sps.scaling_list_enabled_flag
        );
        debug_trace!(
            "DEBUG: SPS: max_transform_hierarchy_depth_intra={}",
            sps.max_transform_hierarchy_depth_intra
        );
        debug_trace!(
            "DEBUG: SPS: log2_min_tb={}, log2_max_tb={}",
            sps.log2_min_tb_size(),
            sps.log2_max_tb_size()
        );

        let cabac = CabacDecoder::new(slice_data)?;
        let (_range, _offset) = cabac.get_state();
        debug_trace!(
            "DEBUG: CABAC init state: range={}, offset={}",
            _range,
            _offset
        );

        // Initialize context models with correct init table for slice type
        // H.265 9.3.2.2: initType depends on slice_type and cabac_init_flag
        use super::cabac::{INIT_VALUES_B, INIT_VALUES_P};
        let init_table: &[u8; context::NUM_CONTEXTS] = match header.slice_type {
            SliceType::I => &INIT_VALUES,
            SliceType::P => {
                if header.cabac_init_flag {
                    &INIT_VALUES_B
                } else {
                    &INIT_VALUES_P
                }
            }
            SliceType::B => {
                if header.cabac_init_flag {
                    &INIT_VALUES_P
                } else {
                    &INIT_VALUES_B
                }
            }
        };
        let mut ctx = [ContextModel::new(154); context::NUM_CONTEXTS];
        let slice_qp = header.slice_qp_y;

        for (i, init_val) in init_table.iter().enumerate() {
            ctx[i].init(*init_val, slice_qp);
        }

        // Calculate chroma QP values (H.265 Table 8-10 and section 8.6.1)
        // qPi_Cb = qP_Y + pps_cb_qp_offset + slice_cb_qp_offset
        // qPi_Cr = qP_Y + pps_cr_qp_offset + slice_cr_qp_offset
        let qp_i_cb = slice_qp + pps.pps_cb_qp_offset as i32 + header.slice_cb_qp_offset as i32;
        let qp_i_cr = slice_qp + pps.pps_cr_qp_offset as i32 + header.slice_cr_qp_offset as i32;

        // H.265 8.6.1: Table 8-10 applies ONLY to ChromaArrayType == 1 (4:2:0).
        // For 4:2:2 and 4:4:4, QpC = Min(qPi, 51). Below qPi 30 the table is the
        // identity, so this only diverges on high-QP content.
        let chroma_array_type = if sps.separate_colour_plane_flag {
            0
        } else {
            sps.chroma_format_idc
        };
        let map_qp = |qp_i: i32| -> i32 {
            if chroma_array_type == 1 {
                chroma_qp_mapping(qp_i.clamp(0, 57))
            } else {
                qp_i.clamp(0, 51)
            }
        };
        let qp_cb = map_qp(qp_i_cb);
        let qp_cr = map_qp(qp_i_cr);

        debug_trace!(
            "DEBUG: Chroma QP: qp_y={}, qp_cb={}, qp_cr={}",
            slice_qp,
            qp_cb,
            qp_cr
        );
        debug_trace!(
            "DEBUG: sign_data_hiding_enabled_flag={}",
            pps.sign_data_hiding_enabled_flag
        );
        debug_trace!(
            "DEBUG: tiles_enabled={} entropy_coding_sync={}",
            pps.tiles_enabled_flag,
            pps.entropy_coding_sync_enabled_flag
        );
        // Initialize ct_depth_map for split_cu_flag context derivation
        // Map is in units of min_cb_size (typically 8x8)
        let min_cb_size = 1u32 << sps.log2_min_cb_size();
        let ct_depth_map_stride = sps.pic_width_in_luma_samples.div_ceil(min_cb_size);
        let ct_depth_map_height = sps.pic_height_in_luma_samples.div_ceil(min_cb_size);
        let ct_map_size = (ct_depth_map_stride as usize)
            .checked_mul(ct_depth_map_height as usize)
            .ok_or_else(|| at!(HevcError::DecodingError("ct_depth_map size overflow")))?;
        let ct_depth_map =
            try_vec![0xFFu8; ct_map_size].map_err(|_| at!(HevcError::AllocationFailed))?;

        // Intra mode map at min_pu_size granularity (= min_cb_size / 2)
        // This supports NxN partition PU-level resolution
        let min_pu_size = (min_cb_size / 2).max(1);
        let intra_mode_map_stride = sps.pic_width_in_luma_samples.div_ceil(min_pu_size);
        let intra_mode_map_height = sps.pic_height_in_luma_samples.div_ceil(min_pu_size);
        let pu_map_size = (intra_mode_map_stride as usize)
            .checked_mul(intra_mode_map_height as usize)
            .ok_or_else(|| at!(HevcError::DecodingError("pu_map size overflow")))?;
        let intra_mode_map = try_vec![IntraPredMode::Dc.as_u8(); pu_map_size]
            .map_err(|_| at!(HevcError::AllocationFailed))?;
        let intra_chroma_mode_map = try_vec![IntraPredMode::Dc.as_u8(); pu_map_size]
            .map_err(|_| at!(HevcError::AllocationFailed))?;

        // QP map at min_tb_size granularity
        let min_tb_size = 1u32 << sps.log2_min_tb_size();
        let qp_map_stride = sps.pic_width_in_luma_samples.div_ceil(min_tb_size);
        let qp_map_height = sps.pic_height_in_luma_samples.div_ceil(min_tb_size);
        let qp_map_size = (qp_map_stride as usize)
            .checked_mul(qp_map_height as usize)
            .ok_or_else(|| at!(HevcError::DecodingError("qp_map size overflow")))?;
        let qp_map =
            try_vec![slice_qp as i8; qp_map_size].map_err(|_| at!(HevcError::AllocationFailed))?;

        Ok(Self {
            sps,
            pps,
            header,
            cabac,
            ctx,
            ctb_x: 0,
            ctb_y: 0,
            qp_y: slice_qp,
            qp_cb,
            qp_cr,
            is_cu_qp_delta_coded: false,
            cu_qp_delta: 0,
            cu_transquant_bypass_flag: false,
            debug_ctu: false,
            mv_trace: false,
            chroma_pred_count: 0,
            ct_depth_map,
            ct_depth_map_stride,
            intra_mode_map,
            intra_mode_map_stride,
            intra_chroma_mode_map,
            cu_base_x: 0,
            cu_base_y: 0,
            cu_log2_size: 0,
            qp_map,
            qp_map_stride,
            current_qpy: slice_qp,
            last_qpy_in_prev_qg: slice_qp,
            current_qg_x: -1,
            current_qg_y: -1,
            sao_map: SaoMap::new(sps.pic_width_in_ctbs(), sps.pic_height_in_ctbs())?,
            residual_buf: [0i16; 1024],
            scaling_buf: [16u8; 1024],
            stat_coeff: [0u8; 4],
            ep_byte_positions: Vec::new(),
            mc_scratch: mc::McScratch::default(),
            pred_mode_map: try_vec![PredMode::Intra; pu_map_size]
                .map_err(|_| at!(HevcError::AllocationFailed))?,
            mv_info: try_vec![PbMotion::UNAVAILABLE; pu_map_size]
                .map_err(|_| at!(HevcError::AllocationFailed))?,
            cbf_map: {
                let cbf_size = (sps.pic_width_in_luma_samples.div_ceil(4) as usize)
                    .checked_mul(sps.pic_height_in_luma_samples.div_ceil(4) as usize)
                    .ok_or_else(|| at!(HevcError::DecodingError("cbf_map size overflow")))?;
                try_vec![false; cbf_size].map_err(|_| at!(HevcError::AllocationFailed))?
            },
            cbf_map_stride: sps.pic_width_in_luma_samples.div_ceil(4),
            curr_poc: 0,
            ref_pic_lists: RefPicLists::default(),
            ref_frames: Vec::new(),
            collocated_data: None,
            tile_col_bd: if pps.tiles_enabled_flag {
                let pic_w_ctbs = sps.pic_width_in_ctbs();
                let pic_h_ctbs = sps.pic_height_in_ctbs();
                compute_tile_boundaries(pps, pic_w_ctbs, pic_h_ctbs).0
            } else {
                vec![0, sps.pic_width_in_ctbs()]
            },
            tile_row_bd: if pps.tiles_enabled_flag {
                let pic_w_ctbs = sps.pic_width_in_ctbs();
                let pic_h_ctbs = sps.pic_height_in_ctbs();
                compute_tile_boundaries(pps, pic_w_ctbs, pic_h_ctbs).1
            } else {
                vec![0, sps.pic_height_in_ctbs()]
            },
        })
    }

    /// Provide emulation prevention byte positions for this slice's data.
    ///
    /// `positions` must list, in ascending order, the EBSP positions (within
    /// `slice_data`) of the 0x03 bytes that were stripped from the RBSP.
    /// Used by the WPP / tile boundary seek logic to convert the
    /// EBSP-relative entry point offsets from the slice header into RBSP
    /// offsets within `slice_data`.
    pub fn set_ep_byte_positions(&mut self, positions: Vec<u32>) {
        self.ep_byte_positions = positions;
    }

    /// Convert an EBSP byte offset within `slice_data` to the corresponding
    /// RBSP offset, accounting for stripped emulation prevention bytes.
    fn ebsp_offset_to_rbsp(&self, ebsp_offset: u32) -> u32 {
        let count = self
            .ep_byte_positions
            .iter()
            .filter(|&&p| p < ebsp_offset)
            .count() as u32;
        ebsp_offset - count
    }

    /// Inject picture-level maps from a previous slice into this slice context.
    ///
    /// This is called for continuation slices (not the first slice in a picture)
    /// so that CABAC context derivation and neighbor lookups can see data from
    /// previously-decoded slices within the same picture.
    pub fn inject_picture_maps(&mut self, maps: PictureMaps) {
        self.ct_depth_map = maps.ct_depth_map;
        self.ct_depth_map_stride = maps.ct_depth_map_stride;
        self.intra_mode_map = maps.intra_mode_map;
        self.intra_mode_map_stride = maps.intra_mode_map_stride;
        self.intra_chroma_mode_map = maps.intra_chroma_mode_map;
        self.pred_mode_map = maps.pred_mode_map;
        self.mv_info = maps.mv_info;
        self.cbf_map = maps.cbf_map;
        self.cbf_map_stride = maps.cbf_map_stride;
        self.qp_map = maps.qp_map;
        self.qp_map_stride = maps.qp_map_stride;
        self.sao_map = maps.sao_map;
    }

    /// Extract picture-level maps from this slice context after decoding.
    ///
    /// The returned maps can be injected into the next slice's context.
    pub fn extract_picture_maps(&mut self) -> PictureMaps {
        PictureMaps {
            ct_depth_map: core::mem::take(&mut self.ct_depth_map),
            ct_depth_map_stride: self.ct_depth_map_stride,
            intra_mode_map: core::mem::take(&mut self.intra_mode_map),
            intra_mode_map_stride: self.intra_mode_map_stride,
            intra_chroma_mode_map: core::mem::take(&mut self.intra_chroma_mode_map),
            pred_mode_map: core::mem::take(&mut self.pred_mode_map),
            mv_info: core::mem::take(&mut self.mv_info),
            cbf_map: core::mem::take(&mut self.cbf_map),
            cbf_map_stride: self.cbf_map_stride,
            qp_map: core::mem::take(&mut self.qp_map),
            qp_map_stride: self.qp_map_stride,
            sao_map: core::mem::replace(
                &mut self.sao_map,
                // SaoMap::new(0, 0) allocates 0 elements — cannot fail
                SaoMap::new(0, 0).unwrap_or_else(|_| unreachable!()),
            ),
        }
    }

    /// Decode all CTUs in the slice.
    ///
    /// `stop` is polled every [`STOP_CHECK_CTU_INTERVAL`] CTUs so a long
    /// single-tile intra frame can be cancelled mid-decode; on cancellation
    /// this returns [`HevcError::Cancelled`]. Pass [`enough::Unstoppable`]
    /// when cancellation is not needed.
    pub fn decode_slice(
        &mut self,
        frame: &mut DecodedFrame,
        stop: &dyn enough::Stop,
    ) -> Result<()> {
        // Initialize CABAC tracker for debugging
        debug::init_tracker();

        let ctb_size = self.sps.ctb_size();
        let pic_width_in_ctbs = self.sps.pic_width_in_ctbs();
        let pic_height_in_ctbs = self.sps.pic_height_in_ctbs();
        #[allow(unused)]
        let tile_debug = false;
        let wpp = self.pps.entropy_coding_sync_enabled_flag;
        let tiles = self.pps.tiles_enabled_flag;

        // Start from slice segment address
        let start_addr = self.header.slice_segment_address;

        // `slice_segment_address` is attacker-controlled; it indexes per-CTB
        // tables (e.g. `tile_scan_idx` below) and drives the CTB walk. An
        // address at/beyond the picture's CTB count — or a zero-CTB picture —
        // otherwise panics (OOB index `len N index N`, or a divide-by-zero in
        // the `ctb_x`/`ctb_y` derivation when the picture is 0 CTBs wide).
        // Reject as a malformed bitstream. Fuzz heic#26.
        let total_ctus = pic_width_in_ctbs
            .checked_mul(pic_height_in_ctbs)
            .unwrap_or(0);
        if total_ctus == 0 || start_addr >= total_ctus {
            return Err(at!(HevcError::InvalidBitstream(
                "slice_segment_address out of range for picture CTB count",
            )));
        }

        self.ctb_y = start_addr / pic_width_in_ctbs;
        self.ctb_x = start_addr % pic_width_in_ctbs;

        let mut ctu_count = 0u32;

        // Tiles: compute tile column/row boundaries and tile scan order
        // col_bd[i] = first CTB column of tile column i
        // row_bd[i] = first CTB row of tile row i
        let (tile_col_bd, tile_row_bd) = if tiles {
            compute_tile_boundaries(self.pps, pic_width_in_ctbs, pic_height_in_ctbs)
        } else {
            (vec![0, pic_width_in_ctbs], vec![0, pic_height_in_ctbs])
        };

        // Build CTB tile-scan order: CtbAddrTsToRs[ts] gives (ctb_x, ctb_y) in raster order
        // Tile scan: tiles in raster, CTBs within each tile in raster
        let tile_scan: Vec<(u32, u32)> = if tiles {
            build_tile_scan_order(&tile_col_bd, &tile_row_bd)
        } else {
            // No tiles: simple raster scan
            Vec::new()
        };

        // Build reverse mapping: given (ctb_x, ctb_y), what is the tile-scan index?
        let tile_scan_idx: Vec<u32> = if tiles {
            let map_size = (pic_width_in_ctbs * pic_height_in_ctbs) as usize;
            let mut idx = try_vec![0u32; map_size].map_err(|_| at!(HevcError::AllocationFailed))?;
            for (ts, &(cx, cy)) in tile_scan.iter().enumerate() {
                let i = (cy * pic_width_in_ctbs + cx) as usize;
                if i >= map_size {
                    return Err(at!(HevcError::InvalidParameterSet {
                        kind: "PPS",
                        msg: alloc::format!(
                            "tile scan coordinate ({cx},{cy}) exceeds picture CTB dimensions"
                        ),
                    }));
                }
                idx[i] = ts as u32;
            }
            idx
        } else {
            Vec::new()
        };

        // For tiles: current tile-scan position (used to determine tile boundaries)
        let mut tile_scan_pos = if tiles {
            let rs = start_addr;
            tile_scan_idx[rs as usize]
        } else {
            start_addr
        };

        #[cfg(feature = "std")]
        if tile_debug {
            eprintln!(
                "TILE_DBG: slice start_addr={} tile_scan_pos={} tile_col_bd={:?} tile_row_bd={:?} entry_points={} offsets={:?} pic_ctbs={}x{}",
                start_addr,
                tile_scan_pos,
                tile_col_bd,
                tile_row_bd,
                self.header.entry_point_offsets.len(),
                self.header.entry_point_offsets,
                pic_width_in_ctbs,
                pic_height_in_ctbs
            );
        }

        // Entry point byte offsets (for tiles and WPP)
        let mut entry_byte_offsets = Vec::new();
        if (tiles || wpp) && !self.header.entry_point_offsets.is_empty() {
            let mut cumulative = 0u32;
            for &offset in &self.header.entry_point_offsets {
                // Untrusted slice-header deltas: a malformed file can sum past
                // u32::MAX. Saturate — `cabac.seek_to` clamps to data.len(), so
                // an over-large position yields a clean incomplete-decode error
                // (caught by the UNINIT guard) rather than an overflow panic.
                cumulative = cumulative.saturating_add(offset);
                entry_byte_offsets.push(cumulative);
            }
        }
        let mut entry_idx = 0usize;

        #[cfg(feature = "std")]
        if tile_debug && !entry_byte_offsets.is_empty() {
            eprintln!(
                "TILE_DBG: cumulative entry offsets={:?}",
                entry_byte_offsets
            );
        }

        // WPP: saved context models from CTB column 1 of previous row
        let mut wpp_saved_ctx: Option<[super::cabac::ContextModel; context::NUM_CONTEXTS]> = None;

        // WPP: saved StatCoeff (rice parameter init states) from CTB column 1
        let mut wpp_saved_stat_coeff: Option<[u8; 4]> = None;

        #[cfg(feature = "std")]
        if WPP_TRACE && wpp {
            eprintln!(
                "WPP: slice type={:?} entry_point_offsets={:?} cumulative={:?}",
                self.header.slice_type, self.header.entry_point_offsets, entry_byte_offsets
            );
        }

        loop {
            // Cooperative cancellation: poll the stop token at CTU-block
            // granularity (first CTU, then every STOP_CHECK_CTU_INTERVAL) so a
            // long single-tile intra frame — which only hits the slice-entry
            // check once — can still be interrupted mid-decode. The check is
            // hoisted out of `decode_ctu`'s per-sample work; one masked compare
            // per CTU is negligible against a CTU's decode cost. Mirrors the
            // slice-entry check in `mod::decode_with_config_stop`.
            if ctu_count.is_multiple_of(STOP_CHECK_CTU_INTERVAL) && stop.should_stop() {
                return Err(at!(HevcError::Cancelled(enough::StopReason::Cancelled)));
            }

            // Bail out of a truncated/over-declaring slice. A conformant slice
            // terminates via end_of_slice_segment_flag before its CABAC data is
            // exhausted; once the decoder has fabricated more zero-bytes past the
            // end of the (sub)stream than the substream itself contained, every
            // remaining CTU is garbage. Without this, a tiny input declaring a
            // huge picture (e.g. 16384×16384 ⇒ ~1M CTUs) grinds through them all
            // for tens of seconds (fuzz #34). `seek_to`/`reinit` reset the
            // over-read budget so valid multi-tile / WPP substreams are unaffected.
            if self.cabac.overread_past_data() {
                return Err(at!(HevcError::InvalidBitstream(
                    "CABAC read past end of slice data (truncated or over-declared picture)",
                )));
            }

            // WPP: at start of each new row (ctb_x==0, ctb_y>0), restore saved context
            // and reinitialize CABAC at the substream entry point
            if wpp && self.ctb_x == 0 && self.ctb_y > 0 && pic_width_in_ctbs > 1 {
                #[cfg(feature = "std")]
                let pre_seek_pos = {
                    let (bp, _, _) = self.cabac.get_position();
                    bp
                };

                if let Some(saved) = wpp_saved_ctx {
                    self.ctx = saved;
                }
                // Restore StatCoeff (rice parameter init states) for WPP
                if let Some(saved_sc) = wpp_saved_stat_coeff {
                    self.stat_coeff = saved_sc;
                } else {
                    // No saved state: reset to 0 per spec
                    self.stat_coeff = [0; 4];
                }
                // Reinitialize CABAC at the substream entry point.
                // Spec 7.4.7.1: entry point offsets are in EBSP byte space
                // (i.e. they count emulation prevention bytes that are part of
                // the bitstream). Our slice_data is RBSP, so convert before
                // seeking.
                if entry_idx < entry_byte_offsets.len() {
                    let ebsp_target = entry_byte_offsets[entry_idx];
                    let target_byte = self.ebsp_offset_to_rbsp(ebsp_target) as usize;
                    self.cabac.seek_to(target_byte);
                    self.cabac.reinit();
                    #[cfg(feature = "std")]
                    if WPP_TRACE {
                        let (post_pos, _, _) = self.cabac.get_position();
                        let (r, v, bn) = self.cabac.get_state_extended();
                        eprintln!(
                            "WPP: row {} seek {}→{} (entry_ebsp={}) after_reinit_pos={} cabac(r={},v={},bn={})",
                            self.ctb_y, pre_seek_pos, target_byte, ebsp_target, post_pos, r, v, bn
                        );
                        // Print first few context model states
                        let ctx_sum: u32 = self.ctx.iter().map(|c| c.get_state().0 as u32).sum();
                        eprintln!("WPP: row {} ctx_checksum={}", self.ctb_y, ctx_sum);
                    }
                    entry_idx += 1;
                }
            }

            // Decode one CTU
            let x_ctb = self.ctb_x * ctb_size;
            let y_ctb = self.ctb_y * ctb_size;

            // Track CTU position for debugging
            let (byte_pos, _, _) = self.cabac.get_position();
            debug::track_ctu_start(ctu_count, byte_pos);

            // CTU-CK trace for per-CTU comparison with dec265
            #[cfg(feature = "std")]
            if self.mv_trace {
                let mut cksum: u64 = 0;
                for c in self.ctx.iter() {
                    let (s, m) = c.get_state();
                    cksum += s as u64 * 3 + m as u64;
                }
                eprintln!("CTU-CK ctu={} bp={} ck={}", ctu_count, byte_pos, cksum);
            }

            // DEBUG: Print CTU state periodically
            if ctu_count.is_multiple_of(50) || ctu_count <= 3 {
                let (_range, _offset) = self.cabac.get_state();
                debug_trace!(
                    "DEBUG: CTU {} byte={} cabac=({},{}) x={} y={}",
                    ctu_count,
                    byte_pos,
                    _range,
                    _offset,
                    self.ctb_x,
                    self.ctb_y
                );
            }
            // Enable debug for CTU 1 (where first large coefficient occurs)
            self.debug_ctu = ctu_count == 1;

            self.decode_ctu(x_ctb, y_ctb, frame)?;
            ctu_count += 1;

            // WPP: save context models and StatCoeff after decoding CTB column 1
            // Per H.265 9.3.2, storage happens after the CTU is decoded but before
            // end_of_slice_segment_flag, matching libde265's ordering.
            if wpp && self.ctb_x == 1 && self.ctb_y < pic_height_in_ctbs - 1 {
                wpp_saved_ctx = Some(self.ctx);
                wpp_saved_stat_coeff = Some(self.stat_coeff);
                #[cfg(feature = "std")]
                if WPP_TRACE {
                    let (bp, _, _) = self.cabac.get_position();
                    let ctx_sum: u32 = self.ctx.iter().map(|c| c.get_state().0 as u32).sum();
                    eprintln!(
                        "WPP: save ctx at ({},{}) byte={} ctx_checksum={}",
                        self.ctb_x, self.ctb_y, bp, ctx_sum
                    );
                }
            }

            // Check for end of slice segment
            let end_of_slice = self.cabac.decode_terminate()?;

            #[cfg(feature = "std")]
            if tile_debug && ctu_count <= 80 {
                let (bp, _, _) = self.cabac.get_position();
                let tile_id = get_tile_id(&tile_col_bd, &tile_row_bd, self.ctb_x, self.ctb_y);
                eprintln!(
                    "TILE_DBG: CTU {} at ({},{}) tile={} end_of_slice={} bp={}",
                    ctu_count, self.ctb_x, self.ctb_y, tile_id, end_of_slice, bp
                );
            }
            se_trace("end_of_slice", end_of_slice as i64, &self.cabac);

            if end_of_slice != 0 {
                debug_trace!(
                    "DEBUG: end_of_slice after CTU {}, decoded {}/{} CTUs",
                    ctu_count,
                    ctu_count,
                    total_ctus
                );
                break;
            }

            // Track previous position for boundary detection
            let prev_ctb_x = self.ctb_x;
            let prev_ctb_y = self.ctb_y;

            // Move to next CTB (tile scan order or raster scan)
            tile_scan_pos += 1;
            if tiles && (tile_scan_pos as usize) < tile_scan.len() {
                let (nx, ny) = tile_scan[tile_scan_pos as usize];
                self.ctb_x = nx;
                self.ctb_y = ny;
            } else if !tiles {
                self.ctb_x += 1;
                if self.ctb_x >= pic_width_in_ctbs {
                    self.ctb_x = 0;
                    self.ctb_y += 1;
                }
            } else {
                // Past end of tile scan
                break;
            }

            // Tiles: detect tile boundary and reinit CABAC
            if tiles {
                let prev_tile = get_tile_id(&tile_col_bd, &tile_row_bd, prev_ctb_x, prev_ctb_y);
                let curr_tile = get_tile_id(&tile_col_bd, &tile_row_bd, self.ctb_x, self.ctb_y);
                if curr_tile != prev_tile {
                    // Decode end_of_subset_one_bit at tile boundary
                    let _eoss = self.cabac.decode_terminate()?;

                    #[cfg(feature = "std")]
                    if tile_debug && ctu_count <= 120 {
                        let (bp, _, _) = self.cabac.get_position();
                        eprintln!(
                            "TILE_DBG: tile boundary at CTU {} ({},{})->({},{}), tile {}→{}, eoss={}, cabac_pos={}, entry_idx={}",
                            ctu_count,
                            prev_ctb_x,
                            prev_ctb_y,
                            self.ctb_x,
                            self.ctb_y,
                            prev_tile,
                            curr_tile,
                            _eoss,
                            bp,
                            entry_idx
                        );
                    }

                    // Reinitialize CABAC at the entry point for this tile.
                    // Entry point offsets are EBSP-relative; convert to RBSP.
                    if entry_idx < entry_byte_offsets.len() {
                        let ebsp_target = entry_byte_offsets[entry_idx];
                        let target_byte = self.ebsp_offset_to_rbsp(ebsp_target) as usize;
                        self.cabac.seek_to(target_byte);
                        self.cabac.reinit();
                        entry_idx += 1;
                    }

                    // Reinitialize context models (same as slice-level init)
                    {
                        use super::cabac::{INIT_VALUES_B, INIT_VALUES_P};
                        let init_table: &[u8; context::NUM_CONTEXTS] = match self.header.slice_type
                        {
                            SliceType::I => &INIT_VALUES,
                            SliceType::P => {
                                if self.header.cabac_init_flag {
                                    &INIT_VALUES_B
                                } else {
                                    &INIT_VALUES_P
                                }
                            }
                            SliceType::B => {
                                if self.header.cabac_init_flag {
                                    &INIT_VALUES_P
                                } else {
                                    &INIT_VALUES_B
                                }
                            }
                        };
                        for (i, init_val) in init_table.iter().enumerate() {
                            self.ctx[i].init(*init_val, self.header.slice_qp_y);
                        }
                    }

                    // Reset StatCoeff
                    self.stat_coeff = [0; 4];

                    // Reset QP state to slice QP (tile start = new QP prediction)
                    self.current_qpy = self.header.slice_qp_y;
                    self.last_qpy_in_prev_qg = self.header.slice_qp_y;
                    self.current_qg_x = -1;
                    self.current_qg_y = -1;
                    self.is_cu_qp_delta_coded = false;
                    self.cu_qp_delta = 0;
                }
            }

            // WPP: at row boundaries, decode end_of_subset_one_bit and byte-align
            // This consumes the substream termination syntax before the next row
            // starts fresh from the entry point.
            if wpp && self.ctb_y != prev_ctb_y {
                #[cfg(feature = "std")]
                let pre_eoss_pos = {
                    let (bp, _, _) = self.cabac.get_position();
                    bp
                };
                let _eoss = self.cabac.decode_terminate()?;
                #[cfg(feature = "std")]
                if WPP_TRACE {
                    let (bp, _, _) = self.cabac.get_position();
                    eprintln!(
                        "WPP: end_of_subset at row boundary (row {} → {}): eoss={} byte {}→{}",
                        prev_ctb_y, self.ctb_y, _eoss, pre_eoss_pos, bp
                    );
                }
                // Note: no reinit here — the seek_to + reinit at the top of the
                // loop handles CABAC reinitialization for the next substream.
            }

            // Check for end of picture
            if self.ctb_y >= pic_height_in_ctbs {
                break;
            }
        }

        #[cfg(feature = "std")]
        if tile_debug {
            eprintln!(
                "TILE_DBG: slice done, decoded {} CTUs (expected {}), entry_idx={}, last_pos=({},{})",
                ctu_count, total_ctus, entry_idx, self.ctb_x, self.ctb_y
            );
        }

        if DEBUG_TRACE {
            debug::print_tracker_summary();
        }
        Ok(())
    }

    /// Decode a single CTU (Coding Tree Unit)
    fn decode_ctu(&mut self, x_ctb: u32, y_ctb: u32, frame: &mut DecodedFrame) -> Result<()> {
        let log2_ctb_size = self.sps.log2_ctb_size();

        // Reset per-CTU state
        if self.pps.cu_qp_delta_enabled_flag {
            self.is_cu_qp_delta_coded = false;
            self.cu_qp_delta = 0;
        }

        // Decode SAO syntax elements (must consume from CABAC stream even if not applied)
        if self.header.slice_sao_luma_flag || self.header.slice_sao_chroma_flag {
            self.decode_sao(x_ctb, y_ctb)?;
        }

        // Decode the coding quadtree
        self.decode_coding_quadtree(x_ctb, y_ctb, log2_ctb_size, 0, frame)
    }

    /// Decode SAO (Sample Adaptive Offset) syntax elements from CABAC stream
    /// and store them in the SAO map for later filtering.
    fn decode_sao(&mut self, x_ctb_pixels: u32, y_ctb_pixels: u32) -> Result<()> {
        let ctb_size = self.sps.ctb_size();
        let x_ctb = x_ctb_pixels / ctb_size;
        let y_ctb = y_ctb_pixels / ctb_size;

        let mut sao_merge_left_flag = false;
        let mut sao_merge_up_flag = false;

        // sao_merge_left_flag: available if left CTB is in same slice and same tile
        if x_ctb > 0 {
            let pic_width_ctbs = self.sps.pic_width_in_ctbs();
            let ctb_addr_rs = y_ctb * pic_width_ctbs + x_ctb;
            let slice_addr_rs = self.header.slice_segment_address;
            let left_in_slice = ctb_addr_rs > slice_addr_rs;
            let left_in_tile = !self.pps.tiles_enabled_flag
                || get_tile_id(&self.tile_col_bd, &self.tile_row_bd, x_ctb - 1, y_ctb)
                    == get_tile_id(&self.tile_col_bd, &self.tile_row_bd, x_ctb, y_ctb);
            if left_in_slice && left_in_tile {
                let ctx_idx = context::SAO_MERGE_FLAG;
                sao_merge_left_flag = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
                se_trace("sao_merge_left", sao_merge_left_flag as i64, &self.cabac);
            }
        }

        // sao_merge_up_flag: available if above CTB is in same slice and same tile
        if y_ctb > 0 && !sao_merge_left_flag {
            let pic_width_ctbs = self.sps.pic_width_in_ctbs();
            let ctb_addr_rs = y_ctb * pic_width_ctbs + x_ctb;
            let slice_addr_rs = self.header.slice_segment_address;
            let up_in_slice = ctb_addr_rs >= pic_width_ctbs + slice_addr_rs;
            let up_in_tile = !self.pps.tiles_enabled_flag
                || get_tile_id(&self.tile_col_bd, &self.tile_row_bd, x_ctb, y_ctb - 1)
                    == get_tile_id(&self.tile_col_bd, &self.tile_row_bd, x_ctb, y_ctb);
            if up_in_slice && up_in_tile {
                let ctx_idx = context::SAO_MERGE_FLAG;
                sao_merge_up_flag = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
                se_trace("sao_merge_up", sao_merge_up_flag as i64, &self.cabac);
            }
        }

        let sao_info = if sao_merge_left_flag {
            *self.sao_map.get(x_ctb - 1, y_ctb)
        } else if sao_merge_up_flag {
            *self.sao_map.get(x_ctb, y_ctb - 1)
        } else {
            let mut info = super::sao::SaoInfo::default();
            let is_mono = self.sps.chroma_format_idc == 0;
            let n_chroma = if is_mono { 1 } else { 3 };

            #[allow(unused_assignments)]
            let mut sao_type_idx_luma = 0u8;
            let mut sao_type_idx_chroma = 0u8;
            let mut eo_class_chroma = 0u8;

            for c_idx in 0..n_chroma {
                let should_decode = (self.header.slice_sao_luma_flag && c_idx == 0)
                    || (self.header.slice_sao_chroma_flag && c_idx > 0);

                if !should_decode {
                    continue;
                }

                let sao_type_idx = if c_idx == 0 {
                    sao_type_idx_luma = self.decode_sao_type_idx()?;
                    se_trace("sao_type_idx_luma", sao_type_idx_luma as i64, &self.cabac);
                    sao_type_idx_luma
                } else if c_idx == 1 {
                    sao_type_idx_chroma = self.decode_sao_type_idx()?;
                    se_trace(
                        "sao_type_idx_chroma",
                        sao_type_idx_chroma as i64,
                        &self.cabac,
                    );
                    sao_type_idx_chroma
                } else {
                    sao_type_idx_chroma
                };

                info.sao_type_idx[c_idx] = sao_type_idx;

                if sao_type_idx != 0 {
                    let bit_depth = if c_idx == 0 {
                        self.sps.bit_depth_y() as u32
                    } else {
                        self.sps.bit_depth_c() as u32
                    };
                    let c_max = (1u32 << (bit_depth.min(10) - 5)) - 1;
                    let offset_scale = 1i32 << (bit_depth.saturating_sub(bit_depth.min(10)));

                    let mut offsets_abs = [0u32; 4];
                    for elem in &mut offsets_abs {
                        *elem = self.decode_cabac_tu_bypass(c_max)?;
                        se_trace("sao_offset_abs", *elem as i64, &self.cabac);
                    }

                    if sao_type_idx == 1 {
                        // Band offset: decode signs + band position
                        let mut signed_offsets = [0i16; 4];
                        for i in 0..4 {
                            if offsets_abs[i] != 0 {
                                let sign = self.cabac.decode_bypass()?;
                                se_trace("sao_offset_sign", sign as i64, &self.cabac);
                                let val = (offsets_abs[i] as i32 * offset_scale) as i16;
                                signed_offsets[i] = if sign != 0 { -val } else { val };
                            }
                        }
                        info.sao_offset_val[c_idx] = signed_offsets;

                        let band_pos = self.cabac.decode_bypass_bits(5)?;
                        se_trace("sao_band_position", band_pos as i64, &self.cabac);
                        info.sao_band_position[c_idx] = band_pos as u8;
                    } else {
                        // Edge offset: store absolute values (sign applied during filtering)
                        for (i, &offset) in offsets_abs.iter().enumerate() {
                            info.sao_offset_val[c_idx][i] = (offset as i32 * offset_scale) as i16;
                        }

                        if c_idx <= 1 {
                            let eo_class = self.cabac.decode_bypass_bits(2)?;
                            se_trace("sao_eo_class", eo_class as i64, &self.cabac);
                            if c_idx == 0 {
                                info.sao_eo_class[0] = eo_class as u8;
                            } else {
                                eo_class_chroma = eo_class as u8;
                                info.sao_eo_class[1] = eo_class_chroma;
                            }
                        } else {
                            info.sao_eo_class[2] = eo_class_chroma;
                        }
                    }
                }
            }
            info
        };

        if let Some(entry) = self.sao_map.get_mut(x_ctb, y_ctb) {
            *entry = sao_info;
        }
        Ok(())
    }

    /// Decode sao_type_idx: context bin + optional bypass bin
    fn decode_sao_type_idx(&mut self) -> Result<u8> {
        let ctx_idx = context::SAO_TYPE_IDX;
        let bit0 = self.cabac.decode_bin(&mut self.ctx[ctx_idx])?;
        if bit0 == 0 {
            Ok(0)
        } else {
            let bit1 = self.cabac.decode_bypass()?;
            if bit1 == 0 { Ok(1) } else { Ok(2) }
        }
    }

    /// Decode truncated unary with bypass bins (for sao_offset_abs)
    fn decode_cabac_tu_bypass(&mut self, c_max: u32) -> Result<u32> {
        for i in 0..c_max {
            let bit = self.cabac.decode_bypass()?;
            if bit == 0 {
                return Ok(i);
            }
        }
        Ok(c_max)
    }

    /// Decode coding quadtree recursively
    fn decode_coding_quadtree(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        frame: &mut DecodedFrame,
    ) -> Result<()> {
        let cb_size = 1u32 << log2_cb_size;
        let pic_width = self.sps.pic_width_in_luma_samples;
        let pic_height = self.sps.pic_height_in_luma_samples;
        let log2_min_cb_size = self.sps.log2_min_cb_size();

        // Determine if we need to split
        let split_flag = if x0 + cb_size <= pic_width
            && y0 + cb_size <= pic_height
            && log2_cb_size > log2_min_cb_size
        {
            // Decode split_cu_flag
            let flag = self.decode_split_cu_flag(x0, y0, ct_depth)?;
            if self.debug_ctu {
                let (_r, _o) = self.cabac.get_state();
                debug_trace!(
                    "  CTU37: split_cu_flag at ({},{}) depth={} log2={} → {} (r={},o={})",
                    x0,
                    y0,
                    ct_depth,
                    log2_cb_size,
                    flag,
                    _r,
                    _o
                );
            }
            flag
        } else if log2_cb_size > log2_min_cb_size {
            // Must split if partially outside picture
            if self.debug_ctu {
                debug_trace!(
                    "  CTU37: forced split at ({},{}) depth={} - outside picture",
                    x0,
                    y0,
                    ct_depth
                );
            }
            true
        } else {
            // At minimum size, don't split
            if self.debug_ctu {
                debug_trace!(
                    "  CTU37: no split at ({},{}) depth={} - min size",
                    x0,
                    y0,
                    ct_depth
                );
            }
            false
        };

        // Handle QP delta depth: reset at quantization group boundaries
        // Log2MinCuQpDeltaSize = Log2CtbSizeY - diff_cu_qp_delta_depth
        if self.pps.cu_qp_delta_enabled_flag
            && log2_cb_size
                >= self
                    .sps
                    .log2_ctb_size()
                    .saturating_sub(self.pps.diff_cu_qp_delta_depth)
        {
            self.is_cu_qp_delta_coded = false;
            self.cu_qp_delta = 0;
        }

        if split_flag {
            let half = cb_size / 2;
            let x1 = x0 + half;
            let y1 = y0 + half;

            // Decode four sub-CUs
            self.decode_coding_quadtree(x0, y0, log2_cb_size - 1, ct_depth + 1, frame)?;

            if x1 < pic_width {
                self.decode_coding_quadtree(x1, y0, log2_cb_size - 1, ct_depth + 1, frame)?;
            }

            if y1 < pic_height {
                self.decode_coding_quadtree(x0, y1, log2_cb_size - 1, ct_depth + 1, frame)?;
            }

            if x1 < pic_width && y1 < pic_height {
                self.decode_coding_quadtree(x1, y1, log2_cb_size - 1, ct_depth + 1, frame)?;
            }
        } else {
            // Decode the coding unit
            self.decode_coding_unit(x0, y0, log2_cb_size, ct_depth, frame)?;
        }

        Ok(())
    }

    /// Get ctDepth at a pixel position (returns 0xFF if not yet decoded)
    fn get_ct_depth(&self, x: u32, y: u32) -> u8 {
        let min_cb_size = 1u32 << self.sps.log2_min_cb_size();
        let map_x = x / min_cb_size;
        let map_y = y / min_cb_size;

        if map_x >= self.ct_depth_map_stride
            || map_y * self.ct_depth_map_stride + map_x >= self.ct_depth_map.len() as u32
        {
            return 0xFF; // Out of bounds
        }

        self.ct_depth_map[(map_y * self.ct_depth_map_stride + map_x) as usize]
    }

    /// Set ctDepth for a CU region
    fn set_ct_depth(&mut self, x0: u32, y0: u32, log2_cb_size: u8, ct_depth: u8) {
        let min_cb_size = 1u32 << self.sps.log2_min_cb_size();
        let cb_size = 1u32 << log2_cb_size;

        // Fill the ct_depth_map for this CU region
        let start_x = x0 / min_cb_size;
        let start_y = y0 / min_cb_size;
        let num_blocks = cb_size / min_cb_size;

        for dy in 0..num_blocks {
            for dx in 0..num_blocks {
                let map_x = start_x + dx;
                let map_y = start_y + dy;
                if map_x < self.ct_depth_map_stride {
                    let idx = (map_y * self.ct_depth_map_stride + map_x) as usize;
                    if idx < self.ct_depth_map.len() {
                        self.ct_depth_map[idx] = ct_depth;
                    }
                }
            }
        }
    }

    /// Check if a neighbor position is available (within picture bounds)
    fn is_neighbor_available(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 {
            return false;
        }
        let xu = x as u32;
        let yu = y as u32;
        if xu >= self.sps.pic_width_in_luma_samples || yu >= self.sps.pic_height_in_luma_samples {
            return false;
        }
        // Tile boundary check: neighbor must be in the same tile as current CTB
        if self.pps.tiles_enabled_flag {
            let ctb_size = self.sps.ctb_size();
            let nb_ctb_x = xu / ctb_size;
            let nb_ctb_y = yu / ctb_size;
            if get_tile_id(&self.tile_col_bd, &self.tile_row_bd, nb_ctb_x, nb_ctb_y)
                != get_tile_id(&self.tile_col_bd, &self.tile_row_bd, self.ctb_x, self.ctb_y)
            {
                return false;
            }
        }
        true
    }

    /// Decode split_cu_flag using CABAC
    fn decode_split_cu_flag(&mut self, x0: u32, y0: u32, ct_depth: u8) -> Result<bool> {
        // Context selection based on neighboring CU depths (H.265 9.3.4.2.2)
        // condTermL: 1 if left neighbor has larger depth (was split more)
        // condTermA: 1 if above neighbor has larger depth
        // ctxInc = condTermL + condTermA

        let available_l = self.is_neighbor_available(x0 as i32 - 1, y0 as i32);
        let available_a = self.is_neighbor_available(x0 as i32, y0 as i32 - 1);

        let mut cond_l = 0;
        let mut cond_a = 0;

        if available_l {
            let depth_l = self.get_ct_depth(x0 - 1, y0);
            if depth_l != 0xFF && depth_l > ct_depth {
                cond_l = 1;
            }
        }

        if available_a {
            let depth_a = self.get_ct_depth(x0, y0 - 1);
            if depth_a != 0xFF && depth_a > ct_depth {
                cond_a = 1;
            }
        }

        let ctx_idx = context::SPLIT_CU_FLAG + cond_l + cond_a;
        let bin = self.cabac.decode_bin(&mut self.ctx[ctx_idx])?;
        se_trace("split_cu_flag", bin as i64, &self.cabac);

        Ok(bin != 0)
    }

    /// Decode a coding unit
    fn decode_coding_unit(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        frame: &mut DecodedFrame,
    ) -> Result<()> {
        let cb_size = 1u32 << log2_cb_size;
        let _ = cb_size; // Used in PartNxN

        // Track CU base position for transform unit QP derivation
        self.cu_base_x = x0;
        self.cu_base_y = y0;
        self.cu_log2_size = log2_cb_size;

        // Decode quantization parameters at CU start (H.265 8.6.1)
        self.decode_quantization_parameters(x0, y0, x0, y0);
        self.store_qpy(x0, y0, log2_cb_size, self.current_qpy);

        // Set ct_depth for this CU (used by split_cu_flag context derivation)
        self.set_ct_depth(x0, y0, log2_cb_size, ct_depth);

        let is_intra_slice = self.header.slice_type.is_intra();

        // --- cu_skip_flag (P/B slices only) ---
        let cu_skip = if !is_intra_slice {
            self.decode_cu_skip_flag(x0, y0)?
        } else {
            false
        };

        // Determine prediction mode
        let pred_mode;
        if cu_skip {
            pred_mode = PredMode::Skip;
        } else {
            // Decode transquant_bypass_flag if enabled
            self.cu_transquant_bypass_flag = if self.pps.transquant_bypass_enabled_flag {
                let ctx_idx = context::CU_TRANSQUANT_BYPASS_FLAG;
                self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0
            } else {
                false
            };

            if !is_intra_slice {
                pred_mode = self.decode_pred_mode_flag()?;
            } else {
                pred_mode = PredMode::Intra;
            }
        }

        // Store prediction mode for the CU
        self.store_pred_mode(x0, y0, log2_cb_size, pred_mode);

        // --- PCM mode (H.265 7.3.8.5) ---
        // Check for pcm_flag when intra and CU size is within PCM range
        if pred_mode == PredMode::Intra
            && !cu_skip
            && let Some(ref pcm) = self.sps.pcm_params
        {
            let log2_min_ipcm = pcm.log2_min_pcm_luma_coding_block_size_minus3 + 3;
            let log2_max_ipcm = log2_min_ipcm + pcm.log2_diff_max_min_pcm_luma_coding_block_size;
            if log2_cb_size >= log2_min_ipcm && log2_cb_size <= log2_max_ipcm {
                let pcm_flag = self.cabac.decode_terminate()?;
                se_trace("pcm_flag", pcm_flag as i64, &self.cabac);
                if pcm_flag != 0 {
                    // PCM mode: terminate CABAC, read raw samples, reinit
                    self.decode_pcm_samples(x0, y0, log2_cb_size, frame)?;
                    // Mark boundaries and QP for deblocking
                    frame.mark_tu_boundary(x0, y0, cb_size);
                    frame.store_block_qp(x0, y0, cb_size, self.current_qpy as i8);
                    self.store_cbf(x0, y0, cb_size, false);
                    return Ok(());
                }
            }
        }

        // --- Skip mode: decode merge_idx only, no partition/residual ---
        if pred_mode == PredMode::Skip {
            let coding = self.decode_inter_pu(x0, y0, cb_size, cb_size, ct_depth, true)?;

            // Resolve merge candidate → real motion vectors
            let motion =
                self.resolve_motion(&coding, x0, y0, cb_size, cb_size, 0, PartMode::Part2Nx2N);

            self.store_mv_info(x0, y0, cb_size, cb_size, motion);

            // Mark CU boundary and QP for deblocking (skip has no residual, so cbf=false)
            frame.mark_tu_boundary(x0, y0, cb_size);
            frame.store_block_qp(x0, y0, cb_size, self.current_qpy as i8);
            self.store_cbf(x0, y0, cb_size, false);

            if self.mv_trace {
                #[cfg(feature = "std")]
                eprintln!(
                    "MV_TRACE: SKIP ({},{}) {}x{} merge_idx={} L0=({},{})r{} L1=({},{})r{} pred=[{},{}]",
                    x0,
                    y0,
                    cb_size,
                    cb_size,
                    coding.merge_idx,
                    motion.mv[0].x,
                    motion.mv[0].y,
                    motion.ref_idx[0],
                    motion.mv[1].x,
                    motion.mv[1].y,
                    motion.ref_idx[1],
                    motion.pred_flag[0] as u8,
                    motion.pred_flag[1] as u8
                );
            }

            // Apply motion compensation (prediction → frame)
            let mut mc_scratch = core::mem::take(&mut self.mc_scratch);
            self.apply_mc(&motion, x0, y0, cb_size, cb_size, &mut mc_scratch, frame);
            self.mc_scratch = mc_scratch;
            return Ok(());
        }

        // --- Decode partition mode ---
        let part_mode = if pred_mode == PredMode::Intra {
            if log2_cb_size == self.sps.log2_min_cb_size() {
                let pm = self.decode_part_mode(pred_mode, log2_cb_size)?;
                if pm == PartMode::PartNxN {
                    static NXN_COUNT: core::sync::atomic::AtomicU32 =
                        core::sync::atomic::AtomicU32::new(0);
                    let count = NXN_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                    if count == 0 || x0 < 64 && y0 < 64 {
                        let (_r, _o) = self.cabac.get_state();
                        debug_trace!(
                            "DEBUG: part_mode at ({},{}) log2={}: {:?} cabac=({},{})",
                            x0,
                            y0,
                            log2_cb_size,
                            pm,
                            _r,
                            _o
                        );
                    }
                }
                if self.debug_ctu {
                    let (_r, _o) = self.cabac.get_state();
                    debug_trace!(
                        "  CTU37: CU at ({},{}) log2={} part_mode={:?} (r={},o={})",
                        x0,
                        y0,
                        log2_cb_size,
                        pm,
                        _r,
                        _o
                    );
                }
                pm
            } else {
                if self.debug_ctu {
                    debug_trace!(
                        "  CTU37: CU at ({},{}) log2={} part_mode=2Nx2N (implicit)",
                        x0,
                        y0,
                        log2_cb_size
                    );
                }
                PartMode::Part2Nx2N
            }
        } else {
            // Inter: decode partition mode with all 8 modes available
            self.decode_part_mode(pred_mode, log2_cb_size)?
        };

        // --- Decode prediction info ---
        let mut pu_list_is_merge = false;
        let (intra_luma_mode, intra_chroma_mode) = if pred_mode == PredMode::Intra {
            match part_mode {
                PartMode::Part2Nx2N => {
                    let modes = self.decode_intra_prediction(x0, y0, log2_cb_size, true, frame)?;
                    if self.debug_ctu {
                        let (_r, _o) = self.cabac.get_state();
                        debug_trace!(
                            "  CTU37: After intra_prediction: mode={:?} (r={},o={}) bits={}",
                            modes,
                            _r,
                            _o,
                            self.cabac.get_position().2
                        );
                    }
                    modes
                }
                PartMode::PartNxN => {
                    let half = cb_size / 2;
                    let log2_pu_size = log2_cb_size - 1;

                    let prev_flags = [
                        self.decode_prev_intra_luma_pred_flag()?,
                        self.decode_prev_intra_luma_pred_flag()?,
                        self.decode_prev_intra_luma_pred_flag()?,
                        self.decode_prev_intra_luma_pred_flag()?,
                    ];

                    let luma_mode_0 = self.derive_intra_luma_mode(x0, y0, prev_flags[0])?;
                    self.store_intra_mode(x0, y0, log2_pu_size, luma_mode_0);

                    let luma_mode_1 = self.derive_intra_luma_mode(x0 + half, y0, prev_flags[1])?;
                    self.store_intra_mode(x0 + half, y0, log2_pu_size, luma_mode_1);

                    let luma_mode_2 = self.derive_intra_luma_mode(x0, y0 + half, prev_flags[2])?;
                    self.store_intra_mode(x0, y0 + half, log2_pu_size, luma_mode_2);

                    let luma_mode_3 =
                        self.derive_intra_luma_mode(x0 + half, y0 + half, prev_flags[3])?;
                    self.store_intra_mode(x0 + half, y0 + half, log2_pu_size, luma_mode_3);

                    // H.265 7.3.8.5: ChromaArrayType==3 (4:4:4) reads one
                    // intra_chroma_pred_mode per luma PB (four for NxN); else one.
                    let chroma_array_type = if self.sps.separate_colour_plane_flag {
                        0
                    } else {
                        self.sps.chroma_format_idc
                    };
                    let chroma_mode = if chroma_array_type == 3 {
                        let cm0 = self.decode_intra_chroma_mode(luma_mode_0)?;
                        self.store_intra_chroma_mode(x0, y0, log2_pu_size, cm0);
                        let cm1 = self.decode_intra_chroma_mode(luma_mode_1)?;
                        self.store_intra_chroma_mode(x0 + half, y0, log2_pu_size, cm1);
                        let cm2 = self.decode_intra_chroma_mode(luma_mode_2)?;
                        self.store_intra_chroma_mode(x0, y0 + half, log2_pu_size, cm2);
                        let cm3 = self.decode_intra_chroma_mode(luma_mode_3)?;
                        self.store_intra_chroma_mode(x0 + half, y0 + half, log2_pu_size, cm3);
                        cm0
                    } else {
                        let cm = self.decode_intra_chroma_mode(luma_mode_0)?;
                        self.store_intra_chroma_mode(x0, y0, log2_cb_size, cm);
                        cm
                    };

                    (luma_mode_0, chroma_mode)
                }
                _ => {
                    return Err(at!(HevcError::InvalidBitstream(
                        "invalid intra partition mode"
                    )));
                }
            }
        } else {
            // Inter prediction: decode PUs, resolve motion, apply MC
            let pu_list = partition_to_pu_list(part_mode, x0, y0, cb_size);
            let mut any_merge = false;
            for (part_idx, &(px, py, pw, ph)) in pu_list.iter().enumerate() {
                let coding = self.decode_inter_pu(px, py, pw, ph, ct_depth, false)?;
                if coding.merge_flag {
                    any_merge = true;
                }
                let motion =
                    self.resolve_motion(&coding, px, py, pw, ph, part_idx as u8, part_mode);
                self.store_mv_info(px, py, pw, ph, motion);

                if self.mv_trace {
                    #[cfg(feature = "std")]
                    eprintln!(
                        "MV_TRACE: INTER ({},{}) {}x{} merge={} idx={} L0=({},{})r{} L1=({},{})r{} pred=[{},{}]",
                        px,
                        py,
                        pw,
                        ph,
                        coding.merge_flag as u8,
                        coding.merge_idx,
                        motion.mv[0].x,
                        motion.mv[0].y,
                        motion.ref_idx[0],
                        motion.mv[1].x,
                        motion.mv[1].y,
                        motion.ref_idx[1],
                        motion.pred_flag[0] as u8,
                        motion.pred_flag[1] as u8
                    );
                }

                let mut mc_scratch = core::mem::take(&mut self.mc_scratch);
                self.apply_mc(&motion, px, py, pw, ph, &mut mc_scratch, frame);
                self.mc_scratch = mc_scratch;
            }
            pu_list_is_merge = any_merge;

            // Mark prediction block boundaries for deblocking (H.265 8.7.2.3)
            // Internal PB edges are distinct from TB edges; CBF check doesn't apply at PB-only edges
            mark_pb_boundaries(frame, part_mode, x0, y0, cb_size);

            // Inter CUs use DC for intra modes (unused in transform path)
            (IntraPredMode::Dc, IntraPredMode::Dc)
        };

        // --- Decode residual (transform tree) ---
        if pred_mode == PredMode::Inter {
            // Inter: rqt_root_cbf determines if there's any residual
            // For Part2Nx2N merge CUs, rqt_root_cbf is implied 1 (H.265 7.3.8.5)
            let is_merge_2nx2n = part_mode == PartMode::Part2Nx2N && pu_list_is_merge;
            let has_residual = if is_merge_2nx2n {
                true // rqt_root_cbf implied 1
            } else {
                self.decode_rqt_root_cbf()?
            };
            if has_residual {
                let intra_split_flag = false;
                // H.265 7.3.8.7: interSplitFlag forces TU split when
                // max_transform_hierarchy_depth_inter==0 and PartMode != 2Nx2N
                let inter_split_flag = self.sps.max_transform_hierarchy_depth_inter == 0
                    && part_mode != PartMode::Part2Nx2N;
                self.decode_transform_tree(
                    x0,
                    y0,
                    log2_cb_size,
                    0,
                    intra_luma_mode,
                    intra_chroma_mode,
                    intra_split_flag,
                    inter_split_flag,
                    frame,
                )?;
            } else {
                // No residual: still need to mark CU boundary and QP for deblocking
                frame.mark_tu_boundary(x0, y0, cb_size);
                frame.store_block_qp(x0, y0, cb_size, self.current_qpy as i8);
                self.store_cbf(x0, y0, cb_size, false);
            }
        } else {
            // Intra: residual always present (rqt_root_cbf implied 1). The
            // transform tree is parsed even under cu_transquant_bypass: bypass
            // only skips the inverse transform/dequant in reconstruction, not
            // the residual_coding syntax — skipping it desyncs CABAC.
            let intra_split_flag = part_mode == PartMode::PartNxN;
            self.decode_transform_tree(
                x0,
                y0,
                log2_cb_size,
                0,
                intra_luma_mode,
                intra_chroma_mode,
                intra_split_flag,
                false, // inter_split_flag: not applicable to intra
                frame,
            )?;

            if self.debug_ctu {
                let (_r, _o) = self.cabac.get_state();
                debug_trace!(
                    "  CTU37: After transform_tree at ({},{}) log2={} (r={},o={})",
                    x0,
                    y0,
                    log2_cb_size,
                    _r,
                    _o
                );
            }
        }

        Ok(())
    }

    /// Decode transform tree recursively
    #[allow(clippy::too_many_arguments)]
    fn decode_transform_tree(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        intra_luma_mode: IntraPredMode,
        intra_chroma_mode: IntraPredMode,
        intra_split_flag: bool,
        inter_split_flag: bool,
        frame: &mut DecodedFrame,
    ) -> Result<()> {
        // Start with root having chroma responsibility
        self.decode_transform_tree_inner(
            x0,
            y0,
            log2_size,
            trafo_depth,
            intra_luma_mode,
            intra_chroma_mode,
            intra_split_flag,
            inter_split_flag,
            ChromaCbf::ROOT,
            frame,
        )
    }

    /// Inner transform tree decoding
    /// parent_cbf: whether parent says chroma has residuals (ROOT at depth 0)
    #[allow(clippy::too_many_arguments)]
    fn decode_transform_tree_inner(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        intra_luma_mode: IntraPredMode,
        intra_chroma_mode: IntraPredMode,
        intra_split_flag: bool,
        inter_split_flag: bool,
        parent_cbf: ChromaCbf,
        frame: &mut DecodedFrame,
    ) -> Result<()> {
        // Per H.265: MaxTrafoDepth depends on prediction mode
        // Intra: max_transform_hierarchy_depth_intra + IntraSplitFlag
        // Inter: max_transform_hierarchy_depth_inter
        let max_trafo_depth =
            if intra_split_flag || self.get_pred_mode_at(x0, y0) == PredMode::Intra {
                self.sps.max_transform_hierarchy_depth_intra + if intra_split_flag { 1 } else { 0 }
            } else {
                self.sps.max_transform_hierarchy_depth_inter
            };
        let log2_min_trafo_size = self.sps.log2_min_tb_size();
        let log2_max_trafo_size = self.sps.log2_max_tb_size();

        // Per HEVC spec 7.3.8.7, the order is:
        // 1. split_transform_flag (if applicable)
        // 2. cbf_cb (if applicable)
        // 3. cbf_cr (if applicable)

        // Debug for specific position
        let debug_tt = self.debug_ctu;

        // Step 1: Determine if we should split
        // Per H.265 7.3.8.7: decode split_transform_flag only when all conditions met AND
        // NOT (IntraSplitFlag && trafoDepth == 0)
        let split_transform = if log2_size <= log2_max_trafo_size
            && log2_size > log2_min_trafo_size
            && trafo_depth < max_trafo_depth
            && !(intra_split_flag && trafo_depth == 0)
        {
            // Decode split_transform_flag
            let ctx_idx = context::SPLIT_TRANSFORM_FLAG + (5 - log2_size as usize).min(2);
            let flag = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
            se_trace("split_transform", flag as i64, &self.cabac);
            flag
        } else if log2_size > log2_max_trafo_size
            || (intra_split_flag && trafo_depth == 0)
            || (inter_split_flag && trafo_depth == 0)
        {
            // Must split: larger than max, IntraSplitFlag at depth 0,
            // or interSplitFlag at depth 0 (H.265 7.3.8.7)
            true
        } else {
            if debug_tt {
                debug_trace!(
                    "    TT(1144,120): no split (log2={} min={} max={} depth={} maxdepth={})",
                    log2_size,
                    log2_min_trafo_size,
                    log2_max_trafo_size,
                    trafo_depth,
                    max_trafo_depth
                );
            }
            false
        };

        // Step 2: Decode cbf_cb and cbf_cr.
        // H.265 7.3.8.8 gates these on `log2TrafoSize > 2 && ChromaArrayType != 0`
        // (plus a `ChromaArrayType == 3` carve-out at the smallest size). For
        // MONOCHROME (ChromaArrayType == 0) there is no chroma, so cbf_cb/cbf_cr
        // are NOT present in the bitstream — reading them consumes ~2 phantom
        // CABAC bins per TU and desyncs the entire decode (the Apple HDR gain
        // map, a 4:0:0 stream, decoded ~95% UNINIT because of this). Infer 0.
        // ChromaArrayType is 0 when chroma_format_idc == 0 OR
        // separate_colour_plane_flag == 1.
        let chroma_array_type = if self.sps.separate_colour_plane_flag {
            0
        } else {
            self.sps.chroma_format_idc
        };
        let cbf = if chroma_array_type == 0 {
            ChromaCbf::NONE
        } else if log2_size > 2 || chroma_array_type == 3 {
            // H.265 7.3.8.8: cbf_cb/cbf_cr present when log2TrafoSize > 2 OR
            // ChromaArrayType == 3 (4:4:4 reads them at the 4x4 leaf too, where
            // chroma is decoded at the leaf rather than inherited from parent).
            //
            // ChromaArrayType == 2 (4:2:2): each chroma TB is two square blocks
            // stacked vertically, and a SECOND cbf_cb / cbf_cr is coded for the
            // bottom block whenever this node is a leaf, or an 8x8 about to
            // split into 4x4 luma (whose chroma is coded here, at the parent).
            // Skipping those bins desynced CABAC on every 4:2:2 stream (#48).
            // Both bins of a pair share ctxInc = trafoDepth (Table 9-41).
            let second = chroma_array_type == 2 && (!split_transform || log2_size == 3);
            let ctx_idx = context::CBF_CBCR + trafo_depth as usize;
            let mut cbf = ChromaCbf::NONE;
            // Decode cbf_cb if trafo_depth == 0 (always) or parent had cbf_cb
            if trafo_depth == 0 || parent_cbf.cb != 0 {
                let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
                se_trace("cbf_cb", val as i64, &self.cabac);
                cbf.cb = val as u8;
                if second {
                    let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
                    se_trace("cbf_cb", val as i64, &self.cabac);
                    cbf.cb |= (val as u8) << 1;
                }
            }
            // Decode cbf_cr if trafo_depth == 0 (always) or parent had cbf_cr
            if trafo_depth == 0 || parent_cbf.cr != 0 {
                let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
                se_trace("cbf_cr", val as i64, &self.cabac);
                cbf.cr = val as u8;
                if second {
                    let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
                    se_trace("cbf_cr", val as i64, &self.cabac);
                    cbf.cr |= (val as u8) << 1;
                }
            }
            cbf
        } else {
            // log2_size == 2: inherit from parent (chroma decoded at parent level)
            parent_cbf
        };

        if split_transform {
            let half = 1u32 << (log2_size - 1);
            let new_depth = trafo_depth + 1;
            let new_log2_size = log2_size - 1;

            self.decode_transform_tree_inner(
                x0,
                y0,
                new_log2_size,
                new_depth,
                intra_luma_mode,
                intra_chroma_mode,
                intra_split_flag,
                inter_split_flag,
                cbf,
                frame,
            )?;
            self.decode_transform_tree_inner(
                x0 + half,
                y0,
                new_log2_size,
                new_depth,
                intra_luma_mode,
                intra_chroma_mode,
                intra_split_flag,
                inter_split_flag,
                cbf,
                frame,
            )?;
            self.decode_transform_tree_inner(
                x0,
                y0 + half,
                new_log2_size,
                new_depth,
                intra_luma_mode,
                intra_chroma_mode,
                intra_split_flag,
                inter_split_flag,
                cbf,
                frame,
            )?;
            self.decode_transform_tree_inner(
                x0 + half,
                y0 + half,
                new_log2_size,
                new_depth,
                intra_luma_mode,
                intra_chroma_mode,
                intra_split_flag,
                inter_split_flag,
                cbf,
                frame,
            )?;

            // For 4:2:0 / 4:2:2, if we split from 8x8 to 4x4, predict + decode
            // chroma now (4x4 luma children can't have chroma TUs; H.265
            // 7.3.8.10 codes it at blkIdx == 3 with the parent's xBase/yBase).
            // For 4:4:4, each child handles its own chroma — skip this.
            // 4:2:2: chroma is half width, FULL height, so the 8x8 luma parent
            // owns a 4x8 chroma region = two 4x4 TBs stacked at cy and cy + 4,
            // decoded top then bottom per component (the bottom block's intra
            // prediction reads the reconstructed top block).
            if log2_size == 3 && frame.chroma_format != 3 {
                let sis = self.sps.strong_intra_smoothing_enabled_flag;
                let is_intra_cu = self.get_pred_mode_at(x0, y0) == PredMode::Intra;
                let scan_order = if is_intra_cu {
                    residual::get_scan_order(2, intra_chroma_mode.as_u8(), 1, false)
                } else {
                    ScanOrder::Diagonal
                };
                let (cx, cy, n_blocks) = if frame.chroma_format == 2 {
                    (x0 / 2, y0, 2)
                } else {
                    (x0 / 2, y0 / 2, 1)
                };

                for c_idx in 1..=2u8 {
                    let cbf_bits = cbf.bits(c_idx);
                    for blk in 0..n_blocks {
                        let by = cy + blk * 4;
                        // Predict (intra only — inter MC already wrote prediction)
                        if is_intra_cu {
                            intra::predict_intra(frame, cx, by, 2, intra_chroma_mode, c_idx, sis)?;
                        }
                        if cbf_bits & (1 << blk) != 0 {
                            self.decode_and_apply_residual(cx, by, 2, c_idx, scan_order, frame)?;
                        }
                    }
                }
            }
        } else {
            // Decode transform unit (leaf node)
            self.decode_transform_unit_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                intra_luma_mode,
                intra_chroma_mode,
                cbf,
                frame,
            )?;
        }

        Ok(())
    }

    /// Decode transform unit at leaf node
    ///
    /// Per libde265's decode_TU(): prediction and reconstruction happen PER TU,
    /// so each TU is fully predicted + reconstructed before the next TU starts.
    /// This ensures subsequent TUs read reconstructed neighbor samples (not just
    /// prediction values) for their own intra prediction.
    #[allow(clippy::too_many_arguments)]
    fn decode_transform_unit_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        _intra_luma_mode: IntraPredMode,
        // Chroma mode is looked up per-leaf-position via `get_intra_chroma_mode_at`
        // (4:4:4 NxN has a distinct chroma mode per quadrant), so the threaded
        // value is no longer read here — kept for signature symmetry.
        _intra_chroma_mode: IntraPredMode,
        cbf: ChromaCbf,
        frame: &mut DecodedFrame,
    ) -> Result<()> {
        let debug_tt = self.debug_ctu;
        let is_intra_cu = self.get_pred_mode_at(x0, y0) == PredMode::Intra;

        // Decode cbf_luma - per H.265 spec 7.3.8.6:
        // cbf_luma is coded if: CuPredMode == MODE_INTRA || trafoDepth != 0 || cbf.any()
        // For inter at trafo_depth==0 without chroma CBF, cbf_luma is implied 1
        // (rqt_root_cbf was already true, so there must be residual somewhere)
        let cbf_luma = if is_intra_cu || trafo_depth != 0 || cbf.any() {
            let ctx_offset = if trafo_depth == 0 { 1 } else { 0 };
            let ctx_idx = context::CBF_LUMA + ctx_offset;
            let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
            se_trace("cbf_luma", val as i64, &self.cabac);
            val
        } else {
            true // Implied: inter, trafo_depth==0, no chroma CBF
        };

        // Per H.265 7.3.8.11: decode cu_qp_delta before residuals
        // Condition: (cbf_luma || cbf.any()) && cu_qp_delta_enabled_flag && !IsCuQpDeltaCoded
        if (cbf_luma || cbf.any())
            && self.pps.cu_qp_delta_enabled_flag
            && !self.is_cu_qp_delta_coded
        {
            let cu_qp_delta_abs = self.decode_cu_qp_delta_abs()?;
            let cu_qp_delta_sign = if cu_qp_delta_abs != 0 {
                self.cabac.decode_bypass()?
            } else {
                0
            };
            self.is_cu_qp_delta_coded = true;
            // H.265 7.4.9.14: CuQpDeltaVal = cu_qp_delta_abs * (1 - 2 * cu_qp_delta_sign_flag)
            // is constrained to the range [-(26 + QpBdOffsetY/2), +(25 + QpBdOffsetY/2)].
            // A crafted CABAC EGk suffix can decode a huge cu_qp_delta_abs; reject it here
            // (rather than let the QPY reconstruction below overflow i32).
            let cu_qp_delta = (cu_qp_delta_abs as i64) * (1 - 2 * cu_qp_delta_sign as i64);
            let qp_bd_offset_y = 6 * (self.sps.bit_depth_y() as i64 - 8);
            if cu_qp_delta < -(26 + qp_bd_offset_y / 2) || cu_qp_delta > 25 + qp_bd_offset_y / 2 {
                return Err(at!(HevcError::InvalidBitstream(
                    "cu_qp_delta out of range [-(26+QpBdOffsetY/2), 25+QpBdOffsetY/2]",
                )));
            }
            self.cu_qp_delta = cu_qp_delta as i32;
            se_trace("cu_qp_delta", self.cu_qp_delta as i64, &self.cabac);

            // Re-derive quantization parameters with the actual delta
            let cu_x = self.cu_base_x;
            let cu_y = self.cu_base_y;
            let cu_log2 = self.cu_log2_size;
            self.decode_quantization_parameters(x0, y0, cu_x, cu_y);
            // Store QPY in the QP map for neighbor lookups
            self.store_qpy(cu_x, cu_y, cu_log2, self.current_qpy);
        }

        // Mark TU boundary and store QP for deblocking
        let tu_size = 1u32 << log2_size;
        frame.mark_tu_boundary(x0, y0, tu_size);
        frame.store_block_qp(x0, y0, tu_size, self.current_qpy as i8);

        // Store CBF for deblocking boundary strength (bS=2 at TU boundaries with coefficients)
        self.store_cbf(x0, y0, tu_size, cbf_luma || cbf.any());

        // Look up intra mode at actual TU position (correct for NxN where sub-TUs differ)
        let actual_luma_mode = self.get_intra_mode_at(x0, y0);
        let sis = self.sps.strong_intra_smoothing_enabled_flag;

        // Predict luma at TU level BEFORE residual application
        // Only for intra CUs — inter CUs already have MC prediction in the frame
        if is_intra_cu {
            intra::predict_intra(frame, x0, y0, log2_size, actual_luma_mode, 0, sis)?;
        }

        let scan_order = if is_intra_cu {
            residual::get_scan_order(log2_size, actual_luma_mode.as_u8(), 0, false)
        } else {
            ScanOrder::Diagonal // Inter always uses diagonal scan
        };

        // Decode and apply luma residuals (adds to prediction already in frame)
        if cbf_luma {
            if debug_tt {
                let (_r, _o) = self.cabac.get_state();
                debug_trace!(
                    "    TT: decoding luma residual at ({},{}) log2={} (r={},o={})",
                    x0,
                    y0,
                    log2_size,
                    _r,
                    _o
                );
            }
            self.decode_and_apply_residual(x0, y0, log2_size, 0, scan_order, frame)?;
        }

        // Decode chroma: predict + residual per component if not handled by parent.
        // For 4:2:0: chroma TU is half the luma TU size, minimum 4x4 (log2=2),
        //   so chroma is only decoded here when log2_size >= 3 (8x8+ luma → 4x4+ chroma).
        //   When log2_size < 3, the parent 8x8 node handles chroma.
        // For 4:2:2: chroma is half width but FULL height — the luma TU maps to
        //   TWO square chroma TBs of half size stacked vertically at chroma
        //   (x0/2, y0) and (x0/2, y0 + (1 << log2TrafoSizeC)); H.265 7.3.8.10
        //   codes cb0, cb1, cr0, cr1 in that order, and 8.4.4.1 reconstructs
        //   each block before predicting the next (the bottom block's top
        //   neighbours are the reconstructed top block). Same 8x8 rule as 4:2:0.
        // For 4:4:4: chroma TU is the same size as luma, always decoded here.
        let is_444 = frame.chroma_format == 3;
        let is_422 = frame.chroma_format == 2;
        let chroma_here = if is_444 {
            true // 4:4:4: chroma always at TU level
        } else {
            log2_size >= 3 // 4:2:0 / 4:2:2: only when luma TU >= 8x8
        };

        if chroma_here {
            let (chroma_log2_size, cx, cy) = if is_444 {
                (log2_size, x0, y0)
            } else if is_422 {
                (log2_size - 1, x0 / 2, y0)
            } else {
                (log2_size - 1, x0 / 2, y0 / 2)
            };
            let n_blocks = if is_422 { 2u32 } else { 1 };
            // For 4:4:4 with NxN partitioning each chroma PB has its own mode
            // (H.265 7.3.8.5); the single `intra_chroma_mode` threaded through
            // the transform tree is only the first quadrant's. Look up the mode
            // stored for THIS leaf position — it equals `intra_chroma_mode` for
            // 2Nx2N/4:2:0, and is the per-quadrant mode for 4:4:4 NxN. Using the
            // wrong mode picks the wrong scan order (Angular 10/26 select
            // vertical/horizontal, not diagonal) and desyncs CABAC.
            let chroma_mode_here = self.get_intra_chroma_mode_at(x0, y0);
            let chroma_scan_order = if is_intra_cu {
                residual::get_scan_order(chroma_log2_size, chroma_mode_here.as_u8(), 1, is_444)
            } else {
                ScanOrder::Diagonal
            };

            for c_idx in 1..=2u8 {
                let cbf_bits = cbf.bits(c_idx);
                for blk in 0..n_blocks {
                    let by = cy + (blk << chroma_log2_size);
                    // Predict (intra only — inter MC already wrote prediction)
                    if is_intra_cu {
                        intra::predict_intra(
                            frame,
                            cx,
                            by,
                            chroma_log2_size,
                            chroma_mode_here,
                            c_idx,
                            sis,
                        )?;
                    }
                    if cbf_bits & (1 << blk) != 0 {
                        self.decode_and_apply_residual(
                            cx,
                            by,
                            chroma_log2_size,
                            c_idx,
                            chroma_scan_order,
                            frame,
                        )?;
                    }
                }
            }
        }
        // Note: for 4:2:0 / 4:2:2, if log2_size < 3, chroma was predicted+decoded by parent when splitting from 8x8

        Ok(())
    }

    /// Decode cu_qp_delta_abs per H.265 section 7.3.8.11
    /// TU prefix (up to 5 context-coded bins) + EGk bypass suffix
    fn decode_cu_qp_delta_abs(&mut self) -> Result<u32> {
        let first_bin = self
            .cabac
            .decode_bin(&mut self.ctx[context::CU_QP_DELTA_ABS])?;
        if first_bin == 0 {
            return Ok(0);
        }
        let mut prefix = 1u32;
        for _ in 0..4 {
            let bin = self
                .cabac
                .decode_bin(&mut self.ctx[context::CU_QP_DELTA_ABS + 1])?;
            if bin == 0 {
                break;
            }
            prefix += 1;
        }
        if prefix == 5 {
            // EGk(0) bypass suffix. decode_egk_bypass(0) can return up to
            // u32::MAX - 1 for a crafted bitstream, so `suffix + 5` can
            // overflow u32 — reject rather than wrap/panic.
            let suffix = self.cabac.decode_egk_bypass(0)?;
            suffix.checked_add(5).ok_or_else(|| {
                at!(HevcError::InvalidBitstream(
                    "cu_qp_delta_abs EGk suffix overflow",
                ))
            })
        } else {
            Ok(prefix)
        }
    }

    /// Decode residual coefficients and apply to frame
    fn decode_and_apply_residual(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        scan_order: ScanOrder,
        frame: &mut DecodedFrame,
    ) -> Result<()> {
        // Decode coefficients via CABAC
        let mut coeff_buf = residual::CoeffBuffer::new(log2_size);
        let transform_skip = residual::decode_residual(
            &mut self.cabac,
            &mut self.ctx,
            log2_size,
            c_idx,
            scan_order,
            self.pps.sign_data_hiding_enabled_flag,
            self.cu_transquant_bypass_flag,
            self.pps.transform_skip_enabled_flag,
            x0,
            y0,
            &mut coeff_buf,
        )?;

        if coeff_buf.is_zero() {
            return Ok(());
        }

        let size = 1usize << log2_size;
        let num_coeffs = size * size;
        let is_intra_cu = self.get_pred_mode_at(x0, y0) == PredMode::Intra;

        // Dequantize coefficients in-place
        let coeffs = &mut coeff_buf.coeffs;

        let (qp, bit_depth) = match c_idx {
            0 => (self.qp_y, self.sps.bit_depth_y()),
            1 => (self.qp_cb, self.sps.bit_depth_c()),
            2 => (self.qp_cr, self.sps.bit_depth_c()),
            _ => (self.qp_y, self.sps.bit_depth_y()),
        };

        // Lossless (H.265 8.6.2): cu_transquant_bypass bypasses both scaling
        // (dequant) and the inverse transform — the parsed coefficient levels
        // ARE the residual samples, added directly to the prediction.
        if self.cu_transquant_bypass_flag {
            // Implicit residual DPCM (H.265 8.6.2): only when the SPS enables it
            // AND the intra prediction is pure horizontal (10) or vertical (26);
            // the residual is then cumulatively summed along that direction.
            let rdpcm = if is_intra_cu && self.sps.implicit_rdpcm_enabled_flag {
                let mode = if c_idx == 0 {
                    self.get_intra_mode_at(x0, y0)
                } else {
                    self.get_intra_chroma_mode_at(x0, y0)
                };
                match mode.as_u8() {
                    10 => 1u8, // horizontal: accumulate along rows
                    26 => 2u8, // vertical: accumulate down columns
                    _ => 0u8,
                }
            } else {
                0u8
            };
            let residual = &mut self.residual_buf;
            residual[..num_coeffs].copy_from_slice(&coeffs[..num_coeffs]);
            match rdpcm {
                1 => {
                    for py in 0..size {
                        let row = py * size;
                        for px in 1..size {
                            residual[row + px] =
                                residual[row + px].wrapping_add(residual[row + px - 1]);
                        }
                    }
                }
                2 => {
                    for py in 1..size {
                        let row = py * size;
                        let prev = (py - 1) * size;
                        for px in 0..size {
                            residual[row + px] =
                                residual[row + px].wrapping_add(residual[prev + px]);
                        }
                    }
                }
                _ => {}
            }
            let max_val = (1i32 << bit_depth) - 1;
            let (plane, stride) = frame.plane_mut(c_idx);
            for py in 0..size {
                let row_start = (y0 as usize + py) * stride + x0 as usize;
                for px in 0..size {
                    let idx = row_start + px;
                    if idx < plane.len() {
                        let pred = plane[idx] as i32;
                        let r = residual[py * size + px] as i32;
                        plane[idx] = (pred + r).clamp(0, max_val) as u16;
                    }
                }
            }
            return Ok(());
        }

        let dequant_params = transform::DequantParams {
            qp,
            bit_depth,
            log2_tr_size: log2_size,
        };

        // Use scaling list if enabled (H.265 8.6.3)
        // Per spec: use PPS scaling list if present, else SPS scaling list
        let scaling_list = if self.sps.scaling_list_enabled_flag && !transform_skip {
            self.pps
                .pps_scaling_list
                .as_ref()
                .or(self.sps.scaling_list.as_ref())
        } else {
            None
        };

        if let Some(sl) = scaling_list {
            // matrixId: intra Y=0, Cb=1, Cr=2; inter Y=3, Cb=4, Cr=5
            let matrix_id = if is_intra_cu { c_idx } else { c_idx + 3 };
            // Build scaling matrix in raster order for this TU (reuse persistent buffer)
            let scaling_matrix = &mut self.scaling_buf;
            for py in 0..size {
                for px in 0..size {
                    scaling_matrix[py * size + px] =
                        sl.get_scaling_factor(log2_size, matrix_id, px as u32, py as u32);
                }
            }
            transform::dequantize_scaled(
                &mut coeffs[..num_coeffs],
                dequant_params,
                &scaling_matrix[..num_coeffs],
            );
        } else {
            transform::dequantize(&mut coeffs[..num_coeffs], dequant_params);
        }

        // Apply inverse transform (or skip for transform_skip mode)
        // Reuse persistent buffer — transform writes all size*size elements, no zeroing needed
        let residual = &mut self.residual_buf;
        if transform_skip {
            // Per H.265 8.6.4.1 / libde265 transform_skip_residual_fallback():
            // tsShift = 5 + Log2(nTbS)
            // bdShift = max(20 - bit_depth, 0)
            // residual = (coeff << tsShift + rnd) >> bdShift
            let ts_shift = 5 + log2_size as i32;
            let bd_shift = (20 - bit_depth as i32).max(0);
            let rnd = if bd_shift > 0 {
                1i32 << (bd_shift - 1)
            } else {
                0
            };
            for i in 0..num_coeffs {
                let c = (coeffs[i] as i32) << ts_shift;
                residual[i] = ((c + rnd) >> bd_shift) as i16;
            }
        } else {
            let is_intra_4x4_luma = log2_size == 2 && c_idx == 0 && is_intra_cu;
            transform::inverse_transform(coeffs, residual, size, bit_depth, is_intra_4x4_luma);
        }

        // Add residual to prediction — single SIMD dispatch for entire block
        let max_val = (1i32 << bit_depth) - 1;
        let (plane, stride) = frame.plane_mut(c_idx);
        let last_row_end = (y0 as usize + size - 1) * stride + x0 as usize + size;
        if last_row_end <= plane.len() {
            incant!(
                add_residual_block(
                    plane,
                    stride,
                    x0 as usize,
                    y0 as usize,
                    residual,
                    size,
                    max_val
                ),
                [v3, neon, wasm128, scalar]
            );
        } else {
            for py in 0..size {
                let row_start = (y0 as usize + py) * stride + x0 as usize;
                for px in 0..size {
                    let idx = row_start + px;
                    if idx < plane.len() {
                        let pred = plane[idx] as i32;
                        let r = residual[py * size + px] as i32;
                        plane[idx] = (pred + r).clamp(0, max_val) as u16;
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode partition mode
    fn decode_part_mode(&mut self, pred_mode: PredMode, log2_cb_size: u8) -> Result<PartMode> {
        if pred_mode == PredMode::Intra {
            // For intra, first bin distinguishes 2Nx2N from NxN
            let ctx_idx = context::PART_MODE;
            let bin = self.cabac.decode_bin(&mut self.ctx[ctx_idx])?;
            se_trace("part_mode", bin as i64, &self.cabac);

            if bin != 0 {
                Ok(PartMode::Part2Nx2N)
            } else {
                // NxN only allowed at minimum CU size
                if log2_cb_size == self.sps.log2_min_cb_size() {
                    Ok(PartMode::PartNxN)
                } else {
                    Err(at!(HevcError::InvalidBitstream(
                        "NxN not allowed at this size"
                    )))
                }
            }
        } else {
            // Inter partition modes (H.265 Table 9-34)
            let ctx_base = context::PART_MODE;
            let min_cb_log2 = self.sps.log2_min_cb_size();
            let amp = self.sps.amp_enabled_flag;

            // First bin: 2Nx2N(1) vs other(0)
            let bin0 = self.cabac.decode_bin(&mut self.ctx[ctx_base])?;
            if bin0 != 0 {
                se_trace("part_mode", 0, &self.cabac);
                return Ok(PartMode::Part2Nx2N);
            }

            // At minimum CU size, fewer modes available
            if log2_cb_size == min_cb_log2 {
                // Second bin distinguishes: 2NxN(1), Nx2N+NxN(0)
                let bin1 = self.cabac.decode_bin(&mut self.ctx[ctx_base + 1])?;
                if bin1 != 0 {
                    se_trace("part_mode", 1, &self.cabac);
                    return Ok(PartMode::Part2NxN);
                }
                // Third bin: Nx2N(1) vs NxN(0)
                if log2_cb_size > 3 {
                    // NxN only for inter at min CU if size > 8
                    let bin2 = self.cabac.decode_bin(&mut self.ctx[ctx_base + 2])?;
                    if bin2 != 0 {
                        se_trace("part_mode", 2, &self.cabac);
                        return Ok(PartMode::PartNx2N);
                    }
                    se_trace("part_mode", 3, &self.cabac);
                    Ok(PartMode::PartNxN)
                } else {
                    se_trace("part_mode", 2, &self.cabac);
                    Ok(PartMode::PartNx2N)
                }
            } else if amp {
                // AMP modes available
                let bin1 = self.cabac.decode_bin(&mut self.ctx[ctx_base + 1])?;
                if bin1 != 0 {
                    // Horizontal: 2NxN or AMP 2NxnU/2NxnD
                    let bin3 = self.cabac.decode_bin(&mut self.ctx[ctx_base + 3])?;
                    if bin3 != 0 {
                        se_trace("part_mode", 1, &self.cabac);
                        return Ok(PartMode::Part2NxN);
                    }
                    let bin_bypass = self.cabac.decode_bypass()?;
                    if bin_bypass == 0 {
                        se_trace("part_mode", 4, &self.cabac);
                        Ok(PartMode::Part2NxnU)
                    } else {
                        se_trace("part_mode", 5, &self.cabac);
                        Ok(PartMode::Part2NxnD)
                    }
                } else {
                    // Vertical: Nx2N or AMP nLx2N/nRx2N
                    let bin3 = self.cabac.decode_bin(&mut self.ctx[ctx_base + 3])?;
                    if bin3 != 0 {
                        se_trace("part_mode", 2, &self.cabac);
                        return Ok(PartMode::PartNx2N);
                    }
                    let bin_bypass = self.cabac.decode_bypass()?;
                    if bin_bypass == 0 {
                        se_trace("part_mode", 6, &self.cabac);
                        Ok(PartMode::PartnLx2N)
                    } else {
                        se_trace("part_mode", 7, &self.cabac);
                        Ok(PartMode::PartnRx2N)
                    }
                }
            } else {
                // No AMP: just 2NxN or Nx2N
                let bin1 = self.cabac.decode_bin(&mut self.ctx[ctx_base + 1])?;
                if bin1 != 0 {
                    se_trace("part_mode", 1, &self.cabac);
                    Ok(PartMode::Part2NxN)
                } else {
                    se_trace("part_mode", 2, &self.cabac);
                    Ok(PartMode::PartNx2N)
                }
            }
        }
    }

    /// Decode intra prediction modes and apply prediction
    /// Returns (luma_mode, chroma_mode)
    fn decode_intra_prediction(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        _apply_chroma: bool,
        frame: &mut DecodedFrame,
    ) -> Result<(IntraPredMode, IntraPredMode)> {
        let (intra_luma_mode, intra_chroma_mode) =
            self.decode_intra_prediction_modes(x0, y0, log2_size, frame)?;

        // Store intra modes in the mode map for neighbor lookups and transform tree
        self.store_intra_mode(x0, y0, log2_size, intra_luma_mode);
        self.store_intra_chroma_mode(x0, y0, log2_size, intra_chroma_mode);

        // NOTE: Prediction is NOT applied here. It happens in decode_transform_unit_leaf
        // and the 8x8→4x4 chroma split handler, so each TU is predicted →
        // reconstructed before the next TU reads its neighbors.

        Ok((intra_luma_mode, intra_chroma_mode))
    }

    /// Decode prev_intra_luma_pred_flag (context-coded bin)
    fn decode_prev_intra_luma_pred_flag(&mut self) -> Result<bool> {
        let ctx_idx = context::PREV_INTRA_LUMA_PRED_FLAG;
        let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
        se_trace("prev_intra_luma_pred", val as i64, &self.cabac);
        Ok(val)
    }

    /// Derive intra luma mode from prev_flag using neighbor-based MPM candidates
    ///
    /// Looks up left and above neighbors from the intra mode map, with proper
    /// CTB row boundary check for the above neighbor (H.265 8.4.2).
    fn derive_intra_luma_mode(
        &mut self,
        x0: u32,
        y0: u32,
        prev_flag: bool,
    ) -> Result<IntraPredMode> {
        let cand_a = self.get_neighbor_intra_mode_left(x0, y0);
        let cand_b = self.get_neighbor_intra_mode_above(x0, y0);
        let mpm = intra::fill_mpm_candidates(cand_a, cand_b);

        if prev_flag {
            let mpm_idx = self.decode_mpm_idx()?;
            Ok(mpm[mpm_idx as usize])
        } else {
            let rem = self.decode_rem_intra_luma_pred_mode()?;
            Ok(self.map_rem_mode_to_intra(rem, &mpm))
        }
    }

    /// Decode intra luma mode: flag + mpm/rem in one call (for Part2Nx2N)
    fn decode_intra_luma_mode(&mut self, x0: u32, y0: u32) -> Result<IntraPredMode> {
        let prev_flag = self.decode_prev_intra_luma_pred_flag()?;
        self.derive_intra_luma_mode(x0, y0, prev_flag)
    }

    /// Decode intra chroma mode
    /// Per HEVC spec Table 8-2 and libde265 map_chroma_pred_mode():
    /// - First bin (context-coded): if 0 → mode 4 (derived from luma)
    /// - If first bin is 1: read 2 fixed-length bypass bits → modes 0-3
    /// - If candidate mode collides with luma mode → Angular34
    fn decode_intra_chroma_mode(&mut self, luma_mode: IntraPredMode) -> Result<IntraPredMode> {
        // Monochrome (ChromaArrayType == 0): intra_chroma_pred_mode is NOT in
        // the bitstream (H.265 7.3.8.5 gates it on ChromaArrayType != 0).
        // Reading it consumes a phantom CABAC bin per CU and desyncs the decode.
        // Chroma is unused for 4:0:0, so return the luma mode without reading.
        let chroma_array_type = if self.sps.separate_colour_plane_flag {
            0
        } else {
            self.sps.chroma_format_idc
        };
        if chroma_array_type == 0 {
            return Ok(luma_mode);
        }
        let ctx_idx = context::INTRA_CHROMA_PRED_MODE;
        let first_bin = self.cabac.decode_bin(&mut self.ctx[ctx_idx])?;
        let derived = if first_bin == 0 {
            // Mode 4: derived from luma
            se_trace("intra_chroma_mode", 4, &self.cabac);
            luma_mode
        } else {
            // Read 2 fixed-length bypass bits for modes 0-3
            let mode_idx = self.cabac.decode_bypass_bits(2)? as u8;
            se_trace("intra_chroma_mode", mode_idx as i64, &self.cabac);

            let candidate = match mode_idx {
                0 => IntraPredMode::Planar,
                1 => IntraPredMode::Angular26, // Vertical
                2 => IntraPredMode::Angular10, // Horizontal
                _ => IntraPredMode::Dc,        // mode_idx == 3
            };

            // Per Table 8-2: if candidate collides with luma mode, use Angular34
            if candidate == luma_mode {
                IntraPredMode::Angular34
            } else {
                candidate
            }
        };

        // H.265 8.4.3: for ChromaArrayType == 2 the Table 8-2 result is remapped
        // through Table 8-3 (chroma is half width, full height, so every angle
        // steepens). The remapped mode is IntraPredModeC — it drives both the
        // chroma prediction and the 4x4 scan order (7.4.9.11), so it is applied
        // here, once, before the mode is stored.
        if chroma_array_type == 2 {
            let mapped = CHROMA_422_MODE_MAP[derived.as_u8() as usize];
            return IntraPredMode::from_u8(mapped).ok_or_else(|| {
                at!(HevcError::DecodingError(
                    "Table 8-3 produced an invalid intra mode",
                ))
            });
        }

        Ok(derived)
    }

    /// Decode intra prediction modes (luma + chroma) for Part2Nx2N
    fn decode_intra_prediction_modes(
        &mut self,
        x0: u32,
        y0: u32,
        _log2_size: u8,
        _frame: &DecodedFrame,
    ) -> Result<(IntraPredMode, IntraPredMode)> {
        let intra_luma_mode = self.decode_intra_luma_mode(x0, y0)?;

        // DEBUG: Print first few intra modes
        if x0 < 16 && y0 < 16 {
            debug_trace!(
                "DEBUG: intra_mode at ({},{}) size={}: mode={:?}",
                x0,
                y0,
                1u32 << _log2_size,
                intra_luma_mode
            );
        }

        let intra_chroma_mode = self.decode_intra_chroma_mode(intra_luma_mode)?;

        Ok((intra_luma_mode, intra_chroma_mode))
    }

    /// Get min PU size (= min_cb_size / 2, at least 1)
    pub fn min_pu_size(&self) -> u32 {
        ((1u32 << self.sps.log2_min_cb_size()) / 2).max(1)
    }

    /// Store intra luma mode for a region (in min_pu_size units)
    fn store_intra_mode(&mut self, x0: u32, y0: u32, log2_size: u8, mode: IntraPredMode) {
        let min_pu = self.min_pu_size();
        let stride = self.intra_mode_map_stride;
        let count = ((1u32 << log2_size) / min_pu).max(1);
        let start_x = x0 / min_pu;
        let start_y = y0 / min_pu;
        for dy in 0..count {
            for dx in 0..count {
                let idx = ((start_y + dy) * stride + (start_x + dx)) as usize;
                if idx < self.intra_mode_map.len() {
                    self.intra_mode_map[idx] = mode.as_u8();
                }
            }
        }
    }

    /// Store intra chroma mode for a region (in min_pu_size units)
    fn store_intra_chroma_mode(&mut self, x0: u32, y0: u32, log2_size: u8, mode: IntraPredMode) {
        let min_pu = self.min_pu_size();
        let stride = self.intra_mode_map_stride;
        let count = ((1u32 << log2_size) / min_pu).max(1);
        let start_x = x0 / min_pu;
        let start_y = y0 / min_pu;
        for dy in 0..count {
            for dx in 0..count {
                let idx = ((start_y + dy) * stride + (start_x + dx)) as usize;
                if idx < self.intra_chroma_mode_map.len() {
                    self.intra_chroma_mode_map[idx] = mode.as_u8();
                }
            }
        }
    }

    /// Get intra luma prediction mode at a sample position
    fn get_intra_mode_at(&self, x: u32, y: u32) -> IntraPredMode {
        let min_pu = self.min_pu_size();
        let stride = self.intra_mode_map_stride;
        let idx = ((y / min_pu) * stride + (x / min_pu)) as usize;
        if idx < self.intra_mode_map.len() {
            IntraPredMode::from_u8(self.intra_mode_map[idx]).unwrap_or(IntraPredMode::Dc)
        } else {
            IntraPredMode::Dc
        }
    }

    /// Get intra chroma prediction mode at a sample position
    #[allow(dead_code)]
    fn get_intra_chroma_mode_at(&self, x: u32, y: u32) -> IntraPredMode {
        let min_pu = self.min_pu_size();
        let stride = self.intra_mode_map_stride;
        let idx = ((y / min_pu) * stride + (x / min_pu)) as usize;
        if idx < self.intra_chroma_mode_map.len() {
            IntraPredMode::from_u8(self.intra_chroma_mode_map[idx]).unwrap_or(IntraPredMode::Dc)
        } else {
            IntraPredMode::Dc
        }
    }

    /// Get intra prediction mode of the left neighbor (x0-1, y0)
    ///
    /// Returns DC if the left neighbor is outside the picture boundary or in a different tile.
    fn get_neighbor_intra_mode_left(&self, x0: u32, y0: u32) -> IntraPredMode {
        if x0 == 0 {
            return IntraPredMode::Dc;
        }
        // Tile boundary: left neighbor in different tile → DC
        if self.pps.tiles_enabled_flag {
            let ctb_size = self.sps.ctb_size();
            let left_ctb_x = (x0 - 1) / ctb_size;
            let curr_ctb_x = x0 / ctb_size;
            if left_ctb_x != curr_ctb_x {
                let curr_tile = get_tile_id(
                    &self.tile_col_bd,
                    &self.tile_row_bd,
                    curr_ctb_x,
                    y0 / ctb_size,
                );
                let left_tile = get_tile_id(
                    &self.tile_col_bd,
                    &self.tile_row_bd,
                    left_ctb_x,
                    y0 / ctb_size,
                );
                if curr_tile != left_tile {
                    return IntraPredMode::Dc;
                }
            }
        }
        self.get_intra_mode_at(x0 - 1, y0)
    }

    /// Get intra prediction mode of the above neighbor (x0, y0-1)
    ///
    /// Returns DC if:
    /// - The above neighbor is outside the picture boundary (y0 == 0)
    /// - The above neighbor is in a different CTB row (H.265 8.4.2 / libde265 intrapred.cc:107-109)
    fn get_neighbor_intra_mode_above(&self, x0: u32, y0: u32) -> IntraPredMode {
        if y0 == 0 {
            return IntraPredMode::Dc;
        }
        // CTB row boundary check: if the above sample is in a different CTB row, use DC
        // This implements: y-1 < ((y >> Log2CtbSizeY) << Log2CtbSizeY)
        let ctb_size = self.sps.ctb_size();
        let ctb_y_start = (y0 / ctb_size) * ctb_size;
        if y0 - 1 < ctb_y_start {
            return IntraPredMode::Dc;
        }
        self.get_intra_mode_at(x0, y0 - 1)
    }

    // -- Inter prediction storage methods --

    /// Store prediction mode for a CU region
    fn store_pred_mode(&mut self, x0: u32, y0: u32, log2_size: u8, mode: PredMode) {
        let min_pu = self.min_pu_size();
        let stride = self.intra_mode_map_stride;
        let count = ((1u32 << log2_size) / min_pu).max(1);
        let start_x = x0 / min_pu;
        let start_y = y0 / min_pu;
        for dy in 0..count {
            for dx in 0..count {
                let idx = ((start_y + dy) * stride + (start_x + dx)) as usize;
                if idx < self.pred_mode_map.len() {
                    self.pred_mode_map[idx] = mode;
                }
            }
        }
    }

    /// Get prediction mode at a sample position
    fn get_pred_mode_at(&self, x: u32, y: u32) -> PredMode {
        let min_pu = self.min_pu_size();
        let stride = self.intra_mode_map_stride;
        let idx = ((y / min_pu) * stride + (x / min_pu)) as usize;
        if idx < self.pred_mode_map.len() {
            self.pred_mode_map[idx]
        } else {
            PredMode::Intra
        }
    }

    /// Store motion information for a PU region
    fn store_mv_info(&mut self, x0: u32, y0: u32, w: u32, h: u32, motion: PbMotion) {
        let min_pu = self.min_pu_size();
        let stride = self.intra_mode_map_stride;
        let sx = x0 / min_pu;
        let sy = y0 / min_pu;
        let cw = (w / min_pu).max(1);
        let ch = (h / min_pu).max(1);
        for dy in 0..ch {
            for dx in 0..cw {
                let idx = ((sy + dy) * stride + (sx + dx)) as usize;
                if idx < self.mv_info.len() {
                    self.mv_info[idx] = motion;
                }
            }
        }
    }

    /// Store CBF (coded block flag) for a TU region at 4x4 granularity
    fn store_cbf(&mut self, x0: u32, y0: u32, size: u32, has_coeffs: bool) {
        let bx = x0 / 4;
        let by = y0 / 4;
        let bs = (size / 4).max(1);
        for dy in 0..bs {
            for dx in 0..bs {
                let idx = ((by + dy) * self.cbf_map_stride + (bx + dx)) as usize;
                if idx < self.cbf_map.len() {
                    self.cbf_map[idx] = has_coeffs;
                }
            }
        }
    }

    // -- Inter prediction resolution and motion compensation --

    /// Build an MvContext for merge/AMVP candidate derivation.
    ///
    /// The returned `col_frame` must be kept alive as long as the `MvContext` is used,
    /// since the context borrows from it. Call as:
    /// ```ignore
    /// let col_frame = self.build_collocated_frame();
    /// let mv_ctx = self.build_mv_context(col_frame.as_ref());
    /// ```
    fn build_collocated_frame(&self) -> Option<CollocatedFrame<'_>> {
        let data = self.collocated_data.as_ref()?;
        Some(CollocatedFrame {
            mv_info: &data.mv_info,
            pred_mode: &data.pred_mode,
            pu_stride: data.pu_stride,
            min_pu_size: data.min_pu_size,
            poc: data.poc,
            ref_poc: data.ref_poc,
        })
    }

    /// Build an MvContext for merge/AMVP candidate derivation
    fn build_mv_context<'c>(&'c self, col: Option<&'c CollocatedFrame<'c>>) -> MvContext<'c> {
        // NoBackwardPredFlag (H.265 7-55): true when all ref POCs <= currPOC
        let no_backward_pred_flag = {
            let mut all_before = true;
            for list in 0..2usize {
                for i in 0..self.ref_pic_lists.num_ref_idx_active[list] as usize {
                    if self.ref_pic_lists.poc[list][i] > self.curr_poc {
                        all_before = false;
                    }
                }
            }
            all_before
        };

        MvContext {
            mv_info: &self.mv_info,
            pred_mode: &self.pred_mode_map,
            pu_stride: self.intra_mode_map_stride,
            min_pu_size: self.min_pu_size(),
            pic_width: self.sps.pic_width_in_luma_samples,
            pic_height: self.sps.pic_height_in_luma_samples,
            curr_poc: self.curr_poc,
            ref_pic_lists: &self.ref_pic_lists,
            is_b_slice: self.header.slice_type == SliceType::B,
            log2_parallel_merge_level: self.pps.log2_parallel_merge_level_minus2 + 2,
            collocated: col,
            ctb_size: self.sps.ctb_size(),
            no_backward_pred_flag,
            collocated_from_l0_flag: self.header.collocated_from_l0_flag,
        }
    }

    /// Resolve coded motion (merge or AMVP) into final PbMotion with real MVs
    #[allow(clippy::too_many_arguments)]
    fn resolve_motion(
        &self,
        coding: &PbMotionCoding,
        px: u32,
        py: u32,
        pw: u32,
        ph: u32,
        part_idx: u8,
        part_mode: PartMode,
    ) -> PbMotion {
        let col_frame = self.build_collocated_frame();
        let mv_ctx = self.build_mv_context(col_frame.as_ref());

        if coding.merge_flag {
            // Merge mode: select from merge candidate list
            let pu_params = MergePuParams {
                xp: px,
                yp: py,
                w: pw,
                h: ph,
                part_idx,
                part_mode,
                max_num_merge_cand: self.header.max_num_merge_cand,
            };
            let cand_list = inter::derive_merge_candidates(&mv_ctx, &pu_params);
            let idx = (coding.merge_idx as usize).min(cand_list.len() - 1);
            let mut motion = cand_list[idx];

            // H.265 8.5.3.2.2 step 10: for small bi-predicted PUs (nPbW+nPbH==12),
            // disable L1 only when BOTH L0 and L1 are active
            if pw + ph == 12 && motion.pred_flag[0] && motion.pred_flag[1] {
                motion.pred_flag[1] = false;
                motion.ref_idx[1] = -1;
            }
            motion
        } else {
            // AMVP mode: derive MVP, add MVD
            let uses_l0 = coding.inter_pred_idc == 1 || coding.inter_pred_idc == 3;
            let uses_l1 = coding.inter_pred_idc == 2 || coding.inter_pred_idc == 3;

            let mut motion = PbMotion::UNAVAILABLE;

            if uses_l0 {
                let mvp_list =
                    inter::derive_amvp_candidates(&mv_ctx, px, py, pw, ph, coding.ref_idx[0], 0);
                let mvp = mvp_list[coding.mvp_l0_flag as usize];
                motion.pred_flag[0] = true;
                motion.ref_idx[0] = coding.ref_idx[0];
                motion.mv[0] = MotionVector {
                    x: mvp.x.wrapping_add(coding.mvd[0][0]),
                    y: mvp.y.wrapping_add(coding.mvd[0][1]),
                };
            }

            if uses_l1 {
                let mvp_list =
                    inter::derive_amvp_candidates(&mv_ctx, px, py, pw, ph, coding.ref_idx[1], 1);
                let mvp = mvp_list[coding.mvp_l1_flag as usize];
                motion.pred_flag[1] = true;
                motion.ref_idx[1] = coding.ref_idx[1];
                motion.mv[1] = MotionVector {
                    x: mvp.x.wrapping_add(coding.mvd[1][0]),
                    y: mvp.y.wrapping_add(coding.mvd[1][1]),
                };
            }

            motion
        }
    }

    /// Apply motion compensation for a PU: fetch from reference frame(s), blend, write to frame
    #[allow(clippy::too_many_arguments)]
    fn apply_mc(
        &self,
        motion: &PbMotion,
        px: u32,
        py: u32,
        pw: u32,
        ph: u32,
        scratch: &mut mc::McScratch,
        frame: &mut DecodedFrame,
    ) {
        let bit_depth = self.sps.bit_depth_y();
        let blk = McBlock {
            xp: px,
            yp: py,
            w: pw,
            h: ph,
            bit_depth,
        };
        let buf_size = (pw * ph) as usize;

        // Get reference frame for a given list
        let get_ref_frame = |list: usize| -> Option<&DecodedFrame> {
            if !motion.pred_flag[list] || motion.ref_idx[list] < 0 {
                return None;
            }
            let ref_idx = motion.ref_idx[list] as usize;
            let dpb_idx = self.ref_pic_lists.dpb_index[list].get(ref_idx)?;
            if *dpb_idx < 0 {
                return None;
            }
            self.ref_frames.get(*dpb_idx as usize)?.as_ref()
        };

        let ref_l0 = get_ref_frame(0);
        let ref_l1 = get_ref_frame(1);

        // If no reference frames are available, fill with neutral value to avoid UNINIT
        if ref_l0.is_none() && ref_l1.is_none() {
            let neutral_y = if bit_depth == 8 {
                128u16
            } else {
                1u16 << (bit_depth - 1)
            };
            let stride = frame.width as usize;
            for j in 0..ph {
                for i in 0..pw {
                    let idx = (py + j) as usize * stride + (px + i) as usize;
                    if idx < frame.y_plane.len() {
                        frame.y_plane[idx] = neutral_y;
                    }
                }
            }
            return;
        }

        let is_bi = motion.pred_flag[0] && motion.pred_flag[1];

        // Luma MC — use stack buffers (max 64x64 = 4096 samples)
        let mut pred_l0_buf = [0i16; 4096];
        let mut pred_l1_buf = [0i16; 4096];
        let pred0 = &mut pred_l0_buf[..buf_size];
        let pred1 = &mut pred_l1_buf[..buf_size];

        if is_bi
            && let Some(r0) = ref_l0
            && let Some(r1) = ref_l1
        {
            mc::mc_luma(r0, motion.mv[0], &blk, pred0, true, scratch);
            mc::mc_luma(r1, motion.mv[1], &blk, pred1, true, scratch);
            mc::blend_bi(pred0, pred1, &mut frame.y_plane, frame.width as usize, &blk);
        } else {
            let (ref_frame, mv) = if motion.pred_flag[0] {
                (ref_l0, motion.mv[0])
            } else {
                (ref_l1, motion.mv[1])
            };
            if let Some(rf) = ref_frame {
                mc::mc_luma(rf, mv, &blk, pred0, false, scratch);
                mc::blend_uni(pred0, &mut frame.y_plane, frame.width as usize, &blk);
            }
        }

        // Chroma MC (4:2:0, 4:2:2, 4:4:4)
        if self.sps.chroma_format_idc > 0 {
            let (sub_x, sub_y) = match self.sps.chroma_format_idc {
                1 => (2u32, 2u32),
                2 => (2, 1),
                3 => (1, 1),
                _ => (2, 2),
            };
            let cpw = pw / sub_x;
            let cph = ph / sub_y;
            let cpx = px / sub_x;
            let cpy = py / sub_y;
            let cblk = McBlock {
                xp: cpx,
                yp: cpy,
                w: cpw,
                h: cph,
                bit_depth: self.sps.bit_depth_c(),
            };
            let cbuf_size = (cpw * cph) as usize;
            let mut cpred0 = [0i16; 1024];
            let mut cpred1 = [0i16; 1024];

            for c_idx in 0..2u8 {
                let mc_one = |rf: &DecodedFrame,
                              mv: MotionVector,
                              pred: &mut [i16],
                              bi: bool,
                              sc: &mut mc::McScratch| {
                    let (plane, stride) = rf.plane(c_idx + 1);
                    let (_, c_height) = rf.chroma_dims();
                    let cref = ChromaRef {
                        plane,
                        stride,
                        height: c_height,
                        sub_x,
                        sub_y,
                    };
                    mc::mc_chroma(&cref, mv, &cblk, pred, bi, sc);
                };

                let (plane_mut, plane_stride) = frame.plane_mut(c_idx + 1);

                if is_bi {
                    if let (Some(r0), Some(r1)) = (ref_l0, ref_l1) {
                        mc_one(r0, motion.mv[0], &mut cpred0[..cbuf_size], true, scratch);
                        mc_one(r1, motion.mv[1], &mut cpred1[..cbuf_size], true, scratch);
                        mc::blend_bi(
                            &cpred0[..cbuf_size],
                            &cpred1[..cbuf_size],
                            plane_mut,
                            plane_stride,
                            &cblk,
                        );
                    }
                } else {
                    let (rf, mv) = if motion.pred_flag[0] {
                        (ref_l0, motion.mv[0])
                    } else {
                        (ref_l1, motion.mv[1])
                    };
                    if let Some(rf) = rf {
                        mc_one(rf, mv, &mut cpred0[..cbuf_size], false, scratch);
                        mc::blend_uni(&cpred0[..cbuf_size], plane_mut, plane_stride, &cblk);
                    }
                }
            }
        }
    }

    // -- Inter CU/PU CABAC syntax decoders --

    /// Decode cu_skip_flag (H.265 9.3.4.2.1)
    /// Contexts 4-6, depends on left and above neighbor skip status
    fn decode_cu_skip_flag(&mut self, x0: u32, y0: u32) -> Result<bool> {
        let ctx_inc = self.derive_cu_skip_ctx(x0, y0);
        let ctx_idx = context::CU_SKIP_FLAG + ctx_inc;
        let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
        se_trace("cu_skip_flag", val as i64, &self.cabac);
        Ok(val)
    }

    /// Derive context increment for cu_skip_flag from neighbors (H.265 9.3.4.2.2)
    /// Uses check_CTB_available logic: neighbor must be in picture, same slice, same tile
    fn derive_cu_skip_ctx(&self, x0: u32, y0: u32) -> usize {
        let mut ctx_inc = 0;
        // Left neighbor: available if in picture bounds and same slice/tile
        if self.is_neighbor_available(x0 as i32 - 1, y0 as i32)
            && self.get_pred_mode_at(x0 - 1, y0) == PredMode::Skip
        {
            ctx_inc += 1;
        }
        // Above neighbor: available if in picture bounds and same slice/tile
        // NOT restricted to same CTB row — the above CTB row is already decoded
        if self.is_neighbor_available(x0 as i32, y0 as i32 - 1)
            && self.get_pred_mode_at(x0, y0 - 1) == PredMode::Skip
        {
            ctx_inc += 1;
        }
        ctx_inc
    }

    /// Decode pred_mode_flag (H.265 7.3.8.5)
    /// Context 8, 0 = Inter, 1 = Intra
    fn decode_pred_mode_flag(&mut self) -> Result<PredMode> {
        let ctx_idx = context::PRED_MODE_FLAG;
        let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])?;
        se_trace("pred_mode_flag", val as i64, &self.cabac);
        if val != 0 {
            Ok(PredMode::Intra)
        } else {
            Ok(PredMode::Inter)
        }
    }

    /// Decode merge_flag (context 20)
    fn decode_merge_flag(&mut self) -> Result<bool> {
        let ctx_idx = context::MERGE_FLAG;
        let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
        se_trace("merge_flag", val as i64, &self.cabac);
        Ok(val)
    }

    /// Decode merge_idx (context 21 + bypass bins)
    fn decode_merge_idx(&mut self, max_cand: u8) -> Result<u8> {
        if max_cand <= 1 {
            return Ok(0);
        }
        let ctx_idx = context::MERGE_IDX;
        let first = self.cabac.decode_bin(&mut self.ctx[ctx_idx])?;
        se_trace("merge_idx", first as i64, &self.cabac);
        if first == 0 {
            return Ok(0);
        }
        // Remaining bins are bypass (truncated unary)
        let mut idx = 1u8;
        while idx < max_cand - 1 {
            let bit = self.cabac.decode_bypass()?;
            if bit == 0 {
                break;
            }
            idx += 1;
        }
        Ok(idx)
    }

    /// Decode inter_pred_idc (contexts 15-19)
    fn decode_inter_pred_idc(&mut self, ct_depth: u8, pu_w: u32, pu_h: u32) -> Result<u8> {
        // For small blocks (nPbW + nPbH == 12, i.e. 4x8 or 8x4), only L0/L1 (1 bin)
        if pu_w + pu_h == 12 {
            let ctx_idx = context::INTER_PRED_IDC + 4; // ctx 19
            let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])?;
            se_trace("inter_pred_idc", val as i64, &self.cabac);
            return Ok(if val == 0 { 1 } else { 2 }); // L0=1, L1=2
        }
        // First bin: Bi(1) vs uni(0)
        let ctx_idx = context::INTER_PRED_IDC + (ct_depth as usize).min(3);
        let first = self.cabac.decode_bin(&mut self.ctx[ctx_idx])?;
        if first != 0 {
            se_trace("inter_pred_idc", 3, &self.cabac);
            return Ok(3); // Bi
        }
        // Second bin: L0(0) vs L1(1)
        let ctx_idx2 = context::INTER_PRED_IDC + 4;
        let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx2])?;
        let idc: u8 = if val == 0 { 1 } else { 2 };
        se_trace("inter_pred_idc", idc as i64, &self.cabac);
        Ok(idc)
    }

    /// Decode ref_idx (truncated unary, contexts 23-24 + bypass)
    ///
    /// Binarization: truncated unary with cMax = num_active - 1
    /// - Bin 0: context REF_IDX
    /// - Bin 1: context REF_IDX + 1
    /// - Bins 2+: bypass
    fn decode_ref_idx(&mut self, num_active: u8) -> Result<i8> {
        if num_active <= 1 {
            return Ok(0);
        }
        let c_max = num_active as i8 - 1;

        // First bin (context-coded)
        let first = self.cabac.decode_bin(&mut self.ctx[context::REF_IDX])?;
        if first == 0 || c_max == 1 {
            let idx = if first == 0 { 0 } else { 1 };
            se_trace("ref_idx", idx, &self.cabac);
            return Ok(idx as i8);
        }
        // Second bin (context-coded)
        let second = self.cabac.decode_bin(&mut self.ctx[context::REF_IDX + 1])?;
        if second == 0 || c_max == 2 {
            let idx = if second == 0 { 1 } else { 2 };
            se_trace("ref_idx", idx, &self.cabac);
            return Ok(idx as i8);
        }
        // Remaining bins are bypass (truncated unary)
        let mut idx = 2i8;
        while idx < c_max {
            let bit = self.cabac.decode_bypass()?;
            if bit == 0 {
                break;
            }
            idx += 1;
        }
        se_trace("ref_idx", idx as i64, &self.cabac);
        Ok(idx)
    }

    /// Decode MVD (motion vector difference) for one component (H.265 7.3.8.9)
    /// Returns (mvd_x, mvd_y)
    fn decode_mvd(&mut self) -> Result<(i16, i16)> {
        // abs_mvd_greater0_flag for x and y (both use same context per H.265 9.3.3)
        let ctx_gt0 = context::ABS_MVD_GREATER0_FLAG;
        let abs_gt0_x = self.cabac.decode_bin(&mut self.ctx[ctx_gt0])? != 0;
        let abs_gt0_y = self.cabac.decode_bin(&mut self.ctx[ctx_gt0])? != 0;

        // abs_mvd_greater1_flag for x and y (both use same context, next index)
        let ctx_gt1 = context::ABS_MVD_GREATER0_FLAG + 1;
        let abs_gt1_x = if abs_gt0_x {
            self.cabac.decode_bin(&mut self.ctx[ctx_gt1])? != 0
        } else {
            false
        };
        let abs_gt1_y = if abs_gt0_y {
            self.cabac.decode_bin(&mut self.ctx[ctx_gt1])? != 0
        } else {
            false
        };

        // abs_mvd_minus2 (EGk bypass) + sign
        let mvd_x = if abs_gt0_x {
            let abs_val = if abs_gt1_x {
                let rem = self.cabac.decode_egk_bypass(1)?;
                (rem as i32) + 2
            } else {
                1
            };
            let sign = self.cabac.decode_bypass()?;
            if sign != 0 {
                -(abs_val as i16)
            } else {
                abs_val as i16
            }
        } else {
            0
        };

        let mvd_y = if abs_gt0_y {
            let abs_val = if abs_gt1_y {
                let rem = self.cabac.decode_egk_bypass(1)?;
                (rem as i32) + 2
            } else {
                1
            };
            let sign = self.cabac.decode_bypass()?;
            if sign != 0 {
                -(abs_val as i16)
            } else {
                abs_val as i16
            }
        } else {
            0
        };

        se_trace(
            "mvd",
            ((mvd_x as i64) << 16) | (mvd_y as u16 as i64),
            &self.cabac,
        );
        Ok((mvd_x, mvd_y))
    }

    /// Decode mvp_lx_flag (context 22)
    fn decode_mvp_lx_flag(&mut self) -> Result<bool> {
        let ctx_idx = context::MVP_LX_FLAG;
        let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
        se_trace("mvp_lx_flag", val as i64, &self.cabac);
        Ok(val)
    }

    /// Decode rqt_root_cbf for inter CUs (context: CBF_LUMA offset 1)
    fn decode_rqt_root_cbf(&mut self) -> Result<bool> {
        // Use ABS_MVD_GREATER1_FLAG slot (index 27) repurposed for rqt_root_cbf
        // Init value = 79 (set in INIT_VALUES_P/B)
        let ctx_idx = context::ABS_MVD_GREATER1_FLAG;
        let val = self.cabac.decode_bin(&mut self.ctx[ctx_idx])? != 0;
        se_trace("rqt_root_cbf", val as i64, &self.cabac);
        Ok(val)
    }

    /// Decode prediction unit syntax for a single inter PU
    fn decode_inter_pu(
        &mut self,
        _x0: u32,
        _y0: u32,
        w: u32,
        h: u32,
        ct_depth: u8,
        merge: bool,
    ) -> Result<PbMotionCoding> {
        let mut coding = PbMotionCoding::default();

        if merge {
            coding.merge_flag = true;
            coding.merge_idx = self.decode_merge_idx(self.header.max_num_merge_cand)?;
            return Ok(coding);
        }

        coding.merge_flag = self.decode_merge_flag()?;
        if coding.merge_flag {
            coding.merge_idx = self.decode_merge_idx(self.header.max_num_merge_cand)?;
            return Ok(coding);
        }

        // AMVP mode
        let is_b = self.header.slice_type == SliceType::B;
        if is_b {
            coding.inter_pred_idc = self.decode_inter_pred_idc(ct_depth, w, h)?;
        } else {
            coding.inter_pred_idc = 1; // P-slice: always L0
        }

        let uses_l0 = coding.inter_pred_idc == 1 || coding.inter_pred_idc == 3;
        let uses_l1 = coding.inter_pred_idc == 2 || coding.inter_pred_idc == 3;

        // L0
        if uses_l0 {
            coding.ref_idx[0] = self.decode_ref_idx(self.header.num_ref_idx_l0_active)?;
            let (mvd_x, mvd_y) = self.decode_mvd()?;
            coding.mvd[0] = [mvd_x, mvd_y];
            coding.mvp_l0_flag = self.decode_mvp_lx_flag()?;
        }

        // L1
        if uses_l1 {
            coding.ref_idx[1] = self.decode_ref_idx(self.header.num_ref_idx_l1_active)?;
            if !self.header.mvd_l1_zero_flag || coding.inter_pred_idc != 3 {
                let (mvd_x, mvd_y) = self.decode_mvd()?;
                coding.mvd[1] = [mvd_x, mvd_y];
            }
            coding.mvp_l1_flag = self.decode_mvp_lx_flag()?;
        }

        Ok(coding)
    }

    /// Map rem_intra_luma_pred_mode to actual mode (excluding MPM candidates)
    fn map_rem_mode_to_intra(&self, rem: u32, mpm: &[IntraPredMode; 3]) -> IntraPredMode {
        // Sort MPM candidates
        let mut mpm_vals = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
        mpm_vals.sort_unstable();

        // Map remaining mode
        let mut mode = rem as u8;
        for &mpm_val in &mpm_vals {
            if mode >= mpm_val {
                mode += 1;
            }
        }

        IntraPredMode::from_u8(mode).unwrap_or(IntraPredMode::Dc)
    }

    /// Decode mpm_idx (0, 1, or 2)
    fn decode_mpm_idx(&mut self) -> Result<u8> {
        // Truncated unary: 0, 10, 11
        let val = if self.cabac.decode_bypass()? == 0 {
            0
        } else if self.cabac.decode_bypass()? == 0 {
            1
        } else {
            2
        };
        se_trace("mpm_idx", val as i64, &self.cabac);
        Ok(val)
    }

    /// Decode rem_intra_luma_pred_mode (5 bits)
    fn decode_rem_intra_luma_pred_mode(&mut self) -> Result<u32> {
        let mut val = 0u32;
        for _ in 0..5 {
            val = (val << 1) | self.cabac.decode_bypass()? as u32;
        }
        se_trace("rem_intra_luma", val as i64, &self.cabac);
        Ok(val)
    }

    /// H.265 Table 8-6: chroma QP mapping for 4:2:0
    /// H.265 8.6.1: qPi → QpC. Table 8-10 applies only to ChromaArrayType == 1
    /// (4:2:0); for 4:2:2 / 4:4:4 QpC = Min(qPi, 51).
    fn chroma_qp_from_luma(qpi: i32, chroma_array_type: u8) -> i32 {
        static TAB8_22: [i32; 13] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];
        if chroma_array_type != 1 {
            qpi.min(51)
        } else if qpi < 30 {
            qpi
        } else if qpi >= 43 {
            qpi - 6
        } else {
            TAB8_22[(qpi - 30) as usize]
        }
    }

    /// Get QPY at a sample position from the QP map
    fn get_qpy_at(&self, x: u32, y: u32) -> i32 {
        let min_tb = 1u32 << self.sps.log2_min_tb_size();
        let idx = ((y / min_tb) * self.qp_map_stride + (x / min_tb)) as usize;
        if idx < self.qp_map.len() {
            self.qp_map[idx] as i32
        } else {
            self.header.slice_qp_y
        }
    }

    /// Store QPY for a CU region in the QP map
    fn store_qpy(&mut self, x0: u32, y0: u32, log2_cb_size: u8, qpy: i32) {
        let min_tb = 1u32 << self.sps.log2_min_tb_size();
        let count = ((1u32 << log2_cb_size) / min_tb).max(1);
        let start_x = x0 / min_tb;
        let start_y = y0 / min_tb;
        for dy in 0..count {
            for dx in 0..count {
                let idx = ((start_y + dy) * self.qp_map_stride + (start_x + dx)) as usize;
                if idx < self.qp_map.len() {
                    self.qp_map[idx] = qpy as i8;
                }
            }
        }
    }

    /// Decode quantization parameters (H.265 section 8.6.1)
    /// Matching libde265's decode_quantization_parameters()
    fn decode_quantization_parameters(
        &mut self,
        x0: u32,
        _y0: u32,
        x_cu_base: u32,
        y_cu_base: u32,
    ) {
        // saturating: a crafted PPS can set diff_cu_qp_delta_depth > log2_ctb_size
        // (the underflow would panic under overflow-checks / fuzz on untrusted input).
        let log2_min_cu_qp_delta_size = self
            .sps
            .log2_ctb_size()
            .saturating_sub(self.pps.diff_cu_qp_delta_depth);
        let qg_mask = (1u32 << log2_min_cu_qp_delta_size) - 1;

        // Top-left pixel of current quantization group
        let x_qg = (x_cu_base & !qg_mask) as i32;
        let y_qg = (y_cu_base & !qg_mask) as i32;

        // Track QG transitions
        if x_qg != self.current_qg_x || y_qg != self.current_qg_y {
            self.last_qpy_in_prev_qg = self.current_qpy;
            self.current_qg_x = x_qg;
            self.current_qg_y = y_qg;
        }

        // Determine QP prediction
        let ctb_mask = ((1u32 << self.sps.log2_ctb_size()) - 1) as i32;
        let first_in_ctb_row = x_qg == 0 && (y_qg & ctb_mask) == 0;

        let first_ctb_in_slice = self.header.slice_segment_address;
        let slice_start_x = (first_ctb_in_slice % self.sps.pic_width_in_ctbs()) as i32
            * (1 << self.sps.log2_ctb_size());
        let slice_start_y = (first_ctb_in_slice / self.sps.pic_width_in_ctbs()) as i32
            * (1 << self.sps.log2_ctb_size());
        let first_qg_in_slice = slice_start_x == x_qg && slice_start_y == y_qg;

        // H.265 8.6.1: first QG in tile also uses slice_qp_y
        let first_qg_in_tile = self.pps.tiles_enabled_flag && {
            let ctb_size = self.sps.ctb_size();
            let ctb_x = x_qg as u32 / ctb_size;
            let ctb_y = y_qg as u32 / ctb_size;
            // Check if current CTB is the first CTB of its tile
            self.tile_col_bd.windows(2).any(|w| w[0] == ctb_x)
                && self.tile_row_bd.windows(2).any(|w| w[0] == ctb_y)
                && (x_qg as u32).is_multiple_of(ctb_size)
                && (y_qg as u32).is_multiple_of(ctb_size)
        };

        let qp_y_pred = if first_qg_in_slice
            || first_qg_in_tile
            || (first_in_ctb_row && self.pps.entropy_coding_sync_enabled_flag)
        {
            self.header.slice_qp_y
        } else {
            self.last_qpy_in_prev_qg
        };

        // Get neighbor QP values for averaging
        let qp_y_a = if x_qg > 0 {
            // Check if left neighbor is in same CTB
            let left_x = (x_qg - 1) as u32;
            let left_y = y_qg as u32;
            // Simplified: check if in same CTB
            let ctb_size = self.sps.ctb_size();
            let our_ctb_x = x0 / ctb_size;
            let left_ctb_x = left_x / ctb_size;
            if our_ctb_x == left_ctb_x || (x_qg as u32) < ctb_size {
                // Left neighbor might be in previous CTB, use prediction
                if left_ctb_x == self.ctb_x {
                    self.get_qpy_at(left_x, left_y)
                } else {
                    qp_y_pred
                }
            } else {
                qp_y_pred
            }
        } else {
            qp_y_pred
        };

        let qp_y_b = if y_qg > 0 {
            let above_x = x_qg as u32;
            let above_y = (y_qg - 1) as u32;
            let ctb_size = self.sps.ctb_size();
            let above_ctb_y = above_y / ctb_size;
            if above_ctb_y == self.ctb_y {
                self.get_qpy_at(above_x, above_y)
            } else {
                qp_y_pred
            }
        } else {
            qp_y_pred
        };

        let qp_y_pred = (qp_y_a + qp_y_b + 1) >> 1;

        // Compute final QPY
        let qp_bd_offset_y = 6 * (self.sps.bit_depth_y() as i32 - 8);
        let qpy = ((qp_y_pred + self.cu_qp_delta + 52 + 2 * qp_bd_offset_y)
            % (52 + qp_bd_offset_y))
            - qp_bd_offset_y;

        self.qp_y = qpy + qp_bd_offset_y;
        if self.qp_y < 0 {
            self.qp_y = 0;
        }

        // Compute chroma QP (H.265 8.6.1; Table 8-10 only for 4:2:0)
        let chroma_array_type = if self.sps.separate_colour_plane_flag {
            0
        } else {
            self.sps.chroma_format_idc
        };
        let qp_bd_offset_c = 6 * (self.sps.bit_depth_c() as i32 - 8);
        let qpi_cb =
            (qpy + self.pps.pps_cb_qp_offset as i32 + self.header.slice_cb_qp_offset as i32)
                .clamp(-qp_bd_offset_c, 57);
        let qpi_cr =
            (qpy + self.pps.pps_cr_qp_offset as i32 + self.header.slice_cr_qp_offset as i32)
                .clamp(-qp_bd_offset_c, 57);

        self.qp_cb = Self::chroma_qp_from_luma(qpi_cb, chroma_array_type) + qp_bd_offset_c;
        self.qp_cr = Self::chroma_qp_from_luma(qpi_cr, chroma_array_type) + qp_bd_offset_c;

        self.current_qpy = qpy;
    }

    /// Decode PCM samples (H.265 7.3.8.8)
    ///
    /// Reads raw (uncompressed) luma and chroma samples from the bitstream.
    /// After pcm_flag=1 via decode_terminate(), the CABAC engine is terminated.
    /// This function reads raw samples and reinitializes CABAC.
    fn decode_pcm_samples(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        frame: &mut DecodedFrame,
    ) -> Result<()> {
        let pcm = self.sps.pcm_params.as_ref().ok_or_else(|| {
            at!(HevcError::InvalidBitstream(
                "pcm_flag set but SPS has no PCM params",
            ))
        })?;
        let pcm_bit_depth_luma = pcm.pcm_sample_bit_depth_luma_minus1 as u32 + 1;
        let pcm_bit_depth_chroma = pcm.pcm_sample_bit_depth_chroma_minus1 as u32 + 1;

        let cb_size = 1u32 << log2_cb_size;

        // After decode_terminate(), CABAC is in a terminated state.
        // Byte-align and get the current position for raw sample reading.
        // The CABAC byte_pos is already at the right spot after terminate.
        let mut byte_pos = self.cabac.get_position().0;

        // Byte alignment for PCM (skip any remaining bits in current byte)
        // After decode_terminate, the position is byte-aligned already
        // but there may be pcm_alignment_zero_bits

        // Read raw PCM luma samples
        let data = self.cabac.raw_data();
        let bit_depth_y = self.sps.bit_depth_y();
        let stride = frame.width as usize;

        // Use a simple bit reader on the raw data
        let mut bit_offset = 0u32; // within current byte

        // Helper: read N bits from data starting at byte_pos, bit_offset
        let read_bits = |pos: &mut usize, bo: &mut u32, n: u32| -> u16 {
            let mut val = 0u16;
            for _ in 0..n {
                if *pos < data.len() {
                    let bit = (data[*pos] >> (7 - *bo)) & 1;
                    val = (val << 1) | bit as u16;
                    *bo += 1;
                    if *bo >= 8 {
                        *bo = 0;
                        *pos += 1;
                    }
                }
            }
            val
        };

        // Read luma samples
        for dy in 0..cb_size {
            for dx in 0..cb_size {
                let sample = read_bits(&mut byte_pos, &mut bit_offset, pcm_bit_depth_luma);
                // Scale to bit_depth_y if needed
                let px = x0 + dx;
                let py = y0 + dy;
                if px < frame.width && py < frame.height {
                    let idx = py as usize * stride + px as usize;
                    frame.y_plane[idx] = if pcm_bit_depth_luma < bit_depth_y as u32 {
                        sample << (bit_depth_y as u32 - pcm_bit_depth_luma)
                    } else {
                        sample >> (pcm_bit_depth_luma - bit_depth_y as u32)
                    };
                }
            }
        }

        // Read chroma samples (4:2:0)
        let (chroma_w, chroma_h) = match self.sps.chroma_format_idc {
            0 => (0, 0),
            1 => (cb_size / 2, cb_size / 2),
            2 => (cb_size / 2, cb_size),
            3 => (cb_size, cb_size),
            _ => (cb_size / 2, cb_size / 2),
        };
        let chroma_stride = match self.sps.chroma_format_idc {
            0 => 0,
            1 => frame.width.div_ceil(2) as usize,
            2 => frame.width.div_ceil(2) as usize,
            3 => frame.width as usize,
            _ => frame.width.div_ceil(2) as usize,
        };
        let bit_depth_c = self.sps.bit_depth_c();

        // Cb
        let cx0 = match self.sps.chroma_format_idc {
            1 | 2 => x0 / 2,
            _ => x0,
        };
        let cy0 = match self.sps.chroma_format_idc {
            1 => y0 / 2,
            _ => y0,
        };
        for dy in 0..chroma_h {
            for dx in 0..chroma_w {
                let sample = read_bits(&mut byte_pos, &mut bit_offset, pcm_bit_depth_chroma);
                let px = cx0 + dx;
                let py = cy0 + dy;
                if (px as usize) < chroma_stride && py < frame.height.div_ceil(2) {
                    let idx = py as usize * chroma_stride + px as usize;
                    if idx < frame.cb_plane.len() {
                        frame.cb_plane[idx] = if pcm_bit_depth_chroma < bit_depth_c as u32 {
                            sample << (bit_depth_c as u32 - pcm_bit_depth_chroma)
                        } else {
                            sample >> (pcm_bit_depth_chroma - bit_depth_c as u32)
                        };
                    }
                }
            }
        }

        // Cr
        for dy in 0..chroma_h {
            for dx in 0..chroma_w {
                let sample = read_bits(&mut byte_pos, &mut bit_offset, pcm_bit_depth_chroma);
                let px = cx0 + dx;
                let py = cy0 + dy;
                if (px as usize) < chroma_stride && py < frame.height.div_ceil(2) {
                    let idx = py as usize * chroma_stride + px as usize;
                    if idx < frame.cr_plane.len() {
                        frame.cr_plane[idx] = if pcm_bit_depth_chroma < bit_depth_c as u32 {
                            sample << (bit_depth_c as u32 - pcm_bit_depth_chroma)
                        } else {
                            sample >> (pcm_bit_depth_chroma - bit_depth_c as u32)
                        };
                    }
                }
            }
        }

        // Byte-align for CABAC reinit
        if bit_offset > 0 {
            byte_pos += 1;
        }

        // Reinitialize CABAC at the current byte position
        self.cabac.seek_to(byte_pos);
        self.cabac.reinit();

        Ok(())
    }
}
