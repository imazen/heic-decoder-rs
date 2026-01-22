//! HEVC/H.265 decoder
//!
//! This module implements the HEVC (High Efficiency Video Coding) decoder
//! for decoding HEIC still images.

mod bitstream;
mod cabac;
mod ctu;
pub mod debug;
mod intra;
mod params;
mod picture;
mod residual;
mod slice;
mod transform;

pub use picture::DecodedFrame;

use crate::error::HevcError;
use crate::heif::HevcDecoderConfig;
use alloc::vec::Vec;

type Result<T> = core::result::Result<T, HevcError>;

/// Decode HEVC bitstream to pixels (Annex B or raw format)
pub fn decode(data: &[u8]) -> Result<DecodedFrame> {
    // Parse NAL units
    let nal_units = bitstream::parse_nal_units(data)?;
    decode_nal_units(&nal_units)
}

/// Decode HEVC from HEIC container (config + image data)
///
/// This is the preferred method for HEIC files where parameter sets
/// are stored separately in the hvcC box.
pub fn decode_with_config(config: &HevcDecoderConfig, image_data: &[u8]) -> Result<DecodedFrame> {
    let mut nal_units = Vec::new();

    // Parse parameter sets from hvcC
    for nal_data in &config.nal_units {
        if let Ok(nal) = bitstream::parse_single_nal(nal_data) {
            nal_units.push(nal);
        }
    }

    // Parse slice data with correct length size
    let length_size = (config.length_size_minus_one + 1) as usize;
    let mut slice_nals = bitstream::parse_length_prefixed_ext(image_data, length_size)?;
    nal_units.append(&mut slice_nals);

    decode_nal_units(&nal_units)
}

/// Get image info from HEIC config
pub fn get_info_from_config(config: &HevcDecoderConfig) -> Result<ImageInfo> {
    for nal_data in &config.nal_units {
        if let Ok(nal) = bitstream::parse_single_nal(nal_data)
            && nal.nal_type == bitstream::NalType::SpsNut
        {
            let sps = params::parse_sps(&nal.payload)?;
            let (width, height) = get_cropped_dimensions(&sps);
            return Ok(ImageInfo { width, height });
        }
    }
    Err(HevcError::MissingParameterSet("SPS"))
}

/// Internal: decode from parsed NAL units
fn decode_nal_units(nal_units: &[bitstream::NalUnit<'_>]) -> Result<DecodedFrame> {
    // Find and parse parameter sets
    let mut _vps = None;
    let mut sps = None;
    let mut pps = None;

    for nal in nal_units {
        match nal.nal_type {
            bitstream::NalType::VpsNut => {
                _vps = Some(params::parse_vps(&nal.payload)?);
            }
            bitstream::NalType::SpsNut => {
                sps = Some(params::parse_sps(&nal.payload)?);
            }
            bitstream::NalType::PpsNut => {
                pps = Some(params::parse_pps(&nal.payload)?);
            }
            _ => {}
        }
    }

    let sps = sps.ok_or(HevcError::MissingParameterSet("SPS"))?;
    let pps = pps.ok_or(HevcError::MissingParameterSet("PPS"))?;

    // Create frame buffer
    let mut frame = DecodedFrame::new(
        sps.pic_width_in_luma_samples,
        sps.pic_height_in_luma_samples,
    );

    // Set conformance window cropping from SPS
    // Offsets are in units of SubWidthC/SubHeightC, need to convert to luma samples
    if sps.conformance_window_flag {
        let (sub_width_c, sub_height_c) = match sps.chroma_format_idc {
            0 => (1, 1), // Monochrome
            1 => (2, 2), // 4:2:0
            2 => (2, 1), // 4:2:2
            3 => (1, 1), // 4:4:4
            _ => (2, 2), // Default to 4:2:0
        };
        frame.set_crop(
            sps.conf_win_offset.0 * sub_width_c,  // left
            sps.conf_win_offset.1 * sub_width_c,  // right
            sps.conf_win_offset.2 * sub_height_c, // top
            sps.conf_win_offset.3 * sub_height_c, // bottom
        );
    }

    // Decode slice data
    for nal in nal_units {
        if nal.nal_type.is_slice() {
            decode_slice(nal, &sps, &pps, &mut frame)?;
        }
    }

    Ok(frame)
}

/// Get image info without full decoding
pub fn get_info(data: &[u8]) -> Result<ImageInfo> {
    let nal_units = bitstream::parse_nal_units(data)?;

    for nal in &nal_units {
        if nal.nal_type == bitstream::NalType::SpsNut {
            let sps = params::parse_sps(&nal.payload)?;
            let (width, height) = get_cropped_dimensions(&sps);
            return Ok(ImageInfo { width, height });
        }
    }

    Err(HevcError::MissingParameterSet("SPS"))
}

/// Calculate cropped dimensions from SPS conformance window
fn get_cropped_dimensions(sps: &params::Sps) -> (u32, u32) {
    if sps.conformance_window_flag {
        let (sub_width_c, sub_height_c) = match sps.chroma_format_idc {
            0 => (1, 1), // Monochrome
            1 => (2, 2), // 4:2:0
            2 => (2, 1), // 4:2:2
            3 => (1, 1), // 4:4:4
            _ => (2, 2), // Default to 4:2:0
        };
        let crop_left = sps.conf_win_offset.0 * sub_width_c;
        let crop_right = sps.conf_win_offset.1 * sub_width_c;
        let crop_top = sps.conf_win_offset.2 * sub_height_c;
        let crop_bottom = sps.conf_win_offset.3 * sub_height_c;
        (
            sps.pic_width_in_luma_samples - crop_left - crop_right,
            sps.pic_height_in_luma_samples - crop_top - crop_bottom,
        )
    } else {
        (
            sps.pic_width_in_luma_samples,
            sps.pic_height_in_luma_samples,
        )
    }
}

/// Image info from SPS
#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

fn decode_slice(
    nal: &bitstream::NalUnit<'_>,
    sps: &params::Sps,
    pps: &params::Pps,
    frame: &mut DecodedFrame,
) -> Result<()> {
    // 1. Parse slice header and get data offset
    let parse_result = slice::SliceHeader::parse(nal, sps, pps)?;
    let slice_header = parse_result.header;
    let data_offset = parse_result.data_offset;

    // Verify this is an I-slice (required for HEIC still images)
    if !slice_header.slice_type.is_intra() {
        return Err(HevcError::Unsupported(
            "only I-slices supported for still images",
        ));
    }

    // 2. Get slice data (after header)
    // Use the offset from slice header parsing to skip the header bytes
    let slice_data = &nal.payload[data_offset..];

    // 3. Create slice context and decode CTUs
    let mut ctx = ctu::SliceContext::new(sps, pps, &slice_header, slice_data)?;

    // 4. Decode all CTUs in the slice
    ctx.decode_slice(frame)?;

    // 5. TODO: Apply in-loop filters (deblocking, SAO)

    Ok(())
}
