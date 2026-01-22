//! Backwards tracing from bad pixels to CABAC operations
//!
//! This test finds pixels with large RGB differences and traces
//! backwards through the decoding pipeline to find the source.

use heic_decoder::HeicDecoder;
use std::path::Path;

const WASM_MODULE: &str = "/home/lilith/work/heic/wasm-module/heic_decoder.wasm";
const TEST_IMAGE: &str = "/home/lilith/work/heic/libheif/examples/example.heic";

/// Threshold for "bad" pixel - RGB component difference
const BAD_PIXEL_THRESHOLD: i32 = 50;

/// Structure to hold pixel location and error info
#[derive(Debug, Clone)]
struct BadPixel {
    /// X coordinate in output image
    x: u32,
    /// Y coordinate in output image
    y: u32,
    /// Reference RGB values
    ref_rgb: [u8; 3],
    /// Our RGB values
    our_rgb: [u8; 3],
    /// Maximum component difference
    max_diff: i32,
}

/// Information about which TU produced a pixel
#[derive(Debug, Clone)]
struct PixelProvenance {
    /// CTU index (raster scan order)
    ctu_idx: u32,
    /// CTU X position (in CTU units)
    ctu_x: u32,
    /// CTU Y position (in CTU units)
    ctu_y: u32,
    /// Position within CTU (luma)
    local_x: u32,
    local_y: u32,
    /// Chroma position (for 4:2:0)
    chroma_x: u32,
    chroma_y: u32,
}

fn load_reference_decoder() -> heic_wasm_rs::HeicDecoder {
    heic_wasm_rs::HeicDecoder::from_file(Path::new(WASM_MODULE))
        .expect("Failed to load WASM decoder")
}

/// Find all pixels with large differences
fn find_bad_pixels(ref_rgb: &[u8], our_rgb: &[u8], width: u32, height: u32) -> Vec<BadPixel> {
    let mut bad_pixels = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;

            let ref_r = ref_rgb[idx] as i32;
            let ref_g = ref_rgb[idx + 1] as i32;
            let ref_b = ref_rgb[idx + 2] as i32;

            let our_r = our_rgb[idx] as i32;
            let our_g = our_rgb[idx + 1] as i32;
            let our_b = our_rgb[idx + 2] as i32;

            let diff_r = (ref_r - our_r).abs();
            let diff_g = (ref_g - our_g).abs();
            let diff_b = (ref_b - our_b).abs();
            let max_diff = diff_r.max(diff_g).max(diff_b);

            if max_diff >= BAD_PIXEL_THRESHOLD {
                bad_pixels.push(BadPixel {
                    x,
                    y,
                    ref_rgb: [ref_rgb[idx], ref_rgb[idx + 1], ref_rgb[idx + 2]],
                    our_rgb: [our_rgb[idx], our_rgb[idx + 1], our_rgb[idx + 2]],
                    max_diff,
                });
            }
        }
    }

    bad_pixels
}

/// Map a pixel coordinate to its CTU/TU provenance
fn pixel_to_provenance(x: u32, y: u32, ctb_size: u32, pic_width_ctb: u32) -> PixelProvenance {
    let ctu_x = x / ctb_size;
    let ctu_y = y / ctb_size;
    let ctu_idx = ctu_y * pic_width_ctb + ctu_x;

    let local_x = x % ctb_size;
    let local_y = y % ctb_size;

    // For 4:2:0 chroma
    let chroma_x = x / 2;
    let chroma_y = y / 2;

    PixelProvenance {
        ctu_idx,
        ctu_x,
        ctu_y,
        local_x,
        local_y,
        chroma_x,
        chroma_y,
    }
}

#[test]
fn find_first_bad_pixel() {
    let ref_decoder = load_reference_decoder();
    let our_decoder = HeicDecoder::new();

    let data = std::fs::read(TEST_IMAGE).expect("Failed to read test file");

    let ref_image = ref_decoder.decode(&data).expect("Reference decode failed");
    let our_image = our_decoder.decode(&data).expect("Our decode failed");

    assert_eq!(ref_image.width, our_image.width);
    assert_eq!(ref_image.height, our_image.height);

    let width = our_image.width;
    let height = our_image.height;

    // Image params (assuming standard 64x64 CTU)
    let ctb_size = 64u32;
    let pic_width_ctb = width.div_ceil(ctb_size);
    let pic_height_ctb = height.div_ceil(ctb_size);

    println!("\n=== Bad Pixel Analysis ===");
    println!("Image: {}x{}", width, height);
    println!("CTU grid: {}x{} ({}x{} CTUs)", pic_width_ctb, pic_height_ctb, ctb_size, ctb_size);
    println!("Threshold: {} (RGB component diff)", BAD_PIXEL_THRESHOLD);
    println!();

    let bad_pixels = find_bad_pixels(&ref_image.data, &our_image.data, width, height);

    if bad_pixels.is_empty() {
        println!("No bad pixels found (all within threshold)");
        return;
    }

    println!("Found {} bad pixels", bad_pixels.len());

    // Show first 10 bad pixels
    println!("\nFirst 10 bad pixels:");
    for (i, bp) in bad_pixels.iter().take(10).enumerate() {
        let prov = pixel_to_provenance(bp.x, bp.y, ctb_size, pic_width_ctb);
        println!(
            "  {:2}. ({:4},{:4}) diff={:3} ref=RGB({:3},{:3},{:3}) ours=RGB({:3},{:3},{:3})",
            i + 1, bp.x, bp.y, bp.max_diff,
            bp.ref_rgb[0], bp.ref_rgb[1], bp.ref_rgb[2],
            bp.our_rgb[0], bp.our_rgb[1], bp.our_rgb[2],
        );
        println!(
            "      CTU[{:3}] at ({:2},{:2}), local=({:2},{:2}), chroma=({:3},{:3})",
            prov.ctu_idx, prov.ctu_x, prov.ctu_y,
            prov.local_x, prov.local_y,
            prov.chroma_x, prov.chroma_y,
        );
    }

    // Group by CTU
    let mut ctu_errors: std::collections::HashMap<u32, Vec<&BadPixel>> = std::collections::HashMap::new();
    for bp in &bad_pixels {
        let prov = pixel_to_provenance(bp.x, bp.y, ctb_size, pic_width_ctb);
        ctu_errors.entry(prov.ctu_idx).or_default().push(bp);
    }

    println!("\nCTUs with bad pixels: {}", ctu_errors.len());

    // Find first CTU with errors
    let first_bad_ctu = ctu_errors.keys().copied().min().unwrap();
    let first_ctu_errors = &ctu_errors[&first_bad_ctu];
    let first_prov = pixel_to_provenance(first_ctu_errors[0].x, first_ctu_errors[0].y, ctb_size, pic_width_ctb);

    println!("\nFirst CTU with errors: CTU[{}] at ({},{})",
        first_bad_ctu, first_prov.ctu_x, first_prov.ctu_y);
    println!("  {} bad pixels in this CTU", first_ctu_errors.len());
    println!("  Worst error: {}", first_ctu_errors.iter().map(|p| p.max_diff).max().unwrap());

    // Show error distribution by CTU column
    println!("\nBad pixels by CTU column:");
    let mut col_counts: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for bp in &bad_pixels {
        let prov = pixel_to_provenance(bp.x, bp.y, ctb_size, pic_width_ctb);
        *col_counts.entry(prov.ctu_x).or_default() += 1;
    }
    for (col, count) in &col_counts {
        let bar_len = (*count as f64 / bad_pixels.len() as f64 * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  col {:2}: {:6} {}", col, count, bar);
    }
}

/// Examine YCbCr values at the first bad pixel location
#[test]
fn examine_ycbcr_at_bad_pixel() {
    let our_decoder = HeicDecoder::new();

    let data = std::fs::read(TEST_IMAGE).expect("Failed to read test file");

    // Get the raw YCbCr frame
    let frame = our_decoder.decode_to_frame(&data).expect("Decode to frame failed");

    println!("\n=== YCbCr Analysis at Key Positions ===");
    println!("Frame: {}x{} (cropped: {}x{})",
        frame.width, frame.height,
        frame.cropped_width(), frame.cropped_height());
    println!("Bit depth: {}, Chroma format: {}", frame.bit_depth, frame.chroma_format);

    // Analyze a few positions
    let positions = [
        (0, 0),
        (64, 0),      // Start of CTU column 1
        (128, 0),     // Start of CTU column 2
        (104, 0),     // Where we know there's a bad coefficient
        (200, 200),   // Middle of image
    ];

    println!("\nYCbCr values at key positions:");
    for (x, y) in positions {
        if x < frame.width && y < frame.height {
            let y_val = frame.get_y(x, y);
            let cx = x / 2;
            let cy = y / 2;
            let cb_val = frame.get_cb(cx, cy);
            let cr_val = frame.get_cr(cx, cy);

            println!("  ({:4},{:4}): Y={:3} Cb={:3} Cr={:3}",
                x, y, y_val, cb_val, cr_val);
        }
    }

    // Analyze chroma plane averages by CTU row
    println!("\nChroma averages by CTU row (first 5 rows):");
    let ctb_size = 64u32;
    let chroma_ctb = ctb_size / 2;  // For 4:2:0

    for ctu_row in 0..5 {
        let cy_start = ctu_row * chroma_ctb;
        let cy_end = (cy_start + chroma_ctb).min(frame.height / 2);

        let mut cb_sum = 0u64;
        let mut cr_sum = 0u64;
        let mut count = 0u64;

        for cy in cy_start..cy_end {
            for cx in 0..frame.width / 2 {
                cb_sum += frame.get_cb(cx, cy) as u64;
                cr_sum += frame.get_cr(cx, cy) as u64;
                count += 1;
            }
        }

        if count > 0 {
            println!("  Row {}: Cb_avg={:3} Cr_avg={:3}",
                ctu_row, cb_sum / count, cr_sum / count);
        }
    }
}

/// Compare Y plane values between our decoder and reference
/// (We can compute Y from reference RGB approximately)
#[test]
fn compare_y_plane_approximation() {
    let ref_decoder = load_reference_decoder();
    let our_decoder = HeicDecoder::new();

    let data = std::fs::read(TEST_IMAGE).expect("Failed to read test file");

    let ref_image = ref_decoder.decode(&data).expect("Reference decode failed");
    let frame = our_decoder.decode_to_frame(&data).expect("Decode to frame failed");

    // Approximate Y from reference RGB using BT.601:
    // Y = 0.299*R + 0.587*G + 0.114*B
    // Or with integer math: Y = (77*R + 150*G + 29*B) >> 8

    println!("\n=== Y Plane Comparison (RGB->Y approximation) ===");

    let width = ref_image.width;
    let height = ref_image.height;

    // Find first position where Y differs significantly
    let mut first_y_diff: Option<(u32, u32, u16, u16)> = None;

    for y in 0..height {
        for x in 0..width {
            let rgb_idx = ((y * width + x) * 3) as usize;
            let r = ref_image.data[rgb_idx] as u32;
            let g = ref_image.data[rgb_idx + 1] as u32;
            let b = ref_image.data[rgb_idx + 2] as u32;

            // Approximate Y from reference RGB
            let ref_y = ((77 * r + 150 * g + 29 * b) >> 8) as u16;

            // Our Y value (accounting for cropping)
            let our_y = frame.get_y(x + frame.crop_left, y + frame.crop_top);

            let diff = (ref_y as i32 - our_y as i32).abs();

            if diff > 20 && first_y_diff.is_none() {
                first_y_diff = Some((x, y, ref_y, our_y));
            }
        }
    }

    if let Some((x, y, ref_y, our_y)) = first_y_diff {
        let ctb_size = 64u32;
        let ctu_x = x / ctb_size;
        let ctu_y = y / ctb_size;

        println!("First significant Y difference at ({}, {}):", x, y);
        println!("  Reference Y (from RGB): {}", ref_y);
        println!("  Our Y: {}", our_y);
        println!("  CTU: ({}, {})", ctu_x, ctu_y);
    } else {
        println!("No significant Y differences found (all within threshold)");
    }
}
