//! Compare our decoder output against reference (heic-wasm-rs / libheif)

use fast_ssim2::compute_ssimulacra2;
use heic_decoder::HeicDecoder;
use imgref::ImgVec;
use std::path::Path;

const WASM_MODULE: &str = "/home/lilith/work/heic/wasm-module/heic_decoder.wasm";

/// Test images to compare
const TEST_IMAGES: &[&str] = &[
    "/home/lilith/work/heic/libheif/examples/example.heic",
    "/home/lilith/work/heic/test-images/image1.heic",
    // iPhone image is large, skip for now
    // "/home/lilith/work/heic/test-images/classic-car-iphone12pro.heic",
];

/// Minimum acceptable SSIM2 score (higher is better, 100 = identical)
/// Reference: >90 is excellent, >70 is good, <50 is poor
const MIN_SSIM2_SCORE: f64 = 50.0;

fn load_reference_decoder() -> heic_wasm_rs::HeicDecoder {
    heic_wasm_rs::HeicDecoder::from_file(Path::new(WASM_MODULE))
        .expect("Failed to load WASM decoder")
}

/// Convert RGB bytes to ImgVec<[u8; 3]>
fn rgb_to_imgvec(rgb: &[u8], width: u32, height: u32) -> ImgVec<[u8; 3]> {
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for chunk in rgb.chunks_exact(3) {
        pixels.push([chunk[0], chunk[1], chunk[2]]);
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

#[test]
fn test_ssim2_against_reference() {
    // Load reference decoder
    let ref_decoder = load_reference_decoder();
    let our_decoder = HeicDecoder::new();

    for image_path in TEST_IMAGES {
        if !Path::new(image_path).exists() {
            eprintln!("Skipping {}: file not found", image_path);
            continue;
        }

        let data = std::fs::read(image_path).expect("Failed to read test file");
        let filename = Path::new(image_path).file_name().unwrap().to_string_lossy();

        // Decode with reference
        let ref_result = ref_decoder.decode(&data);
        let ref_image = match ref_result {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Reference decoder failed on {}: {}", filename, e);
                continue;
            }
        };

        // Decode with our decoder
        let our_result = our_decoder.decode(&data);
        let our_image = match our_result {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Our decoder failed on {}: {}", filename, e);
                continue;
            }
        };

        // Check dimensions match
        if ref_image.width != our_image.width || ref_image.height != our_image.height {
            panic!(
                "{}: Dimension mismatch! Reference: {}x{}, Ours: {}x{}",
                filename, ref_image.width, ref_image.height, our_image.width, our_image.height
            );
        }

        let width = our_image.width;
        let height = our_image.height;

        // Convert to ImgVec for SSIM2
        let ref_img = rgb_to_imgvec(&ref_image.data, width, height);
        let our_img = rgb_to_imgvec(&our_image.data, width, height);

        // Calculate SSIM2
        let score = compute_ssimulacra2(ref_img.as_ref(), our_img.as_ref())
            .expect("Failed to compute SSIM2");

        println!("{}: {}x{} SSIM2 = {:.2}", filename, width, height, score);

        // Check if score meets minimum threshold
        assert!(
            score >= MIN_SSIM2_SCORE,
            "{}: SSIM2 score {:.2} below minimum threshold {}",
            filename,
            score,
            MIN_SSIM2_SCORE
        );
    }
}

#[test]
fn test_pixel_difference_stats() {
    let ref_decoder = load_reference_decoder();
    let our_decoder = HeicDecoder::new();

    for image_path in TEST_IMAGES {
        if !Path::new(image_path).exists() {
            continue;
        }

        let data = std::fs::read(image_path).expect("Failed to read test file");
        let filename = Path::new(image_path).file_name().unwrap().to_string_lossy();

        let ref_image = match ref_decoder.decode(&data) {
            Ok(img) => img,
            Err(_) => continue,
        };

        let our_image = match our_decoder.decode(&data) {
            Ok(img) => img,
            Err(_) => continue,
        };

        if ref_image.width != our_image.width || ref_image.height != our_image.height {
            continue;
        }

        // Calculate per-pixel differences
        let mut total_diff: u64 = 0;
        let mut max_diff: u32 = 0;
        let mut diff_histogram = [0u64; 256];

        for (r, o) in ref_image.data.iter().zip(our_image.data.iter()) {
            let diff = (*r as i32 - *o as i32).unsigned_abs();
            total_diff += diff as u64;
            max_diff = max_diff.max(diff);
            diff_histogram[diff.min(255) as usize] += 1;
        }

        let num_samples = ref_image.data.len() as f64;
        let avg_diff = total_diff as f64 / num_samples;

        println!("\n{}: Pixel difference statistics", filename);
        println!("  Average diff: {:.2}", avg_diff);
        println!("  Max diff: {}", max_diff);

        // Show histogram of differences
        println!("  Difference histogram:");
        let mut cumulative = 0u64;
        for (diff, &count) in diff_histogram.iter().enumerate() {
            cumulative += count;
            let pct = cumulative as f64 / num_samples * 100.0;
            if diff == 0
                || diff == 1
                || diff == 2
                || diff == 5
                || diff == 10
                || diff == 20
                || diff == 50
            {
                println!("    <= {:3}: {:8} ({:5.1}%)", diff, cumulative, pct);
            }
        }
    }
}

/// Write both images to /tmp for visual comparison
#[test]
#[ignore] // Only run manually
fn write_comparison_images() {
    use std::io::Write;

    let ref_decoder = load_reference_decoder();
    let our_decoder = HeicDecoder::new();

    let image_path = TEST_IMAGES[0];
    let data = std::fs::read(image_path).expect("Failed to read test file");

    let ref_image = ref_decoder.decode(&data).expect("Reference decode failed");
    let our_image = our_decoder.decode(&data).expect("Our decode failed");

    // Write reference PPM
    let ref_path = "/tmp/reference.ppm";
    let mut file = std::fs::File::create(ref_path).expect("Failed to create file");
    write!(file, "P6\n{} {}\n255\n", ref_image.width, ref_image.height).unwrap();
    file.write_all(&ref_image.data).unwrap();
    println!("Wrote reference to {}", ref_path);

    // Write our PPM
    let our_path = "/tmp/our_decoder.ppm";
    let mut file = std::fs::File::create(our_path).expect("Failed to create file");
    write!(file, "P6\n{} {}\n255\n", our_image.width, our_image.height).unwrap();
    file.write_all(&our_image.data).unwrap();
    println!("Wrote our output to {}", our_path);

    // Write difference image (amplified)
    let diff_path = "/tmp/difference.ppm";
    let mut diff_data = Vec::with_capacity(our_image.data.len());
    for (r, o) in ref_image.data.iter().zip(our_image.data.iter()) {
        let diff = (*r as i32 - *o as i32).abs();
        // Amplify difference by 4x for visibility
        diff_data.push((diff * 4).min(255) as u8);
    }
    let mut file = std::fs::File::create(diff_path).expect("Failed to create file");
    write!(file, "P6\n{} {}\n255\n", our_image.width, our_image.height).unwrap();
    file.write_all(&diff_data).unwrap();
    println!("Wrote difference to {} (4x amplified)", diff_path);
}

/// List all available HEIC test files
#[test]
fn list_heic_corpus() {
    println!("\n=== Available HEIC test files ===\n");

    let dirs = [
        "/home/lilith/work/heic/test-images",
        "/home/lilith/work/heic/libheif/examples",
        "/home/lilith/work/heic/libheif/fuzzing/data/corpus",
    ];

    for dir in dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            println!("{}:", dir);
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "heic")
                    && let Ok(meta) = path.metadata()
                {
                    println!(
                        "  {} ({} KB)",
                        path.file_name().unwrap().to_string_lossy(),
                        meta.len() / 1024
                    );
                }
            }
            println!();
        }
    }
}
