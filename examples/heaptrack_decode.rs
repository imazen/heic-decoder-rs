//! Heaptrack harness for decode-from-bytes allocation profiling.
//!
//! Profiles the production-critical path: `DecoderConfig::decode_request(&bytes)
//! .decode()` — decoding an HEIC/HEIF file (untrusted input) all the way to RGBA8
//! pixels. The goal is to surface allocation *pathologies* that don't show up in a
//! wall-clock benchmark: a high allocation *count* relative to image size, per-pixel
//! or per-CTU mallocs, large transient peaks, or unbounded growth across repeated
//! decodes (a leak). High allocation churn hurts most under contended allocators
//! (Windows, multi-threaded servers) where a single decode of an untrusted upload
//! turns into thousands of lock round-trips.
//!
//! Usage:
//!   cargo build -p heic --release --example heaptrack_decode --features backend-rust,std
//!   heaptrack ./target/release/examples/heaptrack_decode                 # default fixture
//!   heaptrack ./target/release/examples/heaptrack_decode <file.heic> [iters]
//!
//! Set `HEIC_PROFILE_RGBA_OUT` to retain the first decoded RGBA8 buffer for exact comparisons.
//!
//! Then inspect:
//!   heaptrack_print heaptrack.heaptrack_decode.*.zst | less
//!
//! Defaults to the bundled `testdata/libheif-examples/example.heic` (1280x854 real
//! photo, a grid of 6x 512x512 HEVC tiles) decoded 8 times — a meaningful CTU/tile
//! count so the allocation count can be judged relative to image size. A large
//! fixture should be decoded fewer times (pass a smaller `iters`).

use std::hint::black_box;
use std::path::{Path, PathBuf};

use heic::{DecoderConfig, ImageInfo, PixelLayout};

/// Resolve the default bundled fixture relative to the crate manifest so the
/// example runs from any working directory.
fn default_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("libheif-examples")
        .join("example.heic")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let path: PathBuf = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None => default_fixture(),
    };
    // Default 8 iterations; a leak shows up as monotonic growth across them, and a
    // healthy decoder's steady-state per-decode allocation count is iterations-stable.
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    let data = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", path.display());
        std::process::exit(1);
    });

    // Probe once so the report can state the dimensions the alloc count is relative to.
    match ImageInfo::from_bytes(&data) {
        Ok(info) => {
            // `output_buffer_size` takes `self` by value; clone so the field reads below stay valid.
            let rgba_bytes = info
                .clone()
                .output_buffer_size(PixelLayout::Rgba8)
                .unwrap_or(0);
            eprintln!("fixture: {} ({} bytes on disk)", path.display(), data.len());
            eprintln!(
                "  decoded image: {}x{} ({:.2} MP), bit_depth {}, chroma {}, alpha {}",
                info.width,
                info.height,
                (info.width as f64 * info.height as f64) / 1.0e6,
                info.bit_depth,
                match info.chroma_format {
                    0 => "4:0:0",
                    1 => "4:2:0",
                    2 => "4:2:2",
                    3 => "4:4:4",
                    _ => "?",
                },
                info.has_alpha,
            );
            eprintln!(
                "  RGBA8 output buffer: {} bytes ({:.2} MiB)",
                rgba_bytes,
                rgba_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        Err(e) => {
            eprintln!("probe failed for {}: {e}", path.display());
            std::process::exit(1);
        }
    }

    eprintln!("decoding {iters}x via DecoderConfig::decode_request(..).decode() ...");

    // Reusable config — exercises the steady-state per-decode allocation profile.
    let config = DecoderConfig::new();
    let mut total_pixels: u64 = 0;
    for i in 0..iters {
        let output = config
            .decode_request(&data)
            .with_output_layout(PixelLayout::Rgba8)
            .decode()
            .unwrap_or_else(|e| {
                eprintln!("decode iteration {i} failed: {e}");
                std::process::exit(1);
            });
        if i == 0
            && let Some(path) = std::env::var_os("HEIC_PROFILE_RGBA_OUT")
        {
            std::fs::write(path, &output.data).expect("write explicit RGBA reference output");
        }
        total_pixels += u64::from(output.width) * u64::from(output.height);
        // Consume the decoded buffer so the optimizer can't elide the decode or the
        // allocation of the output Vec.
        black_box(&output.data);
        black_box(output.width);
        black_box(output.height);
    }

    eprintln!("done: decoded {total_pixels} total pixels across {iters} iterations");
}
