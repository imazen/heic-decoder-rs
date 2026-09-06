# heic-decoder development tasks

# Run unit tests
test:
    cargo test --lib

# Run all tests (requires test files + heic-wasm-rs)
test-all:
    cargo test

# Run tests with parallel feature
test-parallel:
    cargo test --lib --features parallel

# Run the Windows-native backend tests (MediaFoundation + D3D11VA) on the
# Windows HOST via powershell.exe. Only does real work under WSL (the
# backends are target_os=windows and need a real GPU + the HEVC codec, neither
# of which exists in WSL); a no-op elsewhere. See scripts/win-test.ps1.
# Set HEIC_SKIP_WIN_HOST_TESTS=1 to skip during fast iteration.
test-win:
    cargo test --test backend_dispatch windows_backends_via_host -- --nocapture

# Check all feature permutations
feature-check:
    cargo check
    cargo check --features parallel
    cargo check --features zencodec
    cargo check --no-default-features

# Clippy
clippy:
    cargo clippy --all-targets --features parallel -- -D warnings

# Format (also regenerates the public-API surface snapshots).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots (docs/public-api/) only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Local CI sanity check. Under WSL, `test-win` also runs the Windows-native
# backends (MF + D3D11VA) on the Windows host; elsewhere it's a no-op.
ci: fmt clippy feature-check test test-parallel test-win

# ── Release ──────────────────────────────────────────────────────────────
#
# Single-command publish for the entire workspace. Reads the version
# from `[workspace.package].version` in Cargo.toml — the human bumps
# that BEFORE running these (one edit + commit + push + wait-for-CI).
#
# `just publish-dry` runs everything through `cargo publish --dry-run`
# without touching crates.io or pushing tags. Safe to run any time.
#
# `just publish` is the real ship: runs the full validation flow,
# tags the commit, creates a GitHub release, and uploads every
# crate to crates.io in topological order. Prompts for an explicit
# "PUBLISH" confirmation before the irreversible step.
#
# Both delegate to `scripts/release.sh`. See its docstring for the
# full sequence + env-var knobs (PUBLISH_DRY, PUBLISH_SKIP_CI,
# PUBLISH_FORCE).

# Validate the full publish flow without touching crates.io
publish-dry:
    PUBLISH_DRY=true ./scripts/release.sh

# Publish every workspace crate to crates.io (REAL — irreversible)
publish:
    ./scripts/release.sh

# Show the current workspace version (helps you remember what `just publish` will ship)
version:
    @awk '/^\[workspace\.package\]/{f=1; next} /^\[/{f=0} f && /^version = "/{print}' Cargo.toml

# Run reference comparison tests (requires test files)
compare:
    cargo test --test compare_reference -- --nocapture

# Write comparison images (requires test files)
compare-images:
    cargo test --test compare_reference write_comparison_images -- --nocapture --ignored

# Run benchmarks
bench *ARGS:
    cargo bench {{ARGS}}

# --- Fuzzing ---

# Run all fuzzers for N seconds each (default 120)
fuzz seconds="120":
    #!/usr/bin/env bash
    set -euo pipefail
    for target in fuzz_probe fuzz_decode fuzz_hevc_raw fuzz_decode_av1 fuzz_decode_unci; do
        echo "=== $target ({{seconds}}s) ==="
        mkdir -p fuzz/corpus/$target
        cp fuzz/regression/* fuzz/corpus/$target/ 2>/dev/null || true
        cargo +nightly fuzz run $target \
            -- -max_total_time={{seconds}} -dict=fuzz/heif.dict -rss_limit_mb=2048 -jobs=1 \
            || { echo "CRASH in $target"; exit 1; }
    done
    echo "All fuzzers clean."

# Run a single fuzzer (e.g., just fuzz-one fuzz_hevc_raw 300)
fuzz-one target seconds="120":
    mkdir -p fuzz/corpus/{{target}}
    cp fuzz/regression/* fuzz/corpus/{{target}}/ 2>/dev/null || true
    cargo +nightly fuzz run {{target}} \
        -- -max_total_time={{seconds}} -dict=fuzz/heif.dict -rss_limit_mb=2048

# Minimize all fuzz corpora (dedup by coverage)
fuzz-cmin:
    #!/usr/bin/env bash
    set -euo pipefail
    for target in fuzz_probe fuzz_decode fuzz_hevc_raw fuzz_decode_av1 fuzz_decode_unci; do
        if [ -d "fuzz/corpus/$target" ]; then
            before=$(ls fuzz/corpus/$target/ | wc -l)
            cargo +nightly fuzz cmin $target -- -rss_limit_mb=2048
            after=$(ls fuzz/corpus/$target/ | wc -l)
            echo "$target: $before → $after seeds"
        fi
    done

# Generate fuzz coverage report
fuzz-coverage:
    #!/usr/bin/env bash
    set -euo pipefail
    LLVM_PROFDATA=$(find ~/.rustup -name "llvm-profdata" -path "*/nightly*" -type f | head -1)
    LLVM_COV=$(find ~/.rustup -name "llvm-cov" -path "*/nightly*" -type f | head -1)
    rm -rf /tmp/fuzz_cov_combined && mkdir -p /tmp/fuzz_cov_combined
    cd fuzz
    RUSTFLAGS="-C instrument-coverage" cargo +nightly build \
        --bin fuzz_decode --bin fuzz_probe --bin fuzz_hevc_raw
    cd ..
    for target in fuzz_probe fuzz_decode fuzz_hevc_raw; do
        if [ -d "fuzz/corpus/$target" ]; then
            echo "Collecting coverage for $target..."
            LLVM_PROFILE_FILE="/tmp/fuzz_cov_combined/${target}_%m_%p.profraw" \
                fuzz/target/debug/$target fuzz/corpus/$target -runs=0 -rss_limit_mb=512 2>/dev/null || true
        fi
    done
    $LLVM_PROFDATA merge -sparse /tmp/fuzz_cov_combined/ -o /tmp/fuzz_combined.profdata
    echo ""
    echo "=== Combined fuzz coverage ==="
    $LLVM_COV report fuzz/target/debug/fuzz_decode \
        --instr-profile=/tmp/fuzz_combined.profdata \
        --ignore-filename-regex="\.cargo|rustc|registry" \
        | grep -E "^work/zen/heic/src|TOTAL"

# Run fuzz regression tests
fuzz-regression:
    cargo test --all-features --test fuzz_regression

# Profile decode-from-bytes heap allocations with heaptrack (needs heaptrack installed).
# Defaults to the bundled example.heic (1280x854) decoded 8x; pass a path + iters to override.
# Inspect with: heaptrack_print /tmp/heic-ht.zst
heaptrack-decode *ARGS:
    cargo build --release --example heaptrack_decode --features backend-rust,std
    rm -f /tmp/heic-ht.zst
    heaptrack --output /tmp/heic-ht ./target/release/examples/heaptrack_decode {{ARGS}}

# Exact residual arithmetic across native runtime tiers.
arm-residual-check:
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo test -p heic --features backend-rust,std,_dev --lib residual_add_matches_widened_scalar_at_every_prediction_value -- --nocapture

# Requires caller-provided HEIC_BENCH_INPUTS (colon-separated fixture paths).
arm-tier-audit:
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench --bench tier_isolation --features backend-rust,std,_dev

# Apple sample profiler; explicit fixture, 200 complete decodes, retained output.
arm-sample-decode fixture output:
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n19 cargo build --release --example heaptrack_decode --features backend-rust,std
    RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" python3 -c 'import subprocess; f=open("{{output}}.run.log","w"); p=subprocess.Popen(["nice","-n19","./target/release/examples/heaptrack_decode","{{fixture}}","200"],stdout=f,stderr=f); subprocess.run(["sample",str(p.pid),"5","1","-file","{{output}}.sample.txt"],check=True); raise SystemExit(p.wait())'
