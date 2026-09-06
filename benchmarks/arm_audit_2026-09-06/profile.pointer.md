# HEIC sampling profile

Production d83e1d32, Apple M4 Pro, Rust 1.98, `just arm-sample-decode <example.heic> <output>`. 200 complete RGBA8 decodes; five seconds sampled at one-millisecond intervals, including startup.

- `/Users/lilith/work/codec-artifacts/heic-arm-audit/heic-profile.sample.txt` SHA256 `ce4ef0b6acf50d5a1205a23308935fa433c82bada4ebb8f3d603b90fbff5d5e1`

- `/Users/lilith/work/codec-artifacts/heic-arm-audit/heic-profile.run.log` SHA256 `67ef6b418c7d94a30aede5ab6dda7075c1269fe30499a98c2e425af71f9b6743`

- `/Users/lilith/work/codec-artifacts/heic-arm-audit/heic-profile-command.log` SHA256 `561e2ad6f42250b64b5ecc50aff39c5eba3a9b709d97e2c5dc62b1d55105984e`


Sampled leaf counts include `decode_and_apply_residual` 1283, CABAC `decode_bin` 889, memmove 273, intra prediction 257 and RGBA conversion 245. Inlining prevents attributing the entire parent-function count to one algorithm. Disassembly at parent offsets +2452 and +2896 follows significance/CABAC calls; +5972, +5988 and +6004 call memcpy with length 0x804 (2052 bytes), matching the returned coefficient-buffer extent. An internal output-parameter experiment is being checked against baseline pixels.
