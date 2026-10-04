# B-VECTOR-RESTORE — receipt

Row: `APEIRON-B-VECTOR` (`apeiron-b-vector-restore-HC9kdgd1vcpPWFnHT0qdDH`).

Merge `920f87c` dropped the `pub mod gpu;` declaration from `src/lib.rs`, leaving
`src/gpu/{mod,runtime,vulkan}.rs` out of the build path while the `gpu = []` feature still
existed. Restored, behind that existing feature:

```rust
/// The sovereign GPU runtime (dlopen Vulkan device + dispatch) over the ported `simt`
/// kernels. Behind the `gpu` feature; the kernel builders themselves stay always-on.
#[cfg(feature = "gpu")]
pub mod gpu;
```

Gate (live tree): `build-slot --mem 6 --timeout 540 -- systemd-run --user --scope
-p MemoryMax=6G -p MemorySwapMax=0 -- timeout 540 cargo build --locked --features gpu`
— **exit 0**, `Finished dev profile [unoptimized + debuginfo] target(s) in 6.43s`.

Control: without the two lines the `gpu` feature body (`src/gpu/*.rs`) is absent from the
compiled crate; the restore is what puts it back on the path. Grade: `measured` (build) —
no proof claim.
