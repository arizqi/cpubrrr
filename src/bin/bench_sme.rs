//! Phase-1 experiments: direct SME2 programming on M4.
//!
//! E1: smoke test — read SVL, run one fmopa.
//! E2: raw fmopa throughput (register-only): 1 tile (latency-bound) vs 4 tiles
//!     (throughput-bound), then across N threads to count SME units.
//! E3: real fp32 GEMM built on a 32x32 SME outer-product kernel vs Accelerate.

use std::arch::asm;
use std::time::Instant;

const SME: &str = ".arch armv9.2-a+sme2";

fn svl_bytes() -> u64 {
    let x: u64;
    unsafe {
        asm!(".arch armv9.2-a+sme2", "rdsvl {0}, #1", out(reg) x);
    }
    x
}

// ---------------------------------------------------------------------------
// E2: raw fmopa loops (no memory traffic)
// ---------------------------------------------------------------------------

/// 4 independent ZA tiles per iteration -> exposes instruction-level parallelism.
fn raw_fmopa_4tiles(iters: u64) {
    unsafe {
        asm!(
            ".arch armv9.2-a+sme2",
            "smstart",
            "ptrue p0.s",
            "ptrue p1.s",
            "fdup z0.s, #1.0",
            "fdup z1.s, #0.5",
            "zero {{za}}",
            "2:",
            "fmopa za0.s, p0/m, p1/m, z0.s, z1.s",
            "fmopa za1.s, p0/m, p1/m, z0.s, z1.s",
            "fmopa za2.s, p0/m, p1/m, z0.s, z1.s",
            "fmopa za3.s, p0/m, p1/m, z0.s, z1.s",
            "subs {i}, {i}, #1",
            "b.ne 2b",
            "smstop",
            i = inout(reg) iters => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
        );
    }
}

/// Same tile every time -> serial dependency, measures fmopa latency.
fn raw_fmopa_1tile(iters: u64) {
    unsafe {
        asm!(
            ".arch armv9.2-a+sme2",
            "smstart",
            "ptrue p0.s",
            "ptrue p1.s",
            "fdup z0.s, #1.0",
            "fdup z1.s, #0.5",
            "zero {{za}}",
            "2:",
            "fmopa za0.s, p0/m, p1/m, z0.s, z1.s",
            "fmopa za0.s, p0/m, p1/m, z0.s, z1.s",
            "fmopa za0.s, p0/m, p1/m, z0.s, z1.s",
            "fmopa za0.s, p0/m, p1/m, z0.s, z1.s",
            "subs {i}, {i}, #1",
            "b.ne 2b",
            "smstop",
            i = inout(reg) iters => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
        );
    }
}

fn bench_raw(name: &str, f: fn(u64), iters: u64, nthreads: usize, flops_per_iter: f64) -> f64 {
    // warmup
    f(iters / 10);
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..nthreads {
                s.spawn(move || f(iters));
            }
        });
        best = best.min(t.elapsed().as_secs_f64());
    }
    let gflops = nthreads as f64 * iters as f64 * flops_per_iter / best / 1e9;
    println!("{name:<30} threads={nthreads:<3} {gflops:>9.1} G/s");
    gflops
}

/// bf16 widening outer product: 2-deep dot per cell, fp32 accumulate.
fn raw_bfmopa_4tiles(iters: u64) {
    unsafe {
        asm!(
            ".arch armv9.2-a+sme2",
            "smstart",
            "ptrue p0.h",
            "ptrue p1.h",
            "fdup z0.h, #1.0",
            "fdup z1.h, #0.5",
            "zero {{za}}",
            "2:",
            "bfmopa za0.s, p0/m, p1/m, z0.h, z1.h",
            "bfmopa za1.s, p0/m, p1/m, z0.h, z1.h",
            "bfmopa za2.s, p0/m, p1/m, z0.h, z1.h",
            "bfmopa za3.s, p0/m, p1/m, z0.h, z1.h",
            "subs {i}, {i}, #1",
            "b.ne 2b",
            "smstop",
            i = inout(reg) iters => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
        );
    }
}

/// fp16 widening outer product: 2-deep dot per cell, fp32 accumulate.
fn raw_fmopa_f16_4tiles(iters: u64) {
    unsafe {
        asm!(
            ".arch armv9.2-a+sme2",
            "smstart",
            "ptrue p0.h",
            "ptrue p1.h",
            "fdup z0.h, #1.0",
            "fdup z1.h, #0.5",
            "zero {{za}}",
            "2:",
            "fmopa za0.s, p0/m, p1/m, z0.h, z1.h",
            "fmopa za1.s, p0/m, p1/m, z0.h, z1.h",
            "fmopa za2.s, p0/m, p1/m, z0.h, z1.h",
            "fmopa za3.s, p0/m, p1/m, z0.h, z1.h",
            "subs {i}, {i}, #1",
            "b.ne 2b",
            "smstop",
            i = inout(reg) iters => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
        );
    }
}

/// int8 outer product: 4-deep dot per cell, int32 accumulate.
fn raw_smopa_i8_4tiles(iters: u64) {
    unsafe {
        asm!(
            ".arch armv9.2-a+sme2",
            "smstart",
            "ptrue p0.b",
            "ptrue p1.b",
            "dup z0.b, #3",
            "dup z1.b, #5",
            "zero {{za}}",
            "2:",
            "smopa za0.s, p0/m, p1/m, z0.b, z1.b",
            "smopa za1.s, p0/m, p1/m, z0.b, z1.b",
            "smopa za2.s, p0/m, p1/m, z0.b, z1.b",
            "smopa za3.s, p0/m, p1/m, z0.b, z1.b",
            "subs {i}, {i}, #1",
            "b.ne 2b",
            "smstop",
            i = inout(reg) iters => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
        );
    }
}

// ---------------------------------------------------------------------------
// E3: fp32 GEMM on a 32x32 SME kernel
// ---------------------------------------------------------------------------
// C tile layout across the 4 fp32 ZA tiles:
//   za0 = rows 0..16  x cols 0..16     za1 = rows 0..16  x cols 16..32
//   za2 = rows 16..32 x cols 0..16     za3 = rows 16..32 x cols 16..32
// A packed: per k, 32 row-values contiguous. B packed: per k, 32 col-values.

/// c[32 x ldc] += packed_a(kb x 32) * packed_b(kb x 32)
unsafe fn sme_kernel_32x32(kb: u64, ap: *const f32, bp: *const f32, c: *mut f32, ldc_bytes: u64) {
    unsafe {
        asm!(
            ".arch armv9.2-a+sme2",
            "smstart",
            "ptrue p0.s",
            "zero {{za}}",
            // ---- k loop: 2 loads of A, 2 of B, 4 outer products ----
            "2:",
            "ld1w {{z0.s}}, p0/z, [{ap}]",
            "ld1w {{z1.s}}, p0/z, [{ap}, #1, mul vl]",
            "ld1w {{z2.s}}, p0/z, [{bp}]",
            "ld1w {{z3.s}}, p0/z, [{bp}, #1, mul vl]",
            "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
            "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
            "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
            "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
            "add {ap}, {ap}, #128",
            "add {bp}, {bp}, #128",
            "subs {kb}, {kb}, #1",
            "b.ne 2b",
            // ---- writeback: extract ZA slices, add into C ----
            "mov w12, #0",
            "mov {r0}, {c}",                 // row i base
            "add {r1}, {c}, {ldc}, lsl #4",  // row i+16 base
            "3:",
            "mov z4.s, p0/m, za0h.s[w12, 0]",
            "mov z5.s, p0/m, za1h.s[w12, 0]",
            "mov z6.s, p0/m, za2h.s[w12, 0]",
            "mov z7.s, p0/m, za3h.s[w12, 0]",
            "ld1w {{z16.s}}, p0/z, [{r0}]",
            "ld1w {{z17.s}}, p0/z, [{r0}, #1, mul vl]",
            "ld1w {{z18.s}}, p0/z, [{r1}]",
            "ld1w {{z19.s}}, p0/z, [{r1}, #1, mul vl]",
            "fadd z16.s, z16.s, z4.s",
            "fadd z17.s, z17.s, z5.s",
            "fadd z18.s, z18.s, z6.s",
            "fadd z19.s, z19.s, z7.s",
            "st1w {{z16.s}}, p0, [{r0}]",
            "st1w {{z17.s}}, p0, [{r0}, #1, mul vl]",
            "st1w {{z18.s}}, p0, [{r1}]",
            "st1w {{z19.s}}, p0, [{r1}, #1, mul vl]",
            "add {r0}, {r0}, {ldc}",
            "add {r1}, {r1}, {ldc}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 3b",
            "smstop",
            ap = inout(reg) ap => _,
            bp = inout(reg) bp => _,
            kb = inout(reg) kb => _,
            c = in(reg) c,
            ldc = in(reg) ldc_bytes,
            r0 = out(reg) _,
            r1 = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("x12") _,
            options(nostack),
        );
    }
}

/// v2: k unrolled x4 + SME2 multi-vector loads. 23 instructions per 16 fmopa
/// (vs 44 in v1). kb4 = k/4.
unsafe fn sme_kernel_32x32_v2(kb4: u64, ap: *const f32, bp: *const f32, c: *mut f32, ldc_bytes: u64) {
    unsafe {
        asm!(
            ".arch armv9.2-a+sme2",
            "smstart",
            "ptrue p0.s",
            "ptrue pn8.s",
            "zero {{za}}",
            "2:",
            // 4 k-steps of A (8 vectors) and B (8 vectors) in 4 instructions
            "ld1w {{z0.s - z3.s}}, pn8/z, [{ap}]",
            "ld1w {{z4.s - z7.s}}, pn8/z, [{ap}, #4, mul vl]",
            "ld1w {{z16.s - z19.s}}, pn8/z, [{bp}]",
            "ld1w {{z20.s - z23.s}}, pn8/z, [{bp}, #4, mul vl]",
            // k+0: A lo/hi = z0/z1, B lo/hi = z16/z17
            "fmopa za0.s, p0/m, p0/m, z0.s, z16.s",
            "fmopa za1.s, p0/m, p0/m, z0.s, z17.s",
            "fmopa za2.s, p0/m, p0/m, z1.s, z16.s",
            "fmopa za3.s, p0/m, p0/m, z1.s, z17.s",
            // k+1
            "fmopa za0.s, p0/m, p0/m, z2.s, z18.s",
            "fmopa za1.s, p0/m, p0/m, z2.s, z19.s",
            "fmopa za2.s, p0/m, p0/m, z3.s, z18.s",
            "fmopa za3.s, p0/m, p0/m, z3.s, z19.s",
            // k+2
            "fmopa za0.s, p0/m, p0/m, z4.s, z20.s",
            "fmopa za1.s, p0/m, p0/m, z4.s, z21.s",
            "fmopa za2.s, p0/m, p0/m, z5.s, z20.s",
            "fmopa za3.s, p0/m, p0/m, z5.s, z21.s",
            // k+3
            "fmopa za0.s, p0/m, p0/m, z6.s, z22.s",
            "fmopa za1.s, p0/m, p0/m, z6.s, z23.s",
            "fmopa za2.s, p0/m, p0/m, z7.s, z22.s",
            "fmopa za3.s, p0/m, p0/m, z7.s, z23.s",
            "add {ap}, {ap}, #512",
            "add {bp}, {bp}, #512",
            "subs {kb4}, {kb4}, #1",
            "b.ne 2b",
            // writeback identical to v1
            "mov w12, #0",
            "mov {r0}, {c}",
            "add {r1}, {c}, {ldc}, lsl #4",
            "3:",
            "mov z4.s, p0/m, za0h.s[w12, 0]",
            "mov z5.s, p0/m, za1h.s[w12, 0]",
            "mov z6.s, p0/m, za2h.s[w12, 0]",
            "mov z7.s, p0/m, za3h.s[w12, 0]",
            "ld1w {{z16.s}}, p0/z, [{r0}]",
            "ld1w {{z17.s}}, p0/z, [{r0}, #1, mul vl]",
            "ld1w {{z18.s}}, p0/z, [{r1}]",
            "ld1w {{z19.s}}, p0/z, [{r1}, #1, mul vl]",
            "fadd z16.s, z16.s, z4.s",
            "fadd z17.s, z17.s, z5.s",
            "fadd z18.s, z18.s, z6.s",
            "fadd z19.s, z19.s, z7.s",
            "st1w {{z16.s}}, p0, [{r0}]",
            "st1w {{z17.s}}, p0, [{r0}, #1, mul vl]",
            "st1w {{z18.s}}, p0, [{r1}]",
            "st1w {{z19.s}}, p0, [{r1}, #1, mul vl]",
            "add {r0}, {r0}, {ldc}",
            "add {r1}, {r1}, {ldc}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 3b",
            "smstop",
            ap = inout(reg) ap => _,
            bp = inout(reg) bp => _,
            kb4 = inout(reg) kb4 => _,
            c = in(reg) c,
            ldc = in(reg) ldc_bytes,
            r0 = out(reg) _,
            r1 = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("x12") _,
            options(nostack),
        );
    }
}

const TP: usize = 32; // tile panel width (rows of A / cols of B)

fn pack_a32(a: &[f32], lda: usize, m: usize, k: usize, out: &mut [f32]) {
    // panels of 32 rows; within panel, k-major: [k][32 rows]
    for (pi, i0) in (0..m).step_by(TP).enumerate() {
        let panel = &mut out[pi * k * TP..(pi + 1) * k * TP];
        for p in 0..k {
            for r in 0..TP {
                panel[p * TP + r] = a[(i0 + r) * lda + p];
            }
        }
    }
}

fn pack_b32(b: &[f32], ldb: usize, k: usize, n: usize, out: &mut [f32]) {
    // panels of 32 cols; within panel, k-major: [k][32 cols]
    for (pj, j0) in (0..n).step_by(TP).enumerate() {
        let panel = &mut out[pj * k * TP..(pj + 1) * k * TP];
        for p in 0..k {
            panel[p * TP..p * TP + TP].copy_from_slice(&b[p * ldb + j0..p * ldb + j0 + TP]);
        }
    }
}

/// Full GEMM: sizes must be multiples of 32. Threads split A row-panels.
fn sme_gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, nthreads: usize) {
    c.fill(0.0);
    let mut apack = vec![0.0f32; m * k];
    let mut bpack = vec![0.0f32; k * n];
    pack_a32(a, k, m, k, &mut apack);
    pack_b32(b, n, k, n, &mut bpack);
    let apack = &apack;
    let bpack = &bpack;
    let cptr = SendPtr(c.as_mut_ptr());
    let npanels_m = m / TP;
    let chunk = (npanels_m + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let p0 = t * chunk;
            if p0 >= npanels_m {
                break;
            }
            let p1 = (p0 + chunk).min(npanels_m);
            s.spawn(move || {
                let c = cptr.get();
                for pi in p0..p1 {
                    for pj in 0..n / TP {
                        unsafe {
                            sme_kernel_32x32(
                                k as u64,
                                apack[pi * k * TP..].as_ptr(),
                                bpack[pj * k * TP..].as_ptr(),
                                c.add(pi * TP * n + pj * TP),
                                (n * 4) as u64,
                            );
                        }
                    }
                }
            });
        }
    });
}

/// v2 GEMM: unrolled kernel + parallel packing. k must be a multiple of 4.
fn sme_gemm_v2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, nthreads: usize) {
    assert!(k % 4 == 0 && m % TP == 0 && n % TP == 0);
    c.fill(0.0);
    let mut apack = vec![0.0f32; m * k];
    let mut bpack = vec![0.0f32; k * n];
    // Packing is pure data movement -> use all cores (GEMM math uses only 2).
    {
        let ap = SendPtr(apack.as_mut_ptr());
        let bp = SendPtr(bpack.as_mut_ptr());
        par_panels(m / TP, 12, |pi| {
            let out = unsafe { std::slice::from_raw_parts_mut(ap.get().add(pi * k * TP), k * TP) };
            for p in 0..k {
                for r in 0..TP {
                    out[p * TP + r] = a[(pi * TP + r) * k + p];
                }
            }
        });
        par_panels(n / TP, 12, |pj| {
            let out = unsafe { std::slice::from_raw_parts_mut(bp.get().add(pj * k * TP), k * TP) };
            for p in 0..k {
                out[p * TP..p * TP + TP].copy_from_slice(&b[p * n + pj * TP..p * n + pj * TP + TP]);
            }
        });
    }
    let apack = &apack;
    let bpack = &bpack;
    let cptr = SendPtr(c.as_mut_ptr());
    let npanels_m = m / TP;
    let chunk = (npanels_m + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let p0 = t * chunk;
            if p0 >= npanels_m {
                break;
            }
            let p1 = (p0 + chunk).min(npanels_m);
            s.spawn(move || {
                let c = cptr.get();
                for pi in p0..p1 {
                    for pj in 0..n / TP {
                        unsafe {
                            sme_kernel_32x32_v2(
                                (k / 4) as u64,
                                apack[pi * k * TP..].as_ptr(),
                                bpack[pj * k * TP..].as_ptr(),
                                c.add(pi * TP * n + pj * TP),
                                (n * 4) as u64,
                            );
                        }
                    }
                }
            });
        }
    });
}

const KC3: usize = 512; // k block: A block 32*512*4 = 64 KB, fits in 128 KB L1d

/// v3: L1-blocked k loop + one thread wave (pack -> barrier -> compute).
fn sme_gemm_v3(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, nthreads: usize) {
    assert!(k % 4 == 0 && m % TP == 0 && n % TP == 0);
    let kc = KC3.min(k);
    assert!(k % kc == 0);
    c.fill(0.0);
    let mut apack = vec![0.0f32; m * k];
    let mut bpack = vec![0.0f32; k * n];
    let ap = SendPtr(apack.as_mut_ptr());
    let bp = SendPtr(bpack.as_mut_ptr());
    let cp = SendPtr(c.as_mut_ptr());
    let npanels_m = m / TP;
    let npanels_n = n / TP;
    let nthreads = nthreads.min(npanels_m);
    let chunk_m = (npanels_m + nthreads - 1) / nthreads;
    let chunk_n = (npanels_n + nthreads - 1) / nthreads;
    let barrier = std::sync::Barrier::new(nthreads);
    let barrier = &barrier;
    std::thread::scope(|s| {
        for t in 0..nthreads {
            s.spawn(move || {
                // pack phase: this thread's share of A and B panels
                for pi in t * chunk_m..((t + 1) * chunk_m).min(npanels_m) {
                    let out = unsafe { std::slice::from_raw_parts_mut(ap.get().add(pi * k * TP), k * TP) };
                    for p in 0..k {
                        for r in 0..TP {
                            out[p * TP + r] = a[(pi * TP + r) * k + p];
                        }
                    }
                }
                for pj in t * chunk_n..((t + 1) * chunk_n).min(npanels_n) {
                    let out = unsafe { std::slice::from_raw_parts_mut(bp.get().add(pj * k * TP), k * TP) };
                    for p in 0..k {
                        out[p * TP..p * TP + TP].copy_from_slice(&b[p * n + pj * TP..p * n + pj * TP + TP]);
                    }
                }
                barrier.wait();
                // compute phase: pi outer, kc block middle (A block hot in L1), pj inner
                let apack = unsafe { std::slice::from_raw_parts(ap.get(), m * k) };
                let bpack = unsafe { std::slice::from_raw_parts(bp.get(), k * n) };
                let c = cp.get();
                for pi in t * chunk_m..((t + 1) * chunk_m).min(npanels_m) {
                    for pc in (0..k).step_by(kc) {
                        for pj in 0..npanels_n {
                            unsafe {
                                sme_kernel_32x32_v2(
                                    (kc / 4) as u64,
                                    apack[pi * k * TP + pc * TP..].as_ptr(),
                                    bpack[pj * k * TP + pc * TP..].as_ptr(),
                                    c.add(pi * TP * n + pj * TP),
                                    (n * 4) as u64,
                                );
                            }
                        }
                    }
                }
            });
        }
    });
}

fn par_panels<F: Fn(usize) + Sync>(npanels: usize, nthreads: usize, f: F) {
    let chunk = (npanels + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let p0 = t * chunk;
            if p0 >= npanels {
                break;
            }
            let p1 = (p0 + chunk).min(npanels);
            let f = &f;
            s.spawn(move || {
                for p in p0..p1 {
                    f(p)
                }
            });
        }
    });
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Reference + Accelerate for comparison
// ---------------------------------------------------------------------------
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32, alpha: f32,
        a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32,
        c: *mut f32, ldc: i32,
    );
}

fn accelerate(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        cblas_sgemm(101, 111, 111, m as i32, n as i32, k as i32, 1.0,
            a.as_ptr(), k as i32, b.as_ptr(), n as i32, 0.0, c.as_mut_ptr(), n as i32);
    }
}

fn gen(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn main() {
    let _ = SME;
    // ---- E1: smoke ----
    println!("== E1 smoke ==");
    println!("SVL = {} bytes = {} fp32 lanes/vector, ZA tile = 16x16 fp32", svl_bytes(), svl_bytes() / 4);
    raw_fmopa_4tiles(1000);
    println!("fmopa executes: ok (no SIGILL)\n");

    // ---- E2: raw throughput ----
    println!("== E2 raw fmopa throughput (register-only) ==");
    let iters = 100_000_000u64;
    bench_raw("1 tile (latency chain)", raw_fmopa_1tile, iters / 4, 1, 2048.0);
    bench_raw("4 tiles (ILP)", raw_fmopa_4tiles, iters, 1, 2048.0);
    bench_raw("4 tiles (ILP)", raw_fmopa_4tiles, iters, 2, 2048.0);
    println!();

    println!("== E4 precision sweep, raw mopa throughput ==");
    bench_raw("fp32 fmopa (baseline)", raw_fmopa_4tiles, iters, 1, 2048.0);
    bench_raw("fp16 fmopa (2-deep dot)", raw_fmopa_f16_4tiles, iters, 1, 4096.0);
    bench_raw("bf16 bfmopa (2-deep dot)", raw_bfmopa_4tiles, iters, 1, 4096.0);
    bench_raw("int8 smopa (4-deep dot)", raw_smopa_i8_4tiles, iters, 1, 8192.0);
    bench_raw("fp16 fmopa (2-deep dot)", raw_fmopa_f16_4tiles, iters, 2, 4096.0);
    bench_raw("bf16 bfmopa (2-deep dot)", raw_bfmopa_4tiles, iters, 2, 4096.0);
    bench_raw("int8 smopa (4-deep dot)", raw_smopa_i8_4tiles, iters, 2, 8192.0);
    println!();

    // ---- E3 correctness ----
    println!("== E3 SME GEMM vs Accelerate ==");
    {
        let m = 128;
        let a = gen(m * m, 42);
        let b = gen(m * m, 1337);
        let mut cref = vec![0.0f32; m * m];
        accelerate(&a, &b, &mut cref, m, m, m);
        let mut c = vec![0.0f32; m * m];
        sme_gemm(&a, &b, &mut c, m, m, m, 1);
        let d = cref.iter().zip(&c).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(d < 1e-3, "SME GEMM v1 wrong: max abs diff {d}");
        println!("correctness v1 vs Accelerate: max abs diff {d:.2e} ok");
        let mut c2 = vec![0.0f32; m * m];
        sme_gemm_v2(&a, &b, &mut c2, m, m, m, 2);
        let d2 = cref.iter().zip(&c2).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(d2 < 1e-3, "SME GEMM v2 wrong: max abs diff {d2}");
        println!("correctness v2 vs Accelerate: max abs diff {d2:.2e} ok");
        let mut c3 = vec![0.0f32; m * m];
        sme_gemm_v3(&a, &b, &mut c3, m, m, m, 2);
        let d3 = cref.iter().zip(&c3).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(d3 < 1e-3, "SME GEMM v3 wrong: max abs diff {d3}");
        println!("correctness v3 vs Accelerate: max abs diff {d3:.2e} ok");
    }

    // ---- E3 perf ----
    for &size in &[512usize, 1024, 2048, 4096] {
        let (m, n, k) = (size, size, size);
        let flops = 2.0 * (m * n * k) as f64;
        let a = gen(m * k, 42);
        let b = gen(k * n, 1337);
        let mut c = vec![0.0f32; m * n];
        let mut run = |name: &str, f: &mut dyn FnMut()| {
            f();
            let mut best = f64::MAX;
            let mut total = 0.0;
            for _ in 0..10 {
                let t = Instant::now();
                f();
                let dt = t.elapsed().as_secs_f64();
                best = best.min(dt);
                total += dt;
                if total > 4.0 {
                    break;
                }
            }
            let g = flops / best / 1e9;
            println!("{name:<26} {size:>5} {g:>9.1} GFLOPS");
            g
        };
        run("sme gemm v2 1T", &mut || sme_gemm_v2(&a, &b, &mut c, m, n, k, 1));
        run("sme gemm v3 1T", &mut || sme_gemm_v3(&a, &b, &mut c, m, n, k, 1));
        run("sme gemm v3 2T", &mut || sme_gemm_v3(&a, &b, &mut c, m, n, k, 2));
        run("sme gemm v3 4T", &mut || sme_gemm_v3(&a, &b, &mut c, m, n, k, 4));
        run("accelerate", &mut || accelerate(&a, &b, &mut c, m, n, k));
        println!();
    }
}
