//! Phase-0 ceiling benchmark: fp32 matmul ladder on Apple silicon.
//!
//! Ladder: naive -> loop-reordered (autovec) -> cache-blocked -> NEON microkernel
//! (1 thread) -> NEON microkernel (P-core threads) -> Accelerate cblas_sgemm (AMX/SME
//! ceiling). C = A(mxk) * B(kxn), row-major.

use std::time::Instant;

const MR: usize = 8; // microkernel rows
const NR: usize = 8; // microkernel cols (two float32x4)
const KC: usize = 512; // k blocking: A/B panels stay in L1/L2
const MC: usize = 128; // m blocking per packed A block
const NC: usize = 1024; // n blocking: B panel fits L2

// ---------------------------------------------------------------------------
// 1. Naive ijk
// ---------------------------------------------------------------------------
fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Loop-reordered ikp -> unit-stride inner loop, compiler autovectorizes
// ---------------------------------------------------------------------------
fn matmul_ikj(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            let brow = &b[p * n..p * n + n];
            let crow = &mut c[i * n..i * n + n];
            for j in 0..n {
                crow[j] += av * brow[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Cache-blocked ikj
// ---------------------------------------------------------------------------
fn matmul_blocked(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.fill(0.0);
    for jc in (0..n).step_by(NC) {
        let nb = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let kb = KC.min(k - pc);
            for i in 0..m {
                for p in pc..pc + kb {
                    let av = a[i * k + p];
                    let brow = &b[p * n + jc..p * n + jc + nb];
                    let crow = &mut c[i * n + jc..i * n + jc + nb];
                    for j in 0..nb {
                        crow[j] += av * brow[j];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. NEON microkernel with packing (BLIS-style)
// ---------------------------------------------------------------------------
// Packs: A block (MC x KC) into column-of-MR panels; B panel (KC x NC) into
// row-of-NR panels. Microkernel: 8x8 tile of C, 16 fp32x4 accumulators,
// vfmaq_laneq broadcast pattern.

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{KC, MC, MR, NC, NR};
    use std::arch::aarch64::*;

    pub fn pack_a(a: &[f32], lda: usize, mb: usize, kb: usize, out: &mut [f32]) {
        // out: ceil(mb/MR) panels, each panel kb x MR (k-major)
        for (pi, i0) in (0..mb).step_by(MR).enumerate() {
            let rows = MR.min(mb - i0);
            let panel = &mut out[pi * KC * MR..];
            for p in 0..kb {
                for r in 0..MR {
                    panel[p * MR + r] = if r < rows { a[(i0 + r) * lda + p] } else { 0.0 };
                }
            }
        }
    }

    pub fn pack_b(b: &[f32], ldb: usize, kb: usize, nb: usize, out: &mut [f32]) {
        // out: ceil(nb/NR) panels, each panel kb x NR (k-major)
        for (pj, j0) in (0..nb).step_by(NR).enumerate() {
            let cols = NR.min(nb - j0);
            let panel = &mut out[pj * KC * NR..];
            for p in 0..kb {
                for cidx in 0..NR {
                    panel[p * NR + cidx] = if cidx < cols { b[p * ldb + j0 + cidx] } else { 0.0 };
                }
            }
        }
    }

    /// 8x8 microkernel: c[8][8] += a_panel(kb x 8) * b_panel(kb x 8)
    #[inline(always)]
    unsafe fn kernel_8x8(kb: usize, ap: *const f32, bp: *const f32, c: *mut f32, ldc: usize) {
        let mut acc = [[vdupq_n_f32(0.0); 2]; MR];
        let mut ap = ap;
        let mut bp = bp;
        for _ in 0..kb {
            let b0 = vld1q_f32(bp);
            let b1 = vld1q_f32(bp.add(4));
            let a0 = vld1q_f32(ap); // rows 0..4
            let a1 = vld1q_f32(ap.add(4)); // rows 4..8
            acc[0][0] = vfmaq_laneq_f32::<0>(acc[0][0], b0, a0);
            acc[0][1] = vfmaq_laneq_f32::<0>(acc[0][1], b1, a0);
            acc[1][0] = vfmaq_laneq_f32::<1>(acc[1][0], b0, a0);
            acc[1][1] = vfmaq_laneq_f32::<1>(acc[1][1], b1, a0);
            acc[2][0] = vfmaq_laneq_f32::<2>(acc[2][0], b0, a0);
            acc[2][1] = vfmaq_laneq_f32::<2>(acc[2][1], b1, a0);
            acc[3][0] = vfmaq_laneq_f32::<3>(acc[3][0], b0, a0);
            acc[3][1] = vfmaq_laneq_f32::<3>(acc[3][1], b1, a0);
            acc[4][0] = vfmaq_laneq_f32::<0>(acc[4][0], b0, a1);
            acc[4][1] = vfmaq_laneq_f32::<0>(acc[4][1], b1, a1);
            acc[5][0] = vfmaq_laneq_f32::<1>(acc[5][0], b0, a1);
            acc[5][1] = vfmaq_laneq_f32::<1>(acc[5][1], b1, a1);
            acc[6][0] = vfmaq_laneq_f32::<2>(acc[6][0], b0, a1);
            acc[6][1] = vfmaq_laneq_f32::<2>(acc[6][1], b1, a1);
            acc[7][0] = vfmaq_laneq_f32::<3>(acc[7][0], b0, a1);
            acc[7][1] = vfmaq_laneq_f32::<3>(acc[7][1], b1, a1);
            ap = ap.add(MR);
            bp = bp.add(NR);
        }
        for r in 0..MR {
            let cr = c.add(r * ldc);
            vst1q_f32(cr, vaddq_f32(vld1q_f32(cr), acc[r][0]));
            vst1q_f32(cr.add(4), vaddq_f32(vld1q_f32(cr.add(4)), acc[r][1]));
        }
    }

    /// Single-thread packed GEMM over rows [row0, row1). Sizes must be multiples of 8.
    pub fn gemm_rows(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        row0: usize,
        row1: usize,
    ) {
        let mut apack = vec![0.0f32; MC * KC];
        let mut bpack = vec![0.0f32; KC * NC];
        for jc in (0..n).step_by(NC) {
            let nb = NC.min(n - jc);
            for pc in (0..k).step_by(KC) {
                let kb = KC.min(k - pc);
                pack_b(&b[pc * n + jc..], n, kb, nb, &mut bpack);
                for ic in (row0..row1).step_by(MC) {
                    let mb = MC.min(row1 - ic);
                    pack_a(&a[ic * k + pc..], k, mb, kb, &mut apack);
                    for jr in (0..nb).step_by(NR) {
                        for ir in (0..mb).step_by(MR) {
                            unsafe {
                                kernel_8x8(
                                    kb,
                                    apack[(ir / MR) * KC * MR..].as_ptr(),
                                    bpack[(jr / NR) * KC * NR..].as_ptr(),
                                    c.as_mut_ptr().add((ic + ir) * n + jc + jr),
                                    n,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        c.fill(0.0);
        gemm_rows(a, b, c, m, n, k, 0, m);
    }

    pub fn gemm_threaded(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        nthreads: usize,
    ) {
        c.fill(0.0);
        // Split M across threads in MR-aligned chunks; each thread owns disjoint C rows.
        let chunk = (m / nthreads + MR - 1) / MR * MR;
        let cptr = SendPtr(c.as_mut_ptr());
        std::thread::scope(|s| {
            for t in 0..nthreads {
                let row0 = t * chunk;
                if row0 >= m {
                    break;
                }
                let row1 = (row0 + chunk).min(m);
                s.spawn(move || {
                    let cs = unsafe { std::slice::from_raw_parts_mut(cptr.get(), m * n) };
                    gemm_rows(a, b, cs, m, n, k, row0, row1);
                });
            }
        });
    }

    #[derive(Clone, Copy)]
    struct SendPtr(*mut f32);
    unsafe impl Send for SendPtr {}
    impl SendPtr {
        fn get(self) -> *mut f32 {
            self.0
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Accelerate ceiling (AMX/SME via cblas_sgemm)
// ---------------------------------------------------------------------------
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

fn matmul_accelerate(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        cblas_sgemm(
            101, // RowMajor
            111, // NoTrans
            111,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
fn bench<F: FnMut()>(mut f: F, flops: f64, max_secs: f64) -> f64 {
    f(); // warmup + first timing sample
    let mut best = f64::MAX;
    let mut total = 0.0;
    for _ in 0..10 {
        let t = Instant::now();
        f();
        let dt = t.elapsed().as_secs_f64();
        best = best.min(dt);
        total += dt;
        if total > max_secs {
            break;
        }
    }
    flops / best / 1e9
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

fn main() {
    let nthreads: usize = std::env::var("CPUBRRR_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12); // P-cores on M4 Max

    // Correctness check at 256 (all kernels vs naive)
    {
        let m = 256;
        let (a, b) = gen(m, m, m);
        let mut cref = vec![0.0f32; m * m];
        matmul_naive(&a, &b, &mut cref, m, m, m);
        let mut c = vec![0.0f32; m * m];
        for (name, mut f) in checks(&a, &b, m, nthreads) {
            c.fill(f32::NAN);
            f(&mut c);
            let d = max_abs_diff(&cref, &c);
            assert!(d < 1e-2, "{name} wrong: max abs diff {d}");
            println!("correctness {name}: max abs diff {d:.2e} ok");
        }
    }
    println!();

    println!(
        "{:<28} {:>8} {:>10} {:>10}",
        "kernel", "size", "GFLOPS", "%ceiling"
    );
    for &size in &[512usize, 1024, 2048, 4096] {
        let (m, n, k) = (size, size, size);
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let (a, b) = gen(m, n, k);
        let mut c = vec![0.0f32; m * n];

        let ceiling = bench(|| matmul_accelerate(&a, &b, &mut c, m, n, k), flops, 5.0);

        let mut row = |name: &str, g: f64| {
            println!("{:<28} {:>8} {:>10.1} {:>9.1}%", name, size, g, 100.0 * g / ceiling);
        };

        if size <= 1024 {
            row("naive ijk", bench(|| matmul_naive(&a, &b, &mut c, m, n, k), flops, 10.0));
        }
        row("reordered ikj (autovec)", bench(|| matmul_ikj(&a, &b, &mut c, m, n, k), flops, 5.0));
        row("cache-blocked ikj", bench(|| matmul_blocked(&a, &b, &mut c, m, n, k), flops, 5.0));
        row("neon microkernel 1T", bench(|| neon::gemm(&a, &b, &mut c, m, n, k), flops, 5.0));
        row(
            &format!("neon microkernel {nthreads}T"),
            bench(|| neon::gemm_threaded(&a, &b, &mut c, m, n, k, nthreads), flops, 5.0),
        );
        row("accelerate sgemm (ceiling)", ceiling);
        println!();
    }
}

fn gen(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    // Deterministic pseudo-random fill, no deps
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| next()).collect();
    (a, b)
}

fn checks<'x>(
    a: &'x [f32],
    b: &'x [f32],
    m: usize,
    nthreads: usize,
) -> Vec<(&'static str, Box<dyn FnMut(&mut Vec<f32>) + 'x>)> {
    vec![
        ("ikj", Box::new(move |c: &mut Vec<f32>| matmul_ikj(a, b, c, m, m, m))),
        ("blocked", Box::new(move |c: &mut Vec<f32>| matmul_blocked(a, b, c, m, m, m))),
        ("neon", Box::new(move |c: &mut Vec<f32>| neon::gemm(a, b, c, m, m, m))),
        (
            "neon-threaded",
            Box::new(move |c: &mut Vec<f32>| neon::gemm_threaded(a, b, c, m, m, m, nthreads)),
        ),
        ("accelerate", Box::new(move |c: &mut Vec<f32>| matmul_accelerate(a, b, c, m, m, m))),
    ]
}
