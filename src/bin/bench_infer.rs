//! Inference-physics experiments.
//!
//! E5: raw CPU-side memory read bandwidth vs thread count (decode ceiling).
//! E6: int4-weight GEMV (matrix-vector) kernel — the decode inner loop — measured
//!     as GB/s streamed and projected tokens/sec.

use std::arch::aarch64::*;
use std::time::Instant;

// ---------------------------------------------------------------------------
// E5: read bandwidth
// ---------------------------------------------------------------------------

/// NEON sum of a large f32 buffer: pure streaming reads.
unsafe fn sum_f32(p: *const f32, n: usize) -> f32 {
    unsafe {
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= n {
            a0 = vaddq_f32(a0, vld1q_f32(p.add(i)));
            a1 = vaddq_f32(a1, vld1q_f32(p.add(i + 4)));
            a2 = vaddq_f32(a2, vld1q_f32(p.add(i + 8)));
            a3 = vaddq_f32(a3, vld1q_f32(p.add(i + 12)));
            i += 16;
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)))
    }
}

fn read_bw(nthreads: usize) -> f64 {
    const MB: usize = 512;
    let n = MB * 1024 * 1024 / 4;
    let bufs: Vec<Vec<f32>> = (0..nthreads).map(|_| vec![1.0f32; n]).collect();
    let reps = 4;
    let mut best = f64::MAX;
    let mut sink = 0.0f32;
    for _ in 0..3 {
        let t = Instant::now();
        let sums: Vec<f32> = std::thread::scope(|s| {
            let hs: Vec<_> = bufs
                .iter()
                .map(|b| {
                    s.spawn(move || {
                        let mut acc = 0.0f32;
                        for _ in 0..reps {
                            acc += unsafe { sum_f32(b.as_ptr(), n) };
                        }
                        acc
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        best = best.min(t.elapsed().as_secs_f64());
        sink += sums.iter().sum::<f32>();
    }
    if sink.is_nan() {
        println!("(impossible)");
    }
    (nthreads * n * 4 * reps) as f64 / best / 1e9
}

// ---------------------------------------------------------------------------
// E6: int4 GEMV — decode inner loop
// ---------------------------------------------------------------------------
// Weight layout: per 32-element block, 16 bytes; byte i = (w[b*32+16+i] << 4) | w[b*32+i],
// values stored as unsigned nibble = signed weight + 8. Activation x is int8.

/// sdot: 4-way int8 dot product per lane. Intrinsic unstable in Rust 1.88 -> one-line asm.
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        let mut r = acc;
        std::arch::asm!(
            "sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
            r = inout(vreg) r,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        r
    }
}

unsafe fn dot_row_q4(w: *const u8, x: *const i8, d: usize) -> i32 {
    unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mask = vdupq_n_u8(0x0F);
        let bias = vdupq_n_s8(8);
        let mut b = 0;
        while b + 64 <= d {
            // two 32-element blocks per iteration
            let wb0 = vld1q_u8(w.add(b / 2));
            let wb1 = vld1q_u8(w.add(b / 2 + 16));
            let lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(wb0, mask)), bias);
            let hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(wb0)), bias);
            let lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(wb1, mask)), bias);
            let hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(wb1)), bias);
            acc0 = sdot(acc0, lo0, vld1q_s8(x.add(b)));
            acc0 = sdot(acc0, hi0, vld1q_s8(x.add(b + 16)));
            acc1 = sdot(acc1, lo1, vld1q_s8(x.add(b + 32)));
            acc1 = sdot(acc1, hi1, vld1q_s8(x.add(b + 48)));
            b += 64;
        }
        vaddvq_s32(vaddq_s32(acc0, acc1))
    }
}

/// v2: unsigned nibbles + bias fold (y = dot(w_u4, x) - 8*sum(x)), 4 rows per x load.
unsafe fn dot4_q4_v2(w: *const u8, stride: usize, x: *const i8, d: usize) -> [i32; 4] {
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let mut a0 = vdupq_n_s32(0);
        let mut a1 = vdupq_n_s32(0);
        let mut a2 = vdupq_n_s32(0);
        let mut a3 = vdupq_n_s32(0);
        let mut b = 0;
        while b + 32 <= d {
            let x0 = vld1q_s8(x.add(b));
            let x1 = vld1q_s8(x.add(b + 16));
            let off = b / 2;
            let w0 = vld1q_u8(w.add(off));
            let w1 = vld1q_u8(w.add(stride + off));
            let w2 = vld1q_u8(w.add(2 * stride + off));
            let w3 = vld1q_u8(w.add(3 * stride + off));
            a0 = sdot(a0, vreinterpretq_s8_u8(vandq_u8(w0, mask)), x0);
            a0 = sdot(a0, vreinterpretq_s8_u8(vshrq_n_u8::<4>(w0)), x1);
            a1 = sdot(a1, vreinterpretq_s8_u8(vandq_u8(w1, mask)), x0);
            a1 = sdot(a1, vreinterpretq_s8_u8(vshrq_n_u8::<4>(w1)), x1);
            a2 = sdot(a2, vreinterpretq_s8_u8(vandq_u8(w2, mask)), x0);
            a2 = sdot(a2, vreinterpretq_s8_u8(vshrq_n_u8::<4>(w2)), x1);
            a3 = sdot(a3, vreinterpretq_s8_u8(vandq_u8(w3, mask)), x0);
            a3 = sdot(a3, vreinterpretq_s8_u8(vshrq_n_u8::<4>(w3)), x1);
            b += 32;
        }
        [vaddvq_s32(a0), vaddvq_s32(a1), vaddvq_s32(a2), vaddvq_s32(a3)]
    }
}

/// v3: v2 + two accumulators per row (break sdot dependency chains) + prefetch.
unsafe fn dot4_q4_v3(w: *const u8, stride: usize, x: *const i8, d: usize) -> [i32; 4] {
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let mut acc = [vdupq_n_s32(0); 8]; // [row0_a, row0_b, row1_a, ...]
        let mut b = 0;
        while b + 64 <= d {
            let x0 = vld1q_s8(x.add(b));
            let x1 = vld1q_s8(x.add(b + 16));
            let x2 = vld1q_s8(x.add(b + 32));
            let x3 = vld1q_s8(x.add(b + 48));
            let off = b / 2;
            let mut r = 0;
            while r < 4 {
                let base = w.add(r * stride + off);
                std::arch::asm!("prfm pldl1strm, [{p}, #512]", p = in(reg) base, options(nostack, readonly));
                let wa = vld1q_u8(base);
                let wb = vld1q_u8(base.add(16));
                acc[2 * r] = sdot(acc[2 * r], vreinterpretq_s8_u8(vandq_u8(wa, mask)), x0);
                acc[2 * r + 1] = sdot(acc[2 * r + 1], vreinterpretq_s8_u8(vshrq_n_u8::<4>(wa)), x1);
                acc[2 * r] = sdot(acc[2 * r], vreinterpretq_s8_u8(vandq_u8(wb, mask)), x2);
                acc[2 * r + 1] = sdot(acc[2 * r + 1], vreinterpretq_s8_u8(vshrq_n_u8::<4>(wb)), x3);
                r += 1;
            }
            b += 64;
        }
        [
            vaddvq_s32(vaddq_s32(acc[0], acc[1])),
            vaddvq_s32(vaddq_s32(acc[2], acc[3])),
            vaddvq_s32(vaddq_s32(acc[4], acc[5])),
            vaddvq_s32(vaddq_s32(acc[6], acc[7])),
        ]
    }
}

fn gemv_q4_v3(w: &[u8], x: &[i8], y: &mut [i32], rows: usize, d: usize, nthreads: usize) {
    assert!(rows % 4 == 0 && d % 64 == 0);
    let sumx: i32 = x.iter().map(|&v| v as i32).sum();
    let yp = SendPtr(y.as_mut_ptr());
    let nquads = rows / 4;
    let chunk = (nquads + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let q0 = t * chunk;
            if q0 >= nquads {
                break;
            }
            let q1 = (q0 + chunk).min(nquads);
            s.spawn(move || {
                let y = yp.get();
                for q in q0..q1 {
                    let r = q * 4;
                    let acc = unsafe { dot4_q4_v3(w.as_ptr().add(r * d / 2), d / 2, x.as_ptr(), d) };
                    for i in 0..4 {
                        unsafe { *y.add(r + i) = acc[i] - 8 * sumx };
                    }
                }
            });
        }
    });
}

/// v4: quad-interleaved layout -> one sequential stream per thread.
/// Layout: [quad][block][row 0..4][16 bytes]; kernel streams 64 B contiguously.
fn pack_q4_quad(weights: &[i8], rows: usize, d: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows * d / 2];
    let qstride = 4 * d / 2;
    for q in 0..rows / 4 {
        for b in 0..d / 32 {
            for r in 0..4 {
                for i in 0..16 {
                    let lo = (weights[(q * 4 + r) * d + b * 32 + i] + 8) as u8;
                    let hi = (weights[(q * 4 + r) * d + b * 32 + 16 + i] + 8) as u8;
                    out[q * qstride + b * 64 + r * 16 + i] = (hi << 4) | (lo & 0x0F);
                }
            }
        }
    }
    out
}

unsafe fn dot4_q4_v4(w: *const u8, x: *const i8, d: usize) -> [i32; 4] {
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let mut acc = [vdupq_n_s32(0); 8];
        let mut wp = w;
        let mut b = 0;
        while b + 32 <= d {
            let x0 = vld1q_s8(x.add(b));
            let x1 = vld1q_s8(x.add(b + 16));
            let w0 = vld1q_u8(wp);
            let w1 = vld1q_u8(wp.add(16));
            let w2 = vld1q_u8(wp.add(32));
            let w3 = vld1q_u8(wp.add(48));
            acc[0] = sdot(acc[0], vreinterpretq_s8_u8(vandq_u8(w0, mask)), x0);
            acc[1] = sdot(acc[1], vreinterpretq_s8_u8(vshrq_n_u8::<4>(w0)), x1);
            acc[2] = sdot(acc[2], vreinterpretq_s8_u8(vandq_u8(w1, mask)), x0);
            acc[3] = sdot(acc[3], vreinterpretq_s8_u8(vshrq_n_u8::<4>(w1)), x1);
            acc[4] = sdot(acc[4], vreinterpretq_s8_u8(vandq_u8(w2, mask)), x0);
            acc[5] = sdot(acc[5], vreinterpretq_s8_u8(vshrq_n_u8::<4>(w2)), x1);
            acc[6] = sdot(acc[6], vreinterpretq_s8_u8(vandq_u8(w3, mask)), x0);
            acc[7] = sdot(acc[7], vreinterpretq_s8_u8(vshrq_n_u8::<4>(w3)), x1);
            wp = wp.add(64);
            b += 32;
        }
        [
            vaddvq_s32(vaddq_s32(acc[0], acc[1])),
            vaddvq_s32(vaddq_s32(acc[2], acc[3])),
            vaddvq_s32(vaddq_s32(acc[4], acc[5])),
            vaddvq_s32(vaddq_s32(acc[6], acc[7])),
        ]
    }
}

fn gemv_q4_v4(w: &[u8], x: &[i8], y: &mut [i32], rows: usize, d: usize, nthreads: usize) {
    assert!(rows % 4 == 0 && d % 32 == 0);
    let sumx: i32 = x.iter().map(|&v| v as i32).sum();
    let yp = SendPtr(y.as_mut_ptr());
    let nquads = rows / 4;
    let qstride = 4 * d / 2;
    let chunk = (nquads + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let q0 = t * chunk;
            if q0 >= nquads {
                break;
            }
            let q1 = (q0 + chunk).min(nquads);
            s.spawn(move || {
                let y = yp.get();
                for q in q0..q1 {
                    let acc = unsafe { dot4_q4_v4(w.as_ptr().add(q * qstride), x.as_ptr(), d) };
                    for i in 0..4 {
                        unsafe { *y.add(q * 4 + i) = acc[i] - 8 * sumx };
                    }
                }
            });
        }
    });
}

/// Ask the scheduler to treat this thread as latency-critical -> P-cores.
unsafe extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}
const QOS_USER_INTERACTIVE: u32 = 0x21;

/// v5: identical kernel to v4, workers pinned to P-cores via QoS.
fn gemv_q4_v5(w: &[u8], x: &[i8], y: &mut [i32], rows: usize, d: usize, nthreads: usize) {
    assert!(rows % 4 == 0 && d % 32 == 0);
    let sumx: i32 = x.iter().map(|&v| v as i32).sum();
    let yp = SendPtr(y.as_mut_ptr());
    let nquads = rows / 4;
    let qstride = 4 * d / 2;
    let chunk = (nquads + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let q0 = t * chunk;
            if q0 >= nquads {
                break;
            }
            let q1 = (q0 + chunk).min(nquads);
            s.spawn(move || {
                unsafe { pthread_set_qos_class_self_np(QOS_USER_INTERACTIVE, 0) };
                let y = yp.get();
                for q in q0..q1 {
                    let acc = unsafe { dot4_q4_v4(w.as_ptr().add(q * qstride), x.as_ptr(), d) };
                    for i in 0..4 {
                        unsafe { *y.add(q * 4 + i) = acc[i] - 8 * sumx };
                    }
                }
            });
        }
    });
}

fn gemv_q4_v2(w: &[u8], x: &[i8], y: &mut [i32], rows: usize, d: usize, nthreads: usize) {
    assert!(rows % 4 == 0);
    let sumx: i32 = x.iter().map(|&v| v as i32).sum();
    let yp = SendPtr(y.as_mut_ptr());
    let nquads = rows / 4;
    let chunk = (nquads + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let q0 = t * chunk;
            if q0 >= nquads {
                break;
            }
            let q1 = (q0 + chunk).min(nquads);
            s.spawn(move || {
                let y = yp.get();
                for q in q0..q1 {
                    let r = q * 4;
                    let acc = unsafe { dot4_q4_v2(w.as_ptr().add(r * d / 2), d / 2, x.as_ptr(), d) };
                    for i in 0..4 {
                        unsafe { *y.add(r + i) = acc[i] - 8 * sumx };
                    }
                }
            });
        }
    });
}

fn gemv_q4(w: &[u8], x: &[i8], y: &mut [i32], rows: usize, d: usize, nthreads: usize) {
    let yp = SendPtr(y.as_mut_ptr());
    let chunk = (rows + nthreads - 1) / nthreads;
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let r0 = t * chunk;
            if r0 >= rows {
                break;
            }
            let r1 = (r0 + chunk).min(rows);
            s.spawn(move || {
                let y = yp.get();
                for r in r0..r1 {
                    unsafe {
                        *y.add(r) = dot_row_q4(w.as_ptr().add(r * d / 2), x.as_ptr(), d);
                    }
                }
            });
        }
    });
}

#[derive(Clone, Copy)]
struct SendPtr(*mut i32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut i32 {
        self.0
    }
}

fn pack_q4(weights: &[i8], rows: usize, d: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows * d / 2];
    for r in 0..rows {
        for b in 0..d / 32 {
            for i in 0..16 {
                let lo = (weights[r * d + b * 32 + i] + 8) as u8;
                let hi = (weights[r * d + b * 32 + 16 + i] + 8) as u8;
                out[r * d / 2 + b * 16 + i] = (hi << 4) | (lo & 0x0F);
            }
        }
    }
    out
}

fn main() {
    println!("== E5 CPU read bandwidth (streaming f32 sum, 512 MB/thread) ==");
    let mut peak_bw = 0.0f64;
    for t in [1usize, 2, 4, 6, 8, 12, 16] {
        let bw = read_bw(t);
        peak_bw = peak_bw.max(bw);
        println!("threads={t:<3} {bw:>7.1} GB/s");
    }
    println!("peak CPU-side bandwidth: {peak_bw:.0} GB/s (chip spec 546 GB/s)\n");

    println!("== E6 int4 GEMV (decode inner loop) ==");
    // correctness first
    {
        let (rows, d) = (8, 128);
        let wq: Vec<i8> = (0..rows * d).map(|i| ((i * 7 + 3) % 15) as i8 - 7).collect();
        let x: Vec<i8> = (0..d).map(|i| ((i * 5 + 1) % 21) as i8 - 10).collect();
        let packed = pack_q4(&wq, rows, d);
        let mut y = vec![0i32; rows];
        gemv_q4(&packed, &x, &mut y, rows, d, 2);
        for r in 0..rows {
            let expect: i32 = (0..d).map(|j| wq[r * d + j] as i32 * x[j] as i32).sum();
            assert_eq!(y[r], expect, "v1 row {r}");
        }
        let mut y2 = vec![0i32; rows];
        gemv_q4_v2(&packed, &x, &mut y2, rows, d, 2);
        for r in 0..rows {
            let expect: i32 = (0..d).map(|j| wq[r * d + j] as i32 * x[j] as i32).sum();
            assert_eq!(y2[r], expect, "v2 row {r}");
        }
        let mut y3 = vec![0i32; rows];
        gemv_q4_v3(&packed, &x, &mut y3, rows, d, 2);
        for r in 0..rows {
            let expect: i32 = (0..d).map(|j| wq[r * d + j] as i32 * x[j] as i32).sum();
            assert_eq!(y3[r], expect, "v3 row {r}");
        }
        let packed4 = pack_q4_quad(&wq, rows, d);
        let mut y4 = vec![0i32; rows];
        gemv_q4_v4(&packed4, &x, &mut y4, rows, d, 2);
        for r in 0..rows {
            let expect: i32 = (0..d).map(|j| wq[r * d + j] as i32 * x[j] as i32).sum();
            assert_eq!(y4[r], expect, "v4 row {r}");
        }
        println!("correctness v1+v2+v3+v4 vs scalar reference: EXACT");
    }
    // perf: one 8192x8192 layer = 32 MB of int4 weights, streamed from DRAM
    let (rows, d) = (8192usize, 8192usize);
    let w = vec![0x53u8; rows * d / 2];
    let x = vec![3i8; d];
    let mut y = vec![0i32; rows];
    let bytes_per_mv = (rows * d / 2) as f64;
    let mut bench_gemv = |name: &str, f: &mut dyn FnMut(usize)| {
        for t in [1usize, 2, 4, 6, 8, 10, 12] {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let ti = Instant::now();
                for _ in 0..8 {
                    f(t);
                }
                best = best.min(ti.elapsed().as_secs_f64() / 8.0);
            }
            let gbs = bytes_per_mv / best / 1e9;
            println!("{name} threads={t:<3} {gbs:>7.1} GB/s effective weight-streaming");
        }
    };
    bench_gemv("v1", &mut |t| gemv_q4(&w, &x, &mut y, rows, d, t));
    println!();
    bench_gemv("v2", &mut |t| gemv_q4_v2(&w, &x, &mut y, rows, d, t));
    println!();
    bench_gemv("v3", &mut |t| gemv_q4_v3(&w, &x, &mut y, rows, d, t));
    println!();
    bench_gemv("v4", &mut |t| gemv_q4_v4(&w, &x, &mut y, rows, d, t));
    println!();
    bench_gemv("v5", &mut |t| gemv_q4_v5(&w, &x, &mut y, rows, d, t));
    println!("\nprojection: tok/s = achieved GB/s / active-GB per token (see report)");
}
