//! E8: MoE expert-batched decode — bandwidth simulation of a gpt-oss:20b-shaped
//! FFN layer (32 experts, top-4, d=2880, 3 matrices per expert, int4).
//! Thesis: expert routing costs ~nothing if the active experts' GEMVs form one
//! flat work list per token (no per-op dispatch/sync like llama.cpp).

use std::arch::aarch64::*;
use std::time::Instant;

const D: usize = 2880;
const EXPERTS: usize = 32;
const TOPK: usize = 4;
const MATS: usize = 3; // gate, up, down
const LAYERS: usize = 24;

#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        let mut r = acc;
        std::arch::asm!(
            "sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
            r = inout(vreg) r, a = in(vreg) a, b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        r
    }
}

/// v4 kernel: quad-interleaved int4, one sequential stream, 4 rows per pass.
unsafe fn dot4_q4(w: *const u8, x: *const i8, d: usize) -> [i32; 4] {
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

#[derive(Clone, Copy)]
struct SendPtr(*mut i32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut i32 {
        self.0
    }
}

fn main() {
    let mat_bytes = D * D / 2; // 4.15 MB int4
    let qstride = 4 * D / 2;
    let quads_per_mat = D / 4;
    println!("== E8 MoE expert-batched decode (gpt-oss:20b shape) ==");
    println!(
        "{} experts x {} mats x {:.1} MB = {:.0} MB layer; top-{} active -> {:.1} MB/token/layer",
        EXPERTS, MATS, mat_bytes as f64 / 1e6,
        (EXPERTS * MATS * mat_bytes) as f64 / 1e6,
        TOPK, (TOPK * MATS * mat_bytes) as f64 / 1e6
    );

    let w = vec![0xA5u8; EXPERTS * MATS * mat_bytes];
    let x = vec![7i8; D];
    let sumx: i32 = x.iter().map(|&v| v as i32).sum();
    let mut y = vec![0i32; TOPK * MATS * D];

    // pre-rolled expert sets per token (top-4 of 32, no repeats)
    let ntokens = 200;
    let mut rng = 0x243f6a8885a308d3u64;
    let sets: Vec<[usize; TOPK]> = (0..ntokens)
        .map(|_| {
            let mut s = [0usize; TOPK];
            let mut chosen = [false; EXPERTS];
            let mut i = 0;
            while i < TOPK {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let e = (rng % EXPERTS as u64) as usize;
                if !chosen[e] {
                    chosen[e] = true;
                    s[i] = e;
                    i += 1;
                }
            }
            s
        })
        .collect();

    let active_mats = TOPK * MATS;
    let total_quads = active_mats * quads_per_mat;
    let bytes_per_token = (active_mats * mat_bytes) as f64;
    let yp = SendPtr(y.as_mut_ptr());

    for nthreads in [4usize, 6, 8, 10] {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for t in 0..nthreads {
                    let (sets, w, x) = (&sets, &w, &x);
                    s.spawn(move || {
                        let y = yp.get();
                        let chunk = (total_quads + nthreads - 1) / nthreads;
                        let q0 = t * chunk;
                        let q1 = (q0 + chunk).min(total_quads);
                        for tok in 0..ntokens {
                            let set = &sets[tok];
                            for gq in q0..q1 {
                                let mat_i = gq / quads_per_mat; // 0..12 active mat index
                                let quad = gq % quads_per_mat;
                                let expert = set[mat_i / MATS];
                                let mg = expert * MATS + mat_i % MATS; // global mat
                                let base = mg * mat_bytes + quad * qstride;
                                let acc = unsafe { dot4_q4(w.as_ptr().add(base), x.as_ptr(), D) };
                                for i in 0..4 {
                                    unsafe { *y.add(mat_i * D + quad * 4 + i) = acc[i] - 8 * sumx };
                                }
                            }
                        }
                    });
                }
            });
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let per_tok = best / ntokens as f64;
        let gbs = bytes_per_token / per_tok / 1e9;
        // whole-model projection: 24 layers of (12 expert mats + 4 dense attn mats)
        let model_bytes_per_token = (LAYERS * (active_mats + 4) * mat_bytes) as f64;
        let model_toks = gbs * 1e9 / model_bytes_per_token;
        println!(
            "threads={nthreads:<3} {gbs:>6.1} GB/s on expert weights | layer {:.3} ms | full-model proj ~{model_toks:.0} tok/s",
            per_tok * 1e3
        );
    }
    println!("\nreference: llama.cpp CPU on gpt-oss:20b = 14.7 tok/s (E7)");
}
