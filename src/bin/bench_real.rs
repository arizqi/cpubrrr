//! E9: real gpt-oss:20b expert weights (MXFP4) through our MoE decode kernel.
//! Exact MXFP4 math: nibble -> int8 via tbl (values x2), sdot integer dots,
//! per-block e8m0 scale applied via fmla (x2 folded into scale as 2^(e-129)).

use std::arch::aarch64::*;
use std::time::Instant;

const D: usize = 2880;
const EXPERTS: usize = 32;
const TOPK: usize = 4;
const MATS: usize = 3;
const LAYERS: usize = 24;
const BLOCKS: usize = D / 32; // 90 blocks/row
const BPR: usize = BLOCKS * 17; // raw bytes/row

const KV: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

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

/// Quad-interleaved MXFP4: nibbles [quad][block][row 0..4][16B], scales [quad][block][4]f32.
struct Packed {
    nib: Vec<u8>,
    scale: Vec<f32>,
}

fn repack(raw: &[u8], rows: usize) -> Packed {
    let nquads = rows / 4;
    let mut nib = vec![0u8; rows * BLOCKS * 16];
    let mut scale = vec![0f32; rows * BLOCKS];
    for q in 0..nquads {
        for b in 0..BLOCKS {
            for r in 0..4 {
                let row = q * 4 + r;
                let blk = &raw[row * BPR + b * 17..row * BPR + b * 17 + 17];
                // x2 fold: e8m0 gives 2^(e-127); table is value*2 -> scale 2^(e-128)
                scale[(q * BLOCKS + b) * 4 + r] = f32::powi(2.0, blk[0] as i32 - 128);
                nib[(q * BLOCKS + b) * 64 + r * 16..][..16].copy_from_slice(&blk[1..17]);
            }
        }
    }
    Packed { nib, scale }
}

/// 4 rows x full D dot against int8 x. Exact MXFP4.
unsafe fn dot4_mxfp4(nib: *const u8, scale: *const f32, x: *const i8) -> [f32; 4] {
    unsafe {
        let kv = vld1q_s8(KV.as_ptr());
        let mask = vdupq_n_u8(0x0F);
        let mut accf = [vdupq_n_f32(0.0); 4];
        let mut np = nib;
        let mut sp = scale;
        for b in 0..BLOCKS {
            let x0 = vld1q_s8(x.add(b * 32));
            let x1 = vld1q_s8(x.add(b * 32 + 16));
            let sv = vld1q_f32(sp);
            let w0 = vld1q_u8(np);
            let w1 = vld1q_u8(np.add(16));
            let w2 = vld1q_u8(np.add(32));
            let w3 = vld1q_u8(np.add(48));
            let z = vdupq_n_s32(0);
            let t0 = sdot(sdot(z, vqtbl1q_s8(kv, vandq_u8(w0, mask)), x0), vqtbl1q_s8(kv, vshrq_n_u8::<4>(w0)), x1);
            let t1 = sdot(sdot(z, vqtbl1q_s8(kv, vandq_u8(w1, mask)), x0), vqtbl1q_s8(kv, vshrq_n_u8::<4>(w1)), x1);
            let t2 = sdot(sdot(z, vqtbl1q_s8(kv, vandq_u8(w2, mask)), x0), vqtbl1q_s8(kv, vshrq_n_u8::<4>(w2)), x1);
            let t3 = sdot(sdot(z, vqtbl1q_s8(kv, vandq_u8(w3, mask)), x0), vqtbl1q_s8(kv, vshrq_n_u8::<4>(w3)), x1);
            accf[0] = vfmaq_laneq_f32::<0>(accf[0], vcvtq_f32_s32(t0), sv);
            accf[1] = vfmaq_laneq_f32::<1>(accf[1], vcvtq_f32_s32(t1), sv);
            accf[2] = vfmaq_laneq_f32::<2>(accf[2], vcvtq_f32_s32(t2), sv);
            accf[3] = vfmaq_laneq_f32::<3>(accf[3], vcvtq_f32_s32(t3), sv);
            np = np.add(64);
            sp = sp.add(4);
        }
        [vaddvq_f32(accf[0]), vaddvq_f32(accf[1]), vaddvq_f32(accf[2]), vaddvq_f32(accf[3])]
    }
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

fn main() {
    let dir = std::env::args().nth(1).expect("usage: bench_real <extract dir>");
    let x: Vec<i8> = std::fs::read(format!("{dir}/x.bin")).unwrap().iter().map(|&b| b as i8).collect();
    let yref: Vec<f32> = std::fs::read(format!("{dir}/yref.bin")).unwrap()
        .chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    println!("== E9 real gpt-oss:20b expert weights ==");
    let mats: Vec<Packed> = ["gate", "up", "down"].iter().map(|m| {
        let raw = std::fs::read(format!("{dir}/{m}.mxfp4")).unwrap();
        assert_eq!(raw.len(), EXPERTS * D * BPR);
        repack(&raw, EXPERTS * D)
    }).collect();
    println!("repacked 3 x {} MB (32 experts each)", EXPERTS * D * BPR / 1_000_000);

    // verify: first 8 rows of gate expert 0 vs Python f64 reference
    let mut maxrel = 0f32;
    for q in 0..2 {
        let y = unsafe {
            dot4_mxfp4(
                mats[0].nib.as_ptr().add(q * BLOCKS * 64),
                mats[0].scale.as_ptr().add(q * BLOCKS * 4),
                x.as_ptr(),
            )
        };
        for i in 0..4 {
            let r = yref[q * 4 + i];
            let rel = ((y[i] - r) / r.abs().max(1e-3)).abs();
            maxrel = maxrel.max(rel);
        }
    }
    assert!(maxrel < 1e-4, "mismatch vs reference: max rel err {maxrel}");
    println!("correctness vs f64 dequant reference: max rel err {maxrel:.2e} PASS");

    // MoE decode bench: top-4 experts x 3 mats per token, flat quad work list
    let quads_per_mat = D / 4;
    let active = TOPK * MATS;
    let total_quads = active * quads_per_mat;
    let real_bytes_per_token = (active * D * BPR) as f64; // includes scale bytes
    let ntokens = 200;
    let mut rng = 0x243f6a8885a308d3u64;
    let sets: Vec<[usize; TOPK]> = (0..ntokens).map(|_| {
        let mut s = [0usize; TOPK];
        let mut chosen = [false; EXPERTS];
        let mut i = 0;
        while i < TOPK {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let e = (rng % EXPERTS as u64) as usize;
            if !chosen[e] { chosen[e] = true; s[i] = e; i += 1; }
        }
        s
    }).collect();
    let mut y = vec![0f32; active * D];
    let yp = SendPtr(y.as_mut_ptr());
    let equads = D / 4 * BLOCKS; // quad-blocks per expert offset unit
    let _ = equads;

    for nthreads in [6usize, 8, 10] {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for t in 0..nthreads {
                    let (sets, mats, x) = (&sets, &mats, &x);
                    s.spawn(move || {
                        let y = yp.get();
                        let chunk = (total_quads + nthreads - 1) / nthreads;
                        let q0 = t * chunk;
                        let q1 = (q0 + chunk).min(total_quads);
                        for tok in 0..ntokens {
                            let set = &sets[tok];
                            for gq in q0..q1 {
                                let mat_i = gq / quads_per_mat;
                                let quad = gq % quads_per_mat;
                                let expert = set[mat_i / MATS];
                                let m = &mats[mat_i % MATS];
                                let qg = expert * quads_per_mat + quad; // quad index within mat
                                let acc = unsafe {
                                    dot4_mxfp4(
                                        m.nib.as_ptr().add(qg * BLOCKS * 64),
                                        m.scale.as_ptr().add(qg * BLOCKS * 4),
                                        x.as_ptr(),
                                    )
                                };
                                for i in 0..4 {
                                    unsafe { *y.add(mat_i * D + quad * 4 + i) = acc[i] };
                                }
                            }
                        }
                    });
                }
            });
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let per_tok = best / ntokens as f64;
        let gbs = real_bytes_per_token / per_tok / 1e9;
        let model_bytes = (LAYERS * (active + 4) * D * BPR) as f64; // + 4 attn-sized mats
        let proj = gbs * 1e9 / model_bytes;
        println!(
            "threads={nthreads:<3} {gbs:>6.1} GB/s real MXFP4 | layer {:.3} ms | full-model proj ~{proj:.0} tok/s",
            per_tok * 1e3
        );
    }
    println!("\nreference: llama.cpp CPU same model = 14.7 tok/s (E7)");
}
