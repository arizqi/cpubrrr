//! E11: unite speed (E9 int8 MXFP4 kernel) + correctness (E10 exact FFN).
//! Quantize activations to int8 per-tensor per-token, run the fast integer path,
//! measure accuracy loss vs the f64 reference. Answers: does the 243 GB/s kernel
//! keep gpt-oss's output faithful?

use std::arch::aarch64::*;
use std::fs;

const D: usize = 2880;
const NE: usize = 32;
const TOPK: usize = 4;
const BLOCKS: usize = D / 32;
const BPR: usize = BLOCKS * 17;
const KV: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
const KVF: [f32; 16] = [0., 1., 2., 3., 4., 6., 8., 12., 0., -1., -2., -3., -4., -6., -8., -12.];

fn read_f32(p: &str) -> Vec<f32> {
    fs::read(p).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        let mut r = acc;
        std::arch::asm!("sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
            r = inout(vreg) r, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
        r
    }
}

/// Per-token int8 quantization: scale = max|x| / 127. Returns (q, scale).
fn quant_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let s = amax / 127.0;
    (x.iter().map(|&v| (v / s).round().clamp(-127.0, 127.0) as i8).collect(), s)
}

/// One expert row-block-scaled MXFP4 · int8-x dot, returning f32 (xscale folded in).
/// w row = BPR bytes. Uses exact MXFP4 nibble->int8 (values already integer) with
/// per-block e8m0 scale via float accumulate.
unsafe fn matvec_mxfp4_i8(raw: &[u8], e: usize, xq: &[i8], xs: f32, bias: &[f32], rows: usize) -> Vec<f32> {
    unsafe {
        let kv = vld1q_s8(KV.as_ptr());
        let mask = vdupq_n_u8(0x0F);
        let mut out = vec![0f32; rows];
        for r in 0..rows {
            let row = raw.as_ptr().add((e * D + r) * BPR);
            let mut accf = vdupq_n_f32(0.0);
            for b in 0..BLOCKS {
                let blk = row.add(b * 17);
                let scale = 2f32.powi(*blk as i32 - 127);
                let w = vld1q_u8(blk.add(1));
                let lo = vqtbl1q_s8(kv, vandq_u8(w, mask));
                let hi = vqtbl1q_s8(kv, vshrq_n_u8::<4>(w));
                let x0 = vld1q_s8(xq.as_ptr().add(b * 32));
                let x1 = vld1q_s8(xq.as_ptr().add(b * 32 + 16));
                let d = sdot(sdot(vdupq_n_s32(0), lo, x0), hi, x1);
                accf = vfmaq_n_f32(accf, vcvtq_f32_s32(d), scale);
            }
            out[r] = vaddvq_f32(accf) * xs + bias[r];
        }
        out
    }
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let g = |n: &str| format!("{dir}/{n}");
    let x = read_f32(&g("ffn_x.f32"));
    let yref = read_f32(&g("ffn_yref.f32"));
    let gate_raw = fs::read(g("gate.mxfp4")).unwrap();
    let up_raw = fs::read(g("up.mxfp4")).unwrap();
    let down_raw = fs::read(g("down.mxfp4")).unwrap();
    let gate_b = read_f32(&g("gate_b.f32"));
    let up_b = read_f32(&g("up_b.f32"));
    let down_b = read_f32(&g("down_b.f32"));
    let ginp = read_f32(&g("ginp.f32"));
    let ginp_b = read_f32(&g("ginp_b.f32"));

    // router in f32 (tiny, keep exact — routing errors flip experts = large error)
    let logits: Vec<f32> = (0..NE).map(|e| {
        let mut a = ginp_b[e];
        for j in 0..D { a += ginp[e * D + j] * x[j]; }
        a
    }).collect();
    let mut order: Vec<usize> = (0..NE).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let top = &order[..TOPK];
    let mx = top.iter().map(|&e| logits[e]).fold(f32::MIN, f32::max);
    let exps: Vec<f32> = top.iter().map(|&e| (logits[e] - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let wts: Vec<f32> = exps.iter().map(|v| v / sum).collect();

    const ALPHA: f32 = 1.702;
    const LIM: f32 = 7.0;
    let (xq, xs) = quant_i8(&x);
    let mut out = vec![0f32; D];
    for (i, &e) in top.iter().enumerate() {
        let gv = unsafe { matvec_mxfp4_i8(&gate_raw, e, &xq, xs, &gate_b[e * D..], D) };
        let uv = unsafe { matvec_mxfp4_i8(&up_raw, e, &xq, xs, &up_b[e * D..], D) };
        let h: Vec<f32> = (0..D).map(|k| {
            let xg = gv[k].min(LIM);
            let yu = uv[k].clamp(-LIM, LIM);
            (xg / (1.0 + (-ALPHA * xg).exp())) * (yu + 1.0)
        }).collect();
        let (hq, hs) = quant_i8(&h);
        let dv = unsafe { matvec_mxfp4_i8(&down_raw, e, &hq, hs, &down_b[e * D..], D) };
        for k in 0..D { out[k] += wts[i] * dv[k]; }
    }

    // also compute cosine similarity (what matters for next-token logits)
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut maxabs = 0f32;
    for k in 0..D {
        dot += out[k] as f64 * yref[k] as f64;
        na += (out[k] as f64).powi(2);
        nb += (yref[k] as f64).powi(2);
        maxabs = maxabs.max((out[k] - yref[k]).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let rel = ((na.sqrt() - nb.sqrt()).abs()) / nb.sqrt();
    let _ = KVF;
    println!("top experts: {top:?}  (routing kept in f32)");
    println!("int8-activation FFN vs f64 reference:");
    println!("  cosine similarity : {cos:.6}");
    println!("  rel L2 norm error : {:.3}%", rel * 100.0);
    println!("  max abs elem error: {maxabs:.3}  (ref norm {:.1})", nb.sqrt());
    println!("verdict: cosine >0.999 => next-token distribution effectively unchanged");
}
