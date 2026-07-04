//! E10: exact gpt-oss layer-0 MoE-FFN forward in Rust on real weights.
//! Verifies the recovered spec (router top-4 softmax, SwiGLU-OAI a=1.702 lim=7,
//! weighted expert combine) against the numpy f64 reference. Correctness milestone,
//! not the perf path (that is bench_real.rs's int8 kernel).

use std::fs;

const D: usize = 2880;
const NE: usize = 32;
const TOPK: usize = 4;
const BPR: usize = D / 32 * 17;
const KV: [f32; 16] = [0., 1., 2., 3., 4., 6., 8., 12., 0., -1., -2., -3., -4., -6., -8., -12.];

fn read_f32(p: &str) -> Vec<f32> {
    fs::read(p).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Dequantize one MXFP4 expert (D rows x D) to f32.
fn deq_expert(raw: &[u8], e: usize) -> Vec<f32> {
    let mut out = vec![0f32; D * D];
    for r in 0..D {
        let row = &raw[(e * D + r) * BPR..(e * D + r + 1) * BPR];
        for b in 0..D / 32 {
            let blk = &row[b * 17..b * 17 + 17];
            let s = 2f32.powi(blk[0] as i32 - 127);
            for j in 0..16 {
                out[r * D + b * 32 + j] = KV[(blk[1 + j] & 0xF) as usize] * s;
                out[r * D + b * 32 + 16 + j] = KV[(blk[1 + j] >> 4) as usize] * s;
            }
        }
    }
    out
}

fn matvec(w: &[f32], x: &[f32], bias: &[f32], rows: usize) -> Vec<f32> {
    (0..rows).map(|r| {
        let mut a = 0f32;
        for j in 0..D {
            a += w[r * D + j] * x[j];
        }
        a + bias[r]
    }).collect()
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

    // router: logits = ginp @ x + b, over 32 experts
    let logits: Vec<f32> = (0..NE).map(|e| {
        let mut a = ginp_b[e];
        for j in 0..D {
            a += ginp[e * D + j] * x[j];
        }
        a
    }).collect();

    // top-4
    let mut order: Vec<usize> = (0..NE).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let top = &order[..TOPK];

    // softmax over the 4 selected
    let mx = top.iter().map(|&e| logits[e]).fold(f32::MIN, f32::max);
    let exps: Vec<f32> = top.iter().map(|&e| (logits[e] - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let wts: Vec<f32> = exps.iter().map(|v| v / sum).collect();

    println!("top experts: {top:?}");
    println!("softmax weights: {:?}", wts.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());

    const ALPHA: f32 = 1.702;
    const LIM: f32 = 7.0;
    let mut out = vec![0f32; D];
    for (i, &e) in top.iter().enumerate() {
        let ge = deq_expert(&gate_raw, e);
        let ue = deq_expert(&up_raw, e);
        let de = deq_expert(&down_raw, e);
        let gv = matvec(&ge, &x, &gate_b[e * D..], D);
        let uv = matvec(&ue, &x, &up_b[e * D..], D);
        let h: Vec<f32> = (0..D).map(|k| {
            let xg = gv[k].min(LIM);
            let yu = uv[k].clamp(-LIM, LIM);
            (xg / (1.0 + (-ALPHA * xg).exp())) * (yu + 1.0)
        }).collect();
        let dv = matvec(&de, &h, &down_b[e * D..], D);
        for k in 0..D {
            out[k] += wts[i] * dv[k];
        }
    }

    let mut maxabs = 0f32;
    let mut sse = 0f64;
    let mut refn = 0f64;
    for k in 0..D {
        maxabs = maxabs.max((out[k] - yref[k]).abs());
        sse += (out[k] - yref[k]) as f64 * (out[k] - yref[k]) as f64;
        refn += yref[k] as f64 * yref[k] as f64;
    }
    let rel = (sse.sqrt() / refn.sqrt()) as f32;
    println!("out[:6]: {:?}", out[..6].iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
    println!("vs numpy f64 reference: max abs {maxabs:.4}, rel L2 {rel:.2e}");
    assert!(rel < 1e-3, "FFN mismatch");
    println!("PASS — exact gpt-oss MoE-FFN reproduced on real weights");
}
