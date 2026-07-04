//! E12: exact gpt-oss layer-0 attention forward in Rust, single decode step.
//! Verifies GQA (64 q / 8 kv), learned sinks, causal softmax, scale 1/sqrt(64),
//! NeoX RoPE @ freq_base 150000, against numpy reference. bf16 weights.

use std::fs;

const D: usize = 2880;
const NH: usize = 64;
const NKV: usize = 8;
const HD: usize = 64;
const T: usize = 8;
const FREQ_BASE: f32 = 150000.0;

fn f32v(p: &str) -> Vec<f32> {
    fs::read(p).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn matvec(w: &[f32], x: &[f32], b: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).map(|r| {
        let mut a = b[r];
        for j in 0..cols { a += w[r * cols + j] * x[j]; }
        a
    }).collect()
}

fn rope(vec: &mut [f32], pos: usize) {
    let half = HD / 2;
    for i in 0..half {
        let inv = FREQ_BASE.powf(-((i * 2) as f32) / HD as f32);
        let ang = pos as f32 * inv;
        let (c, s) = (ang.cos(), ang.sin());
        let a = vec[i];
        let b = vec[half + i];
        vec[i] = a * c - b * s;
        vec[half + i] = b * c + a * s;
    }
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let g = |n: &str| format!("{dir}/{n}");
    let aw = |n: &str| f32v(&g(&format!("aw_{n}.f32")));
    let anorm = aw("anorm");
    let wq = aw("wq");
    let bq = aw("bq");
    let wk = aw("wk");
    let bk = aw("bk");
    let wv = aw("wv");
    let bv = aw("bv");
    let wo = aw("wo");
    let bo = aw("bo");
    let sinks = aw("sinks");

    let h = f32v(&g("aw_h.f32"));
    let kcache = f32v(&g("aw_kcache.f32")); // T*NKV*HD
    let vcache = f32v(&g("aw_vcache.f32"));
    let yref = f32v(&g("aw_attn_yref.f32"));

    // rmsnorm
    let ms = h.iter().map(|v| v * v).sum::<f32>() / D as f32;
    let inv = 1.0 / (ms + 1e-5).sqrt();
    let x: Vec<f32> = (0..D).map(|i| h[i] * inv * anorm[i]).collect();

    let mut q = matvec(&wq, &x, &bq, NH * HD, D);
    let mut k = matvec(&wk, &x, &bk, NKV * HD, D);
    let v = matvec(&wv, &x, &bv, NKV * HD, D);
    for hd in 0..NH { rope(&mut q[hd * HD..hd * HD + HD], T); }
    for kv in 0..NKV { rope(&mut k[kv * HD..kv * HD + HD], T); }

    let scale = 1.0 / (HD as f32).sqrt();
    let mut attnout = vec![0f32; NH * HD];
    for hd in 0..NH {
        let kv = hd / (NH / NKV);
        // scores over T cached + 1 new
        let mut scores = vec![0f32; T + 1];
        for t in 0..T {
            let mut d = 0.0;
            for j in 0..HD { d += kcache[(t * NKV + kv) * HD + j] * q[hd * HD + j]; }
            scores[t] = d * scale;
        }
        let mut dn = 0.0;
        for j in 0..HD { dn += k[kv * HD + j] * q[hd * HD + j]; }
        scores[T] = dn * scale;
        // softmax with sink
        let m = scores.iter().cloned().fold(sinks[hd], f32::max);
        let es = (sinks[hd] - m).exp();
        let mut denom = es;
        let ex: Vec<f32> = scores.iter().map(|&s| { let e = (s - m).exp(); denom += e; e }).collect();
        for j in 0..HD {
            let mut acc = 0.0;
            for t in 0..T { acc += (ex[t] / denom) * vcache[(t * NKV + kv) * HD + j]; }
            acc += (ex[T] / denom) * v[kv * HD + j];
            attnout[hd * HD + j] = acc;
        }
    }
    let o = matvec(&wo, &attnout, &bo, D, NH * HD);

    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f32);
    for k in 0..D {
        dot += o[k] as f64 * yref[k] as f64;
        na += (o[k] as f64).powi(2);
        nb += (yref[k] as f64).powi(2);
        maxabs = maxabs.max((o[k] - yref[k]).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    println!("attn out[:6]: {:?}", o[..6].iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
    println!("vs numpy reference: cosine {cos:.6}, max abs {maxabs:.4}, ref norm {:.1}", nb.sqrt());
    assert!(cos > 0.99999, "attention mismatch");
    println!("PASS — exact gpt-oss attention (GQA + sinks + RoPE) reproduced on real weights");
}
