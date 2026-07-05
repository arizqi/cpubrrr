//! Qwen3-MoE (qwen3moe) generation engine — Qwen3-Coder-30B-A3B / Qwen3-30B-A3B.
//! Correct-first: k-quant (Q4_K/Q6_K) weights are dequantized to int8-per-32-block at
//! load (verified dequant, see qk_verify), then the whole engine runs on the proven
//! Q8 sdot path. Arch specifics per docs/QWEN3_SPEC.md (verified vs llama.cpp):
//! QK-norm, plain SwiGLU (silu), softmax-then-topk-renorm router, no sinks/SWA/bias.
//! Usage: engine_qwen <data-dir> <blob> [prompt]

use std::arch::aarch64::*;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

const NT: usize = 12;
const MAXSEQ: usize = 4096;

// ---------------- config ----------------
#[derive(Clone, Copy)]
struct Cfg {
    d: usize, nh: usize, nkv: usize, hd: usize, nl: usize, ne: usize, topk: usize,
    ffexp: usize, nvocab: usize, rope_base: f32, rms_eps: f32,
}
static CFG: std::sync::OnceLock<Cfg> = std::sync::OnceLock::new();
#[inline(always)]
fn cfg() -> &'static Cfg { CFG.get().unwrap() }
fn load_cfg(dir: &str) {
    let mut m = HashMap::new();
    for l in fs::read_to_string(format!("{dir}/config.txt")).unwrap().lines() {
        let mut it = l.split_whitespace();
        if let (Some(k), Some(v)) = (it.next(), it.next()) { m.insert(k.to_string(), v.to_string()); }
    }
    let gi = |k: &str| m[k].parse::<usize>().unwrap();
    let gf = |k: &str| m[k].parse::<f32>().unwrap();
    CFG.set(Cfg {
        d: gi("n_embd"), nh: gi("n_head"), nkv: gi("n_head_kv"), hd: gi("head_dim"),
        nl: gi("n_layer"), ne: gi("n_expert"), topk: gi("n_expert_used"),
        ffexp: gi("n_ff_exp"), nvocab: gi("n_vocab"),
        rope_base: gf("rope_freq_base"), rms_eps: gf("rms_eps"),
    }).ok();
}

// ---------------- k-quant dequant (verified vs gguf oracle) ----------------
fn half_to_f32(h: u16) -> f32 {
    let (s, e, mant) = ((h >> 15) as u32 & 1, (h >> 10) as u32 & 0x1f, (h & 0x3ff) as u32);
    let bits = if e == 0 {
        if mant == 0 { s << 31 } else {
            let mut ex = -1i32; let mut mm = mant;
            loop { ex += 1; mm <<= 1; if mm & 0x400 != 0 { break; } }
            (s << 31) | (((112 - ex) as u32) << 23) | ((mm & 0x3ff) << 13)
        }
    } else if e == 0x1f { (s << 31) | (0xff << 23) | (mant << 13) }
    else { (s << 31) | ((e + 112) << 23) | (mant << 13) };
    f32::from_bits(bits)
}
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) }
    else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}
fn dequant_q4k(b: &[u8], out: &mut [f32]) {
    let d = half_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let dmin = half_to_f32(u16::from_le_bytes([b[2], b[3]]));
    let (scales, qs) = (&b[4..16], &b[16..144]);
    let (mut y, mut is, mut qo) = (0, 0, 0);
    while y < 256 {
        let (sc1, m1) = scale_min_k4(is, scales);
        let (sc2, m2) = scale_min_k4(is + 1, scales);
        let (d1, mm1, d2, mm2) = (d * sc1 as f32, dmin * m1 as f32, d * sc2 as f32, dmin * m2 as f32);
        for l in 0..32 { out[y + l] = d1 * (qs[qo + l] & 0xF) as f32 - mm1; }
        for l in 0..32 { out[y + 32 + l] = d2 * (qs[qo + l] >> 4) as f32 - mm2; }
        y += 64; is += 2; qo += 32;
    }
}
fn dequant_q6k(b: &[u8], out: &mut [f32]) {
    let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
    let d = half_to_f32(u16::from_le_bytes([b[208], b[209]]));
    let sci = |i: usize| sc[i] as i8 as f32;
    let (mut base, mut yb) = (0, 0);
    while yb < 256 {
        let (ql, qh) = (&ql[base..], &qh[base / 2..]);
        let scb = (yb / 128) * 8;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            out[yb + l] = d * sci(scb + is) * q1 as f32;
            out[yb + l + 32] = d * sci(scb + is + 2) * q2 as f32;
            out[yb + l + 64] = d * sci(scb + is + 4) * q3 as f32;
            out[yb + l + 96] = d * sci(scb + is + 6) * q4 as f32;
        }
        base += 64; yb += 128;
    }
}

// ---------------- kernels ----------------
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        let mut r = acc;
        std::arch::asm!("sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
            r = inout(vreg) r, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
        r
    }
}
unsafe fn dot_q8(wq: *const i8, ws: *const f32, xq: *const i8, xsb: *const f32, cols: usize) -> f32 {
    unsafe {
        let mut accf = vdupq_n_f32(0.0);
        for b in 0..cols / 32 {
            let d = sdot(sdot(vdupq_n_s32(0), vld1q_s8(wq.add(b * 32)), vld1q_s8(xq.add(b * 32))),
                         vld1q_s8(wq.add(b * 32 + 16)), vld1q_s8(xq.add(b * 32 + 16)));
            accf = vfmaq_n_f32(accf, vcvtq_f32_s32(d), *ws.add(b) * *xsb.add(b));
        }
        vaddvq_f32(accf)
    }
}
fn quant_i8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let nb = x.len() / 32;
    let mut q = vec![0i8; x.len()];
    let mut s = vec![0f32; nb];
    for b in 0..nb {
        let blk = &x[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(1e-12f32, |m, &v| m.max(v.abs()));
        let sc = amax / 127.0;
        s[b] = sc;
        for (i, &v) in blk.iter().enumerate() { q[b * 32 + i] = (v / sc).round().clamp(-127.0, 127.0) as i8; }
    }
    (q, s)
}
fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + cfg().rms_eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}
/// RMSNorm over a single head vector (len hd), weight len hd. No eps-add mismatch: qwen uses same eps.
fn rmsnorm_head(v: &mut [f32], w: &[f32]) {
    let hd = v.len();
    let ms = v.iter().map(|x| x * x).sum::<f32>() / hd as f32;
    let inv = 1.0 / (ms + cfg().rms_eps).sqrt();
    for i in 0..hd { v[i] = v[i] * inv * w[i]; }
}
fn rope(v: &mut [f32], pos: usize) {
    let hd = cfg().hd;
    let half = hd / 2;
    let base = cfg().rope_base;
    for i in 0..half {
        let theta = pos as f32 * base.powf(-(2.0 * i as f32) / hd as f32);
        let (c, s) = (theta.cos(), theta.sin());
        let (a, b) = (v[i], v[half + i]);
        v[i] = a * c - b * s;
        v[half + i] = b * c + a * s;
    }
}

// ---------------- Q8 weight matrix (dequant from k-quant at load) ----------------
struct Q8Mat { q: Vec<i8>, s: Vec<f32>, cols: usize }
impl Q8Mat {
    #[inline(always)]
    unsafe fn dot(&self, row: usize, xq: *const i8, xsb: *const f32) -> f32 {
        unsafe { dot_q8(self.q.as_ptr().add(row * self.cols), self.s.as_ptr().add(row * (self.cols / 32)), xq, xsb, self.cols) }
    }
}
/// Dequant a whole tensor (rows x cols, ggml quant) into Q8Mat (int8 + per-32 scale).
fn dequant_to_q8(raw: &[u8], ttype: u32, rows: usize, cols: usize) -> Q8Mat {
    let nb = cols / 32;
    let mut q = vec![0i8; rows * cols];
    let mut s = vec![0f32; rows * nb];
    let qp = SendI8(q.as_mut_ptr());
    let sp = SendF32(s.as_mut_ptr());
    // bytes per row for the source quant
    let (bpr, sbsz) = match ttype { 12 => (cols / 256 * 144, 256usize), 14 => (cols / 256 * 210, 256), 0 => (cols * 4, 0), _ => panic!("type {ttype}") };
    par_rows(rows, |a, b| {
        let mut buf = vec![0f32; cols];
        for r in a..b {
            match ttype {
                0 => { // f32 source
                    for c in 0..cols { buf[c] = f32::from_le_bytes(raw[(r * cols + c) * 4..(r * cols + c) * 4 + 4].try_into().unwrap()); }
                }
                12 | 14 => {
                    let row = &raw[r * bpr..(r + 1) * bpr];
                    let mut tmp = [0f32; 256];
                    let nsb = cols / sbsz;
                    for sb in 0..nsb {
                        let blk = &row[sb * (if ttype == 12 { 144 } else { 210 })..];
                        if ttype == 12 { dequant_q4k(blk, &mut tmp); } else { dequant_q6k(blk, &mut tmp); }
                        buf[sb * 256..sb * 256 + 256].copy_from_slice(&tmp);
                    }
                }
                _ => unreachable!(),
            }
            // quantize row to int8 per-32-block
            for bl in 0..nb {
                let blk = &buf[bl * 32..bl * 32 + 32];
                let amax = blk.iter().fold(1e-12f32, |m, &v| m.max(v.abs()));
                let sc = amax / 127.0;
                unsafe { *sp.get().add(r * nb + bl) = sc; }
                for i in 0..32 { unsafe { *qp.get().add(r * cols + bl * 32 + i) = (blk[i] / sc).round().clamp(-127.0, 127.0) as i8; } }
            }
        }
    });
    Q8Mat { q, s, cols }
}

#[derive(Clone, Copy)] struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {} unsafe impl Sync for SendPtr {}
impl SendPtr { fn get(self) -> *mut f32 { self.0 } }
#[derive(Clone, Copy)] struct SendI8(*mut i8);
unsafe impl Send for SendI8 {}
unsafe impl Sync for SendI8 {}
impl SendI8 { fn get(self) -> *mut i8 { self.0 } }
#[derive(Clone, Copy)] struct SendF32(*mut f32);
unsafe impl Send for SendF32 {}
unsafe impl Sync for SendF32 {}
impl SendF32 { fn get(self) -> *mut f32 { self.0 } }

fn par_rows<F: Fn(usize, usize) + Sync>(n: usize, f: F) {
    let chunk = (n + NT - 1) / NT;
    std::thread::scope(|s| {
        for t in 0..NT {
            let a = t * chunk;
            if a >= n { break; }
            let b = (a + chunk).min(n);
            let f = &f;
            s.spawn(move || f(a, b));
        }
    });
}

// ---------------- model ----------------
struct Raw { blob: Vec<u8>, idx: HashMap<String, (u32, usize, usize)> }
impl Raw {
    fn f32v(&self, name: &str) -> Vec<f32> {
        let &(t, off, n) = &self.idx[name];
        assert_eq!(t, 0, "{name} not f32");
        self.blob[off..off + n * 4].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn q8(&self, name: &str, rows: usize, cols: usize) -> Q8Mat {
        let &(t, off, n) = &self.idx[name];
        let bytes = match t { 12 => n / 256 * 144, 14 => n / 256 * 210, 0 => n * 4, _ => panic!() };
        dequant_to_q8(&self.blob[off..off + bytes], t, rows, cols)
    }
    fn q8_slice(&self, name: &str) -> (u32, usize, usize) { self.idx[name] }
}

struct Layer {
    attn_norm: Vec<f32>, ffn_norm: Vec<f32>, q_norm: Vec<f32>, k_norm: Vec<f32>,
    wq: Q8Mat, wk: Q8Mat, wv: Q8Mat, wo: Q8Mat,
    gate_inp: Vec<f32>,
    gate: Q8Mat, up: Q8Mat, down: Q8Mat, // experts flattened: (ne*ffexp) rows for gate/up; (ne*d) rows for down
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let blobp = &args[2];
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Write a haiku about CPUs.".into());
    load_cfg(dir);
    let c = *cfg();

    // tokenizer
    let tb = fs::read(format!("{dir}/tokens.bin")).unwrap();
    let ntok = u32::from_le_bytes(tb[0..4].try_into().unwrap()) as usize;
    let mut toks: Vec<&[u8]> = Vec::with_capacity(ntok);
    let mut p = 4;
    for _ in 0..ntok {
        let l = u16::from_le_bytes(tb[p..p + 2].try_into().unwrap()) as usize;
        toks.push(&tb[p + 2..p + 2 + l]); p += 2 + l;
    }
    let mut lookup: HashMap<&[u8], u32> = HashMap::new();
    for (i, t) in toks.iter().enumerate() {
        if !t.is_empty() && !t.starts_with(b"<|") { lookup.entry(*t).or_insert(i as u32); }
    }
    let sp: HashMap<&[u8], u32> = toks.iter().enumerate()
        .filter(|(_, t)| t.starts_with(b"<|")).map(|(i, t)| (*t, i as u32)).collect();
    let encode = |s: &str| -> Vec<u32> {
        let b = s.as_bytes();
        let mut ids = Vec::new();
        let mut i = 0;
        while i < b.len() {
            let mut l = (b.len() - i).min(48);
            loop { if let Some(&id) = lookup.get(&b[i..i + l]) { ids.push(id); i += l; break; } l -= 1; assert!(l > 0, "unenc {}", b[i]); }
        }
        ids
    };
    let tok = |t: &str| sp[t.as_bytes()];

    // model
    eprintln!("loading blob + dequantizing k-quants to int8 (one-time)...");
    let t0 = Instant::now();
    let blob = fs::read(blobp).unwrap();
    let mut idx = HashMap::new();
    let man = fs::read_to_string(format!("{dir}/manifest.txt")).unwrap();
    let mut lines = man.lines();
    let ds: usize = lines.next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    for l in lines {
        let f: Vec<&str> = l.split_whitespace().collect();
        idx.insert(f[0].to_string(), (f[1].parse().unwrap(), ds + f[2].parse::<usize>().unwrap(), f[3].parse().unwrap()));
    }
    let raw = Raw { blob, idx };
    let (d, nh, nkv, hd, ne, ff) = (c.d, c.nh, c.nkv, c.hd, c.ne, c.ffexp);
    let layers: Vec<Layer> = (0..c.nl).map(|i| {
        let pfx = format!("blk.{i}");
        Layer {
            attn_norm: raw.f32v(&format!("{pfx}.attn_norm.weight")),
            ffn_norm: raw.f32v(&format!("{pfx}.ffn_norm.weight")),
            q_norm: raw.f32v(&format!("{pfx}.attn_q_norm.weight")),
            k_norm: raw.f32v(&format!("{pfx}.attn_k_norm.weight")),
            wq: raw.q8(&format!("{pfx}.attn_q.weight"), nh * hd, d),
            wk: raw.q8(&format!("{pfx}.attn_k.weight"), nkv * hd, d),
            wv: raw.q8(&format!("{pfx}.attn_v.weight"), nkv * hd, d),
            wo: raw.q8(&format!("{pfx}.attn_output.weight"), d, nh * hd),
            gate_inp: raw.f32v(&format!("{pfx}.ffn_gate_inp.weight")),
            gate: raw.q8(&format!("{pfx}.ffn_gate_exps.weight"), ne * ff, d),
            up: raw.q8(&format!("{pfx}.ffn_up_exps.weight"), ne * ff, d),
            down: raw.q8(&format!("{pfx}.ffn_down_exps.weight"), ne * d, ff),
        }
    }).collect();
    let tok_embd = raw.q8_slice("token_embd.weight"); // Q4_K
    let out_norm = raw.f32v("output_norm.weight");
    let head = raw.q8("output.weight", c.nvocab, d);
    eprintln!("loaded+dequant in {:.1}s", t0.elapsed().as_secs_f64());

    // embedding lookup: dequant one Q4_K row of token_embd
    let embed = |t: u32| -> Vec<f32> {
        let (tt, off, _) = tok_embd;
        assert_eq!(tt, 12);
        let bpr = d / 256 * 144;
        let row = &raw.blob[off + t as usize * bpr..off + (t as usize + 1) * bpr];
        let mut out = vec![0f32; d];
        let mut tmp = [0f32; 256];
        for sb in 0..d / 256 { dequant_q4k(&row[sb * 144..], &mut tmp); out[sb * 256..sb * 256 + 256].copy_from_slice(&tmp); }
        out
    };

    let mut kc = vec![vec![0f32; MAXSEQ * nkv * hd]; c.nl];
    let mut vc = vec![vec![0f32; MAXSEQ * nkv * hd]; c.nl];

    let mut forward = |t: u32, pos: usize, kc: &mut Vec<Vec<f32>>, vc: &mut Vec<Vec<f32>>| -> u32 {
        let mut x = embed(t);
        for (il, ly) in layers.iter().enumerate() {
            // attention
            let xn = rmsnorm(&x, &ly.attn_norm);
            let (xq, xs) = quant_i8(&xn);
            let mut q = vec![0f32; nh * hd];
            let mut k = vec![0f32; nkv * hd];
            let mut v = vec![0f32; nkv * hd];
            {
                let (qp, kp, vp) = (SendPtr(q.as_mut_ptr()), SendPtr(k.as_mut_ptr()), SendPtr(v.as_mut_ptr()));
                let (xq, xs) = (&xq, &xs);
                par_rows(nh * hd + 2 * nkv * hd, |a, b| {
                    for r in a..b {
                        unsafe {
                            if r < nh * hd { *qp.get().add(r) = ly.wq.dot(r, xq.as_ptr(), xs.as_ptr()); }
                            else if r < nh * hd + nkv * hd { let rr = r - nh * hd; *kp.get().add(rr) = ly.wk.dot(rr, xq.as_ptr(), xs.as_ptr()); }
                            else { let rr = r - nh * hd - nkv * hd; *vp.get().add(rr) = ly.wv.dot(rr, xq.as_ptr(), xs.as_ptr()); }
                        }
                    }
                });
            }
            // QK-norm (per head) then RoPE
            for h in 0..nh { rmsnorm_head(&mut q[h * hd..(h + 1) * hd], &ly.q_norm); rope(&mut q[h * hd..(h + 1) * hd], pos); }
            for h in 0..nkv { rmsnorm_head(&mut k[h * hd..(h + 1) * hd], &ly.k_norm); rope(&mut k[h * hd..(h + 1) * hd], pos); }
            kc[il][pos * nkv * hd..(pos + 1) * nkv * hd].copy_from_slice(&k);
            vc[il][pos * nkv * hd..(pos + 1) * nkv * hd].copy_from_slice(&v);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut ao = vec![0f32; nh * hd];
            {
                let (kcl, vcl, qref) = (&kc[il], &vc[il], &q);
                let out = SendPtr(ao.as_mut_ptr());
                par_rows(nh, |a, b| {
                    for h in a..b {
                        let kvh = h / (nh / nkv);
                        let mut sc: Vec<f32> = (0..=pos).map(|t| {
                            let mut acc = 0.0;
                            for j in 0..hd { acc += kcl[(t * nkv + kvh) * hd + j] * qref[h * hd + j]; }
                            acc * scale
                        }).collect();
                        let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                        let mut den = 0.0;
                        for s in sc.iter_mut() { *s = (*s - mx).exp(); den += *s; }
                        for j in 0..hd {
                            let mut acc = 0.0;
                            for t in 0..=pos { acc += sc[t] * vcl[(t * nkv + kvh) * hd + j]; }
                            unsafe { *out.get().add(h * hd + j) = acc / den; }
                        }
                    }
                });
            }
            let (aq, asc) = quant_i8(&ao);
            {
                let out = SendPtr(x.as_mut_ptr());
                let (aq, asc) = (&aq, &asc);
                par_rows(d, |a, b| { for r in a..b { unsafe { *out.get().add(r) += ly.wo.dot(r, aq.as_ptr(), asc.as_ptr()); } } });
            }
            // MoE FFN
            let xn2 = rmsnorm(&x, &ly.ffn_norm);
            // router: softmax over all NE, then top-k, then renormalize
            let logits: Vec<f32> = (0..ne).map(|e| { let mut a = 0.0; for j in 0..d { a += ly.gate_inp[e * d + j] * xn2[j]; } a }).collect();
            let lmax = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - lmax).exp()).collect();
            let sm: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sm).collect();
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let top = &order[..c.topk];
            let wsum: f32 = top.iter().map(|&e| probs[e]).sum();
            let wts: Vec<f32> = top.iter().map(|&e| probs[e] / wsum).collect();

            let (x2q, x2s) = quant_i8(&xn2);
            let mut ffn = vec![0f32; d];
            let contrib = SendPtr(ffn.as_mut_ptr());
            for (ei, &e) in top.iter().enumerate() {
                let w = wts[ei];
                // h = silu(gate·x) * (up·x), per ffexp row
                let mut h = vec![0f32; ff];
                {
                    let (x2q, x2s) = (&x2q, &x2s);
                    let hp = SendPtr(h.as_mut_ptr());
                    par_rows(ff, |a, b| {
                        for r in a..b {
                            let gr = e * ff + r;
                            let g = unsafe { ly.gate.dot(gr, x2q.as_ptr(), x2s.as_ptr()) };
                            let u = unsafe { ly.up.dot(gr, x2q.as_ptr(), x2s.as_ptr()) };
                            let silu = g / (1.0 + (-g).exp());
                            unsafe { *hp.get().add(r) = silu * u; }
                        }
                    });
                }
                let (hq, hs) = quant_i8(&h);
                {
                    let (hq, hs) = (&hq, &hs);
                    par_rows(d, |a, b| {
                        for r in a..b {
                            let dr = e * d + r;
                            let dv = unsafe { ly.down.dot(dr, hq.as_ptr(), hs.as_ptr()) };
                            unsafe { *contrib.get().add(r) += w * dv; }
                        }
                    });
                }
            }
            for r in 0..d { x[r] += ffn[r]; }
        }
        // head
        let xn = rmsnorm(&x, &out_norm);
        let (xq, xs) = quant_i8(&xn);
        let mut best = vec![(f32::MIN, 0u32); NT];
        let bp = best.as_mut_ptr() as usize;
        let nv = c.nvocab;
        std::thread::scope(|s| {
            let chunk = (nv + NT - 1) / NT;
            for t in 0..NT {
                let (a, b) = (t * chunk, ((t + 1) * chunk).min(nv));
                let (xq, xs, head) = (&xq, &xs, &head);
                s.spawn(move || {
                    let mut m = (f32::MIN, 0u32);
                    for r in a..b { let l = unsafe { head.dot(r, xq.as_ptr(), xs.as_ptr()) }; if l > m.0 { m = (l, r as u32); } }
                    unsafe { *(bp as *mut (f32, u32)).add(t) = m; }
                });
            }
        });
        best.iter().cloned().fold((f32::MIN, 0u32), |m, v| if v.0 > m.0 { v } else { m }).1
    };

    // ChatML prompt
    let mut ids: Vec<u32> = Vec::new();
    ids.push(tok("<|im_start|>")); ids.extend(encode("system\nYou are a helpful assistant.")); ids.push(tok("<|im_end|>"));
    ids.push(tok("<|im_start|>")); ids.extend(encode(&format!("\nuser\n{prompt}"))); ids.push(tok("<|im_end|>"));
    ids.push(tok("<|im_start|>")); ids.extend(encode("\nassistant\n"));
    eprintln!("prompt: {} tokens", ids.len());

    let im_end = tok("<|im_end|>");
    let t1 = Instant::now();
    let mut next = 0u32;
    for (i, &t) in ids.iter().enumerate() { next = forward(t, i, &mut kc, &mut vc); let _ = i; }
    eprintln!("prefill {} tok in {:.2}s", ids.len(), t1.elapsed().as_secs_f64());

    let ngen = 64;
    let t2 = Instant::now();
    let mut pos = ids.len();
    let mut nout = 0;
    for _ in 0..ngen {
        if next == im_end || next as usize >= ntok { break; }
        print!("{}", String::from_utf8_lossy(toks[next as usize]));
        use std::io::Write; std::io::stdout().flush().ok();
        next = forward(next, pos, &mut kc, &mut vc);
        pos += 1; nout += 1;
    }
    let dt = t2.elapsed().as_secs_f64();
    println!("\n---\ndecode: {nout} tokens in {dt:.2}s = {:.1} tok/s", nout as f64 / dt);
}
