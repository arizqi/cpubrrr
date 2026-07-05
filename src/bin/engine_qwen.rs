//! Qwen3-MoE generation engine (Qwen3-Coder-30B-A3B / Qwen3-30B-A3B).
//! Native 4-bit: Q4_K/Q6_K weights stay in the GGUF blob (18GB, 4-bit), matvec
//! dequant-inline via NEON sdot (kernels verified bit-exact, see qk_mv_verify).
//! Persistent thread pool + MoE expert batching. Arch per docs/QWEN3_SPEC.md.
//! Usage: engine_qwen <data-dir> <blob> [prompt]

use std::arch::aarch64::*;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

const NT: usize = 12;
const MAXSEQ: usize = 4096;

#[derive(Clone, Copy)]
struct Cfg { d: usize, nh: usize, nkv: usize, hd: usize, nl: usize, ne: usize, topk: usize,
    ffexp: usize, nvocab: usize, rope_base: f32, rms_eps: f32 }
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
    CFG.set(Cfg { d: gi("n_embd"), nh: gi("n_head"), nkv: gi("n_head_kv"), hd: gi("head_dim"),
        nl: gi("n_layer"), ne: gi("n_expert"), topk: gi("n_expert_used"), ffexp: gi("n_ff_exp"),
        nvocab: gi("n_vocab"), rope_base: gf("rope_freq_base"), rms_eps: gf("rms_eps") }).ok();
}

// ---- k-quant native matvec (verified) ----
fn half_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32 & 1, (h >> 10) as u32 & 0x1f, (h & 0x3ff) as u32);
    let bits = if e == 0 {
        if m == 0 { s << 31 } else {
            let mut ex = -1i32; let mut mm = m;
            loop { ex += 1; mm <<= 1; if mm & 0x400 != 0 { break; } }
            (s << 31) | (((112 - ex) as u32) << 23) | ((mm & 0x3ff) << 13)
        }
    } else if e == 0x1f { (s << 31) | (0xff << 23) | (m << 13) }
    else { (s << 31) | ((e + 112) << 23) | (m << 13) };
    f32::from_bits(bits)
}
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) }
    else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
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
unsafe fn q4k_dot(row: *const u8, xq: *const i8, xs: *const f32, xsum: *const i32, cols: usize) -> f32 {
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let mut facc = vdupq_n_f32(0.0);   // deferred: accumulate scale*sdot lanes, one reduce at end
        let mut minacc = 0f32;
        for sb in 0..cols / 256 {
            let b = row.add(sb * 144);
            let d = half_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
            let dmin = half_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
            let scales = std::slice::from_raw_parts(b.add(4), 12);
            let qs = b.add(16);
            for j in 0..8 {
                let (sc, m) = scale_min_k4(j, scales);
                let blk = sb * 8 + j;
                let qbase = qs.add((j / 2) * 32);
                let w0 = vld1q_u8(qbase);
                let w1 = vld1q_u8(qbase.add(16));
                let (n0, n1) = if j % 2 == 0 { (vandq_u8(w0, mask), vandq_u8(w1, mask)) }
                    else { (vshrq_n_u8::<4>(w0), vshrq_n_u8::<4>(w1)) };
                let s = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(n0), vld1q_s8(xq.add(blk * 32))),
                             vreinterpretq_s8_u8(n1), vld1q_s8(xq.add(blk * 32 + 16)));
                let xsb = *xs.add(blk);
                facc = vfmaq_n_f32(facc, vcvtq_f32_s32(s), xsb * d * sc as f32);
                minacc += xsb * dmin * m as f32 * *xsum.add(blk) as f32;
            }
        }
        vaddvq_f32(facc) - minacc
    }
}
unsafe fn q6k_dot(row: *const u8, xq: *const i8, xs: *const f32, cols: usize) -> f32 {
    unsafe {
        let mut facc = vdupq_n_f32(0.0);
        for sb in 0..cols / 256 {
            let b = row.add(sb * 210);
            let (ql, qh, sc) = (b, b.add(128), b.add(192));
            let d = half_to_f32(u16::from_le_bytes([*b.add(208), *b.add(209)]));
            let mut t = [0i8; 256];
            for h in 0..2 {
                let (qlh, qhh) = (ql.add(h * 64), qh.add(h * 32));
                for l in 0..32 {
                    let (a, a32, ah) = (*qlh.add(l), *qlh.add(l + 32), *qhh.add(l));
                    t[h * 128 + l] = (((a & 0xF) | (((ah >> 0) & 3) << 4)) as i8) - 32;
                    t[h * 128 + l + 32] = (((a32 & 0xF) | (((ah >> 2) & 3) << 4)) as i8) - 32;
                    t[h * 128 + l + 64] = (((a >> 4) | (((ah >> 4) & 3) << 4)) as i8) - 32;
                    t[h * 128 + l + 96] = (((a32 >> 4) | (((ah >> 6) & 3) << 4)) as i8) - 32;
                }
            }
            for s16 in 0..16 {
                let blk = sb * 8 + s16 / 2;
                let sd = sdot(vdupq_n_s32(0), vld1q_s8(t.as_ptr().add(s16 * 16)),
                    vld1q_s8(xq.add(blk * 32 + (s16 % 2) * 16)));
                facc = vfmaq_n_f32(facc, vcvtq_f32_s32(sd), *xs.add(blk) * d * (*sc.add(s16) as i8 as f32));
            }
        }
        vaddvq_f32(facc)
    }
}

// ---- native weight matrix: raw k-quant bytes in the blob ----
#[derive(Clone, Copy)]
struct KMat { ptr: *const u8, ttype: u32, cols: usize, bpr: usize }
unsafe impl Send for KMat {}
unsafe impl Sync for KMat {}
impl KMat {
    #[inline(always)]
    unsafe fn dot(&self, row: usize, xq: *const i8, xs: *const f32, xsum: *const i32) -> f32 {
        unsafe {
            let r = self.ptr.add(row * self.bpr);
            match self.ttype { 12 => q4k_dot(r, xq, xs, xsum, self.cols), 14 => q6k_dot(r, xq, xs, self.cols), _ => 0.0 }
        }
    }
}

fn quant_i8(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    let nb = x.len() / 32;
    let mut q = vec![0i8; x.len()];
    let mut s = vec![0f32; nb];
    let mut sum = vec![0i32; nb];
    for b in 0..nb {
        let blk = &x[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(1e-12f32, |m, &v| m.max(v.abs()));
        let sc = amax / 127.0;
        s[b] = sc;
        let mut acc = 0i32;
        for (i, &v) in blk.iter().enumerate() { let qi = (v / sc).round().clamp(-127.0, 127.0) as i8; q[b * 32 + i] = qi; acc += qi as i32; }
        sum[b] = acc;
    }
    (q, s, sum)
}
fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + cfg().rms_eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}
fn rmsnorm_head(v: &mut [f32], w: &[f32]) {
    let hd = v.len();
    let ms = v.iter().map(|x| x * x).sum::<f32>() / hd as f32;
    let inv = 1.0 / (ms + cfg().rms_eps).sqrt();
    for i in 0..hd { v[i] = v[i] * inv * w[i]; }
}
fn rope(v: &mut [f32], pos: usize) {
    let (hd, base) = (cfg().hd, cfg().rope_base);
    let half = hd / 2;
    for i in 0..half {
        let theta = pos as f32 * base.powf(-(2.0 * i as f32) / hd as f32);
        let (c, s) = (theta.cos(), theta.sin());
        let (a, b) = (v[i], v[half + i]);
        v[i] = a * c - b * s;
        v[half + i] = b * c + a * s;
    }
}

// ---- persistent thread pool (condvar) ----
struct Pool { m: std::sync::Mutex<PoolState>, go: std::sync::Condvar, done: std::sync::Condvar }
struct PoolState { seq: u64, task: [usize; 2], rem: usize }
static POOL: std::sync::OnceLock<&'static Pool> = std::sync::OnceLock::new();
fn pool_init() {
    let sh: &'static Pool = Box::leak(Box::new(Pool {
        m: std::sync::Mutex::new(PoolState { seq: 0, task: [0, 0], rem: 0 }),
        go: std::sync::Condvar::new(), done: std::sync::Condvar::new() }));
    for wid in 0..NT {
        std::thread::spawn(move || {
            let mut seen = 0u64;
            loop {
                let task;
                { let mut st = sh.m.lock().unwrap();
                  while st.seq == seen { st = sh.go.wait(st).unwrap(); }
                  seen = st.seq; task = st.task; }
                let f: &(dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(task) };
                f(wid);
                let mut st = sh.m.lock().unwrap();
                st.rem -= 1;
                if st.rem == 0 { sh.done.notify_one(); }
            }
        });
    }
    POOL.set(sh).ok();
}
fn par(n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    let chunk = (n + NT - 1) / NT;
    let g = move |t: usize| { let a = t * chunk; if a < n { f(a, (a + chunk).min(n)); } };
    let gd: &(dyn Fn(usize) + Sync) = &g;
    let sh = POOL.get().unwrap();
    let fat: [usize; 2] = unsafe { std::mem::transmute(gd as *const (dyn Fn(usize) + Sync)) };
    let mut st = sh.m.lock().unwrap();
    st.task = fat; st.seq += 1; st.rem = NT;
    sh.go.notify_all();
    while st.rem > 0 { st = sh.done.wait(st).unwrap(); }
}

#[derive(Clone, Copy)] struct P(*mut f32);
unsafe impl Send for P {} unsafe impl Sync for P {}
impl P { fn g(self) -> *mut f32 { self.0 } }

struct Layer {
    attn_norm: Vec<f32>, ffn_norm: Vec<f32>, q_norm: Vec<f32>, k_norm: Vec<f32>,
    wq: KMat, wk: KMat, wv: KMat, wo: KMat,
    gate_inp: Vec<f32>, gate: KMat, up: KMat, down: KMat,
}

struct Raw { blob: Vec<u8>, idx: HashMap<String, (u32, usize, usize)> }
impl Raw {
    fn f32v(&self, n: &str) -> Vec<f32> {
        let &(t, off, ne) = &self.idx[n];
        assert_eq!(t, 0, "{n}");
        self.blob[off..off + ne * 4].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn km(&self, n: &str, cols: usize) -> KMat {
        let &(t, off, _) = &self.idx[n];
        let bpr = match t { 12 => cols / 256 * 144, 14 => cols / 256 * 210, _ => panic!("{n} type {t}") };
        KMat { ptr: unsafe { self.blob.as_ptr().add(off) }, ttype: t, cols, bpr }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir, blobp) = (&args[1], &args[2]);
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Write a haiku about CPUs.".into());
    load_cfg(dir);
    let c = *cfg();

    let tb = fs::read(format!("{dir}/tokens.bin")).unwrap();
    let ntok = u32::from_le_bytes(tb[0..4].try_into().unwrap()) as usize;
    let mut toks: Vec<&[u8]> = Vec::with_capacity(ntok);
    let mut p = 4;
    for _ in 0..ntok { let l = u16::from_le_bytes(tb[p..p + 2].try_into().unwrap()) as usize; toks.push(&tb[p + 2..p + 2 + l]); p += 2 + l; }
    let mut lookup: HashMap<&[u8], u32> = HashMap::new();
    for (i, t) in toks.iter().enumerate() { if !t.is_empty() && !t.starts_with(b"<|") { lookup.entry(*t).or_insert(i as u32); } }
    let sp: HashMap<&[u8], u32> = toks.iter().enumerate().filter(|(_, t)| t.starts_with(b"<|")).map(|(i, t)| (*t, i as u32)).collect();
    let encode = |s: &str| -> Vec<u32> {
        let b = s.as_bytes(); let mut ids = Vec::new(); let mut i = 0;
        while i < b.len() { let mut l = (b.len() - i).min(48);
            loop { if let Some(&id) = lookup.get(&b[i..i + l]) { ids.push(id); i += l; break; } l -= 1; assert!(l > 0); } }
        ids
    };
    let tok = |t: &str| sp[t.as_bytes()];

    eprintln!("loading blob (native 4-bit, no dequant at load)...");
    let t0 = Instant::now();
    let blob = fs::read(blobp).unwrap();
    let mut idx = HashMap::new();
    let man = fs::read_to_string(format!("{dir}/manifest.txt")).unwrap();
    let mut lines = man.lines();
    let ds: usize = lines.next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    for l in lines { let f: Vec<&str> = l.split_whitespace().collect();
        idx.insert(f[0].to_string(), (f[1].parse().unwrap(), ds + f[2].parse::<usize>().unwrap(), f[3].parse().unwrap())); }
    let raw = Raw { blob, idx };
    let (d, nh, nkv, hd, ff) = (c.d, c.nh, c.nkv, c.hd, c.ffexp);
    let layers: Vec<Layer> = (0..c.nl).map(|i| { let pfx = format!("blk.{i}"); Layer {
        attn_norm: raw.f32v(&format!("{pfx}.attn_norm.weight")),
        ffn_norm: raw.f32v(&format!("{pfx}.ffn_norm.weight")),
        q_norm: raw.f32v(&format!("{pfx}.attn_q_norm.weight")),
        k_norm: raw.f32v(&format!("{pfx}.attn_k_norm.weight")),
        wq: raw.km(&format!("{pfx}.attn_q.weight"), d),
        wk: raw.km(&format!("{pfx}.attn_k.weight"), d),
        wv: raw.km(&format!("{pfx}.attn_v.weight"), d),
        wo: raw.km(&format!("{pfx}.attn_output.weight"), nh * hd),
        gate_inp: raw.f32v(&format!("{pfx}.ffn_gate_inp.weight")),
        gate: raw.km(&format!("{pfx}.ffn_gate_exps.weight"), d),
        up: raw.km(&format!("{pfx}.ffn_up_exps.weight"), d),
        down: raw.km(&format!("{pfx}.ffn_down_exps.weight"), ff),
    }}).collect();
    let tok_embd = raw.idx["token_embd.weight"];
    let out_norm = raw.f32v("output_norm.weight");
    let head = raw.km("output.weight", d);
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    pool_init();

    let blob_ptr = raw.blob.as_ptr();
    let embed = |t: u32| -> Vec<f32> {
        let (_, off, _) = tok_embd;
        let bpr = d / 256 * 144;
        let mut out = vec![0f32; d];
        let mut tmp = [0f32; 256];
        unsafe {
            let row = blob_ptr.add(off + t as usize * bpr);
            for sb in 0..d / 256 { // dequant Q4_K embedding row
                let b = row.add(sb * 144);
                let dd = half_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                let dmin = half_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
                let scales = std::slice::from_raw_parts(b.add(4), 12);
                let qs = b.add(16);
                let (mut y, mut is, mut qo) = (0usize, 0usize, 0usize);
                while y < 256 {
                    let (s1, m1) = scale_min_k4(is, scales);
                    let (s2, m2) = scale_min_k4(is + 1, scales);
                    for l in 0..32 { tmp[y + l] = dd * s1 as f32 * (*qs.add(qo + l) & 0xF) as f32 - dmin * m1 as f32; }
                    for l in 0..32 { tmp[y + 32 + l] = dd * s2 as f32 * (*qs.add(qo + l) >> 4) as f32 - dmin * m2 as f32; }
                    y += 64; is += 2; qo += 32;
                }
                out[sb * 256..sb * 256 + 256].copy_from_slice(&tmp);
            }
        }
        out
    };

    let mut kc = vec![vec![0f32; MAXSEQ * nkv * hd]; c.nl];
    let mut vc = vec![vec![0f32; MAXSEQ * nkv * hd]; c.nl];

    let mut forward = |t: u32, pos: usize, kc: &mut Vec<Vec<f32>>, vc: &mut Vec<Vec<f32>>| -> u32 {
        let mut x = embed(t);
        for (il, ly) in layers.iter().enumerate() {
            let xn = rmsnorm(&x, &ly.attn_norm);
            let (xq, xs, xsm) = quant_i8(&xn);
            let mut q = vec![0f32; nh * hd];
            let mut k = vec![0f32; nkv * hd];
            let mut v = vec![0f32; nkv * hd];
            {
                let (qp, kp, vp) = (P(q.as_mut_ptr()), P(k.as_mut_ptr()), P(v.as_mut_ptr()));
                let (xq, xs, xsm) = (&xq, &xs, &xsm);
                par(nh * hd + 2 * nkv * hd, &|a, b| {
                    for r in a..b { unsafe {
                        if r < nh * hd { *qp.g().add(r) = ly.wq.dot(r, xq.as_ptr(), xs.as_ptr(), xsm.as_ptr()); }
                        else if r < nh * hd + nkv * hd { let rr = r - nh * hd; *kp.g().add(rr) = ly.wk.dot(rr, xq.as_ptr(), xs.as_ptr(), xsm.as_ptr()); }
                        else { let rr = r - nh * hd - nkv * hd; *vp.g().add(rr) = ly.wv.dot(rr, xq.as_ptr(), xs.as_ptr(), xsm.as_ptr()); }
                    } }
                });
            }
            for h in 0..nh { rmsnorm_head(&mut q[h * hd..(h + 1) * hd], &ly.q_norm); rope(&mut q[h * hd..(h + 1) * hd], pos); }
            for h in 0..nkv { rmsnorm_head(&mut k[h * hd..(h + 1) * hd], &ly.k_norm); rope(&mut k[h * hd..(h + 1) * hd], pos); }
            kc[il][pos * nkv * hd..(pos + 1) * nkv * hd].copy_from_slice(&k);
            vc[il][pos * nkv * hd..(pos + 1) * nkv * hd].copy_from_slice(&v);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut ao = vec![0f32; nh * hd];
            {
                let (kcl, vcl, qref) = (&kc[il], &vc[il], &q);
                let out = P(ao.as_mut_ptr());
                par(nh, &|a, b| {
                    for h in a..b {
                        let kvh = h / (nh / nkv);
                        let mut sc: Vec<f32> = (0..=pos).map(|t| {
                            let mut acc = 0.0; for j in 0..hd { acc += kcl[(t * nkv + kvh) * hd + j] * qref[h * hd + j]; } acc * scale
                        }).collect();
                        let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                        let mut den = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); den += *s; }
                        for j in 0..hd { let mut acc = 0.0; for t in 0..=pos { acc += sc[t] * vcl[(t * nkv + kvh) * hd + j]; }
                            unsafe { *out.g().add(h * hd + j) = acc / den; } }
                    }
                });
            }
            let (aq, asc, asm) = quant_i8(&ao);
            { let out = P(x.as_mut_ptr()); let (aq, asc, asm) = (&aq, &asc, &asm);
              par(d, &|a, b| { for r in a..b { unsafe { *out.g().add(r) += ly.wo.dot(r, aq.as_ptr(), asc.as_ptr(), asm.as_ptr()); } } }); }

            // MoE
            let xn2 = rmsnorm(&x, &ly.ffn_norm);
            let logits: Vec<f32> = (0..c.ne).map(|e| { let mut a = 0.0; for j in 0..d { a += ly.gate_inp[e * d + j] * xn2[j]; } a }).collect();
            let lmax = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - lmax).exp()).collect();
            let sm: f32 = exps.iter().sum();
            let mut order: Vec<usize> = (0..c.ne).collect();
            order.sort_by(|&a, &b| exps[b].partial_cmp(&exps[a]).unwrap());
            let top: Vec<usize> = order[..c.topk].to_vec();
            let wsum: f32 = top.iter().map(|&e| exps[e]).sum();
            let wts: Vec<f32> = top.iter().map(|&e| exps[e] / sm / (wsum / sm)).collect();

            let (x2q, x2s, x2m) = quant_i8(&xn2);
            // gate+up batched over all active experts: topk*ff rows
            let mut hbuf = vec![0f32; c.topk * ff];
            {
                let out = P(hbuf.as_mut_ptr());
                let (x2q, x2s, x2m, top) = (&x2q, &x2s, &x2m, &top);
                par(c.topk * ff, &|a, b| {
                    for it in a..b {
                        let (ei, r) = (it / ff, it % ff);
                        let gr = top[ei] * ff + r;
                        let g = unsafe { ly.gate.dot(gr, x2q.as_ptr(), x2s.as_ptr(), x2m.as_ptr()) };
                        let u = unsafe { ly.up.dot(gr, x2q.as_ptr(), x2s.as_ptr(), x2m.as_ptr()) };
                        unsafe { *out.g().add(it) = (g / (1.0 + (-g).exp())) * u; }
                    }
                });
            }
            // down batched: topk*d rows, weighted accumulate
            let mut ffn = vec![0f32; c.topk * d];
            {
                let out = P(ffn.as_mut_ptr());
                let top = &top;
                // quantize each expert's h once
                let mut hq = vec![0i8; c.topk * ff];
                let mut hs = vec![0f32; c.topk * (ff / 32)];
                let mut hm = vec![0i32; c.topk * (ff / 32)];
                for ei in 0..c.topk { let (a, b, cc) = quant_i8(&hbuf[ei * ff..(ei + 1) * ff]);
                    hq[ei * ff..(ei + 1) * ff].copy_from_slice(&a);
                    hs[ei * (ff / 32)..(ei + 1) * (ff / 32)].copy_from_slice(&b);
                    hm[ei * (ff / 32)..(ei + 1) * (ff / 32)].copy_from_slice(&cc); }
                let (hq, hs, hm) = (&hq, &hs, &hm);
                par(c.topk * d, &|a, b| {
                    for it in a..b {
                        let (ei, r) = (it / d, it % d);
                        let dr = top[ei] * d + r;
                        unsafe { *out.g().add(it) = ly.down.dot(dr, hq.as_ptr().add(ei * ff), hs.as_ptr().add(ei * (ff / 32)), hm.as_ptr().add(ei * (ff / 32))); }
                    }
                });
            }
            for ei in 0..c.topk { let w = wts[ei]; for r in 0..d { x[r] += w * ffn[ei * d + r]; } }
        }
        let xn = rmsnorm(&x, &out_norm);
        let (xq, xs, xsm) = quant_i8(&xn);
        let mut best = vec![(f32::MIN, 0u32); NT];
        let bp = best.as_mut_ptr() as usize;
        let nv = c.nvocab;
        let (xq, xs, xsm) = (&xq, &xs, &xsm);
        par(nv, &|a, b| {
            let mut m = (f32::MIN, 0u32);
            for r in a..b { let l = unsafe { head.dot(r, xq.as_ptr(), xs.as_ptr(), xsm.as_ptr()) }; if l > m.0 { m = (l, r as u32); } }
            let wid = a / ((nv + NT - 1) / NT);
            unsafe { *(bp as *mut (f32, u32)).add(wid) = m; }
        });
        best.iter().cloned().fold((f32::MIN, 0u32), |m, v| if v.0 > m.0 { v } else { m }).1
    };

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
    let mut n = 0;
    for _ in 0..ngen {
        if next == im_end || next as usize >= ntok { break; }
        print!("{}", String::from_utf8_lossy(toks[next as usize]));
        use std::io::Write; std::io::stdout().flush().ok();
        next = forward(next, pos, &mut kc, &mut vc); pos += 1; n += 1;
    }
    let dt = t2.elapsed().as_secs_f64();
    println!("\n---\ndecode: {n} tokens in {dt:.2}s = {:.1} tok/s", n as f64 / dt);

}
