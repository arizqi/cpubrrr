//! gpt-oss engine v2 — worker-driven execution model (A-phase port of engine.rs
//! onto the engine_qwen2 skeleton). 12 persistent workers run the ENTIRE forward
//! pass themselves, syncing at sense-reversing yielding barriers (~1us) instead of
//! fork-join pool dispatch. Workers park only ONCE per token. All serial glue
//! (rmsnorm, quant, router, act) is parallelized too -> no idle cores.
//! Kernels (dot_q8_i8 / dot4_mx_i8 / repack_mx) and all forward math (YaRN RoPE,
//! attention sinks, sliding window, clamped SwiGLU-OAI) are engine.rs's, verbatim.
//! Usage: engine_gpt2 <data-dir> <blob> [prompt|--serve] [ngen]

use std::arch::aarch64::*;
use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const MAXSEQ: usize = 4096;
const NP: usize = 12;        // P-core workers (weight 5)
const EW: u64 = 2;           // E-core worker weight (P weight = 5)
/// total workers; >12 adds E-core workers at UTILITY QoS with weight EW/5
#[allow(non_snake_case)]
fn NT() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("CPBRR_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(12).clamp(1, 32))
}
#[inline(always)]
fn wunits(wid: usize) -> (u64, u64) { // (start_units, width_units)
    if wid < NP { (wid as u64 * 5, 5) } else { (NP as u64 * 5 + (wid - NP) as u64 * EW, EW) }
}
#[inline(always)]
fn total_units() -> u64 {
    static T: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *T.get_or_init(|| { let n = NT(); if n <= NP { n as u64 * 5 } else { NP as u64 * 5 + (n - NP) as u64 * EW } })
}
fn dbg_on() -> bool { static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *D.get_or_init(|| std::env::var("CPBRR_DBG").is_ok()) }
fn dump_on() -> bool { static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *D.get_or_init(|| std::env::var("DUMP").is_ok()) }
unsafe fn dsum(p: *const f32, n: usize) -> f32 { let mut s = 0.0; for i in 0..n { s += unsafe { *p.add(i) }; } s }
static TM: [AtomicU64; 5] = [AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0)];

#[derive(Clone, Copy)]
struct Cfg { d: usize, nh: usize, nkv: usize, hd: usize, nl: usize, ne: usize, topk: usize,
    blocks: usize, nvocab: usize, swa: usize, rope_base: f32, rms_eps: f32 }
static CFG: std::sync::OnceLock<Cfg> = std::sync::OnceLock::new();
#[inline(always)]
fn cfg() -> &'static Cfg { CFG.get().unwrap() }
fn load_cfg(dir: &str) {
    let mut m = HashMap::new();
    for l in fs::read_to_string(format!("{dir}/config.txt")).unwrap().lines() {
        let mut it = l.split_whitespace();
        if let (Some(k), Some(v)) = (it.next(), it.next()) { m.insert(k.to_string(), v.to_string()); }
    }
    let gi = |k: &str| m.get(k).unwrap_or_else(|| panic!("config missing {k}")).parse::<usize>().unwrap();
    let gf = |k: &str| m.get(k).unwrap().parse::<f32>().unwrap();
    let d = gi("n_embd");
    assert_eq!(gi("n_ff_exp"), d, "engine assumes expert ff dim == d (as engine.rs does)");
    assert_eq!(gi("head_dim"), 64, "attention kernels assume hd=64");
    CFG.set(Cfg { d, nh: gi("n_head"), nkv: gi("n_head_kv"), hd: gi("head_dim"),
        nl: gi("n_layer"), ne: gi("n_expert"), topk: gi("n_expert_used"),
        blocks: d / 32, nvocab: gi("n_vocab"),
        swa: m.get("sliding_window").map(|v| v.parse().unwrap()).unwrap_or(1 << 30),
        rope_base: gf("rope_freq_base"), rms_eps: gf("rms_eps") }).ok();
}

// ---- sampling controls (set per request in --serve mode; greedy when TEMP_MILLI=0) ----
static TEMP_MILLI: AtomicU32 = AtomicU32::new(0);
static TOPK_S: AtomicUsize = AtomicUsize::new(40);
static RNG_STATE: AtomicU64 = AtomicU64::new(0x9E3779B97F4A7C15);

fn rng_next() -> u64 {
    let mut x = RNG_STATE.load(Ordering::Relaxed);
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    RNG_STATE.store(x, Ordering::Relaxed);
    x
}

/// Sample from logits with temperature + top-k (k<=64). Greedy if temp==0.
fn sample_logits(logits: &[f32]) -> u32 {
    let tm = TEMP_MILLI.load(Ordering::Relaxed);
    if tm == 0 {
        let mut m = (f32::MIN, 0u32);
        for (i, &l) in logits.iter().enumerate() { if l > m.0 { m = (l, i as u32); } }
        return m.1;
    }
    let temp = tm as f32 / 1000.0;
    let k = TOPK_S.load(Ordering::Relaxed).clamp(1, 64);
    let mut top: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (i, &l) in logits.iter().enumerate() {
        if top.len() < k || l > top.last().unwrap().0 {
            let pos = top.partition_point(|&(v, _)| v > l);
            top.insert(pos, (l, i as u32));
            if top.len() > k { top.pop(); }
        }
    }
    let mx = top[0].0;
    let mut probs: Vec<f32> = top.iter().map(|&(l, _)| ((l - mx) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() { *p /= sum; }
    let r = (rng_next() >> 11) as f32 / (1u64 << 53) as f32;
    let mut acc = 0.0f32;
    for (p, &(_, idx)) in probs.iter().zip(top.iter()) {
        acc += p;
        if r <= acc { return idx; }
    }
    top.last().unwrap().1
}

const KVI: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

// ---------------- kernels (verbatim from engine.rs) ----------------
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        let mut r = acc;
        std::arch::asm!("sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
            r = inout(vreg) r, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
        r
    }
}

/// Q8 weight row · Q8 activation: int sdot per 32-block, scales multiplied in.
/// 2-block unroll with independent fp accumulators: the single-chain version is
/// fma-latency-bound (~4cyc x blocks), not bandwidth-bound. Sum order differs
/// from engine.rs (even+odd chains) — quality re-verified via GSM8K parity.
unsafe fn dot_q8_i8(wq: *const i8, ws: *const f32, xq: *const i8, xsb: *const f32, cols: usize) -> f32 {
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let nb = cols / 32;
        let mut b = 0;
        while b + 2 <= nb {
            let d0 = sdot(sdot(vdupq_n_s32(0), vld1q_s8(wq.add(b * 32)), vld1q_s8(xq.add(b * 32))),
                          vld1q_s8(wq.add(b * 32 + 16)), vld1q_s8(xq.add(b * 32 + 16)));
            let d1 = sdot(sdot(vdupq_n_s32(0), vld1q_s8(wq.add(b * 32 + 32)), vld1q_s8(xq.add(b * 32 + 32))),
                          vld1q_s8(wq.add(b * 32 + 48)), vld1q_s8(xq.add(b * 32 + 48)));
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(d0), *ws.add(b) * *xsb.add(b));
            acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(d1), *ws.add(b + 1) * *xsb.add(b + 1));
            b += 2;
        }
        if b < nb {
            let d = sdot(sdot(vdupq_n_s32(0), vld1q_s8(wq.add(b * 32)), vld1q_s8(xq.add(b * 32))),
                         vld1q_s8(wq.add(b * 32 + 16)), vld1q_s8(xq.add(b * 32 + 16)));
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(d), *ws.add(b) * *xsb.add(b));
        }
        vaddvq_f32(vaddq_f32(acc0, acc1))
    }
}

/// 4 rows x cfg().d against block-quantized int8 x, quad-interleaved MXFP4.
unsafe fn dot4_mx_i8(nib: *const u8, sc: *const u8, xq: *const i8, xsb: *const f32) -> [f32; 4] {
    unsafe {
        let tbl = exp_tbl().as_ptr();
        let kv = vld1q_s8(KVI.as_ptr());
        let mask = vdupq_n_u8(0x0F);
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut np = nib;
        let mut sp = sc;
        for b in 0..cfg().blocks {
            let x0 = vld1q_s8(xq.add(b * 32));
            let x1 = vld1q_s8(xq.add(b * 32 + 16));
            let se = [*tbl.add(*sp as usize), *tbl.add(*sp.add(1) as usize), *tbl.add(*sp.add(2) as usize), *tbl.add(*sp.add(3) as usize)];
            let sv = vmulq_n_f32(vld1q_f32(se.as_ptr()), *xsb.add(b));
            let z = vdupq_n_s32(0);
            let w0 = vld1q_u8(np);
            let w1 = vld1q_u8(np.add(16));
            let w2 = vld1q_u8(np.add(32));
            let w3 = vld1q_u8(np.add(48));
            let t0 = sdot(sdot(z, vqtbl1q_s8(kv, vandq_u8(w0, mask)), x0), vqtbl1q_s8(kv, vshrq_n_u8::<4>(w0)), x1);
            let t1 = sdot(sdot(z, vqtbl1q_s8(kv, vandq_u8(w1, mask)), x0), vqtbl1q_s8(kv, vshrq_n_u8::<4>(w1)), x1);
            let t2 = sdot(sdot(z, vqtbl1q_s8(kv, vandq_u8(w2, mask)), x0), vqtbl1q_s8(kv, vshrq_n_u8::<4>(w2)), x1);
            let t3 = sdot(sdot(z, vqtbl1q_s8(kv, vandq_u8(w3, mask)), x0), vqtbl1q_s8(kv, vshrq_n_u8::<4>(w3)), x1);
            a0 = vfmaq_laneq_f32::<0>(a0, vcvtq_f32_s32(t0), sv);
            a1 = vfmaq_laneq_f32::<1>(a1, vcvtq_f32_s32(t1), sv);
            a2 = vfmaq_laneq_f32::<2>(a2, vcvtq_f32_s32(t2), sv);
            a3 = vfmaq_laneq_f32::<3>(a3, vcvtq_f32_s32(t3), sv);
            np = np.add(64);
            sp = sp.add(4);
        }
        [vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3)]
    }
}

/// Quad-interleaved MXFP4 (E6 lesson 8 layout): one sequential stream per thread.
/// nib: [quad][block][row 0..4][16B], scale: [quad][block][4] (2^(e-128) prefolded).
struct QuadMx { nib: Vec<u8>, exps: Vec<u8> }
/// 2^(e-128) per raw exponent byte, prefolded with the same powi engine.rs uses
/// (u8 exps in the stream instead of f32 scales: 20% less expert traffic).
static EXP_TBL: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
fn exp_tbl() -> &'static [f32; 256] {
    EXP_TBL.get_or_init(|| { let mut t = [0f32; 256]; for e in 0..256 { t[e] = 2f32.powi(e as i32 - 128); } t })
}
fn repack_mx(raw: *const u8, rows: usize) -> QuadMx {
    let mut nib = vec![0u8; rows * cfg().blocks * 16];
    let mut exps = vec![0u8; rows * cfg().blocks];
    {
        let np = SendMutU8(nib.as_mut_ptr());
        let sp = SendMutU8(exps.as_mut_ptr());
        let rp = SendU8(raw);
        par_load(rows / 4, |a, b| {
            for q in a..b {
                for blk in 0..cfg().blocks {
                    for r in 0..4 {
                        unsafe {
                            let src = rp.get().add(((q * 4 + r) * cfg().blocks + blk) * 17);
                            *sp.get().add((q * cfg().blocks + blk) * 4 + r) = *src;
                            std::ptr::copy_nonoverlapping(src.add(1), np.get().add((q * cfg().blocks + blk) * 64 + r * 16), 16);
                        }
                    }
                }
            }
        });
    }
    QuadMx { nib, exps }
}

/// Q8_0 weight matrix converted from bf16 at load (halves traffic vs bf16).
struct Q8Mat { q: Vec<i8>, s: Vec<f32>, cols: usize }
impl Q8Mat {
    fn from_bf16(w: *const u16, rows: usize, cols: usize) -> Self {
        let nb = cols / 32;
        let mut q = vec![0i8; rows * cols];
        let mut s = vec![0f32; rows * nb];
        {
            let qp = SendMutI8(q.as_mut_ptr());
            let sp = SendMutF32(s.as_mut_ptr());
            let wp = SendU16(w);
            par_load(rows, |a, b| {
                for r in a..b {
                    for blk in 0..nb {
                        let mut vals = [0f32; 32];
                        let mut amax = 1e-12f32;
                        for i in 0..32 {
                            let v = f32::from_bits((unsafe { *wp.get().add(r * cols + blk * 32 + i) } as u32) << 16);
                            vals[i] = v;
                            amax = amax.max(v.abs());
                        }
                        let sc = amax / 127.0;
                        unsafe { *sp.get().add(r * nb + blk) = sc };
                        for i in 0..32 {
                            unsafe { *qp.get().add(r * cols + blk * 32 + i) = (vals[i] / sc).round().clamp(-127.0, 127.0) as i8 };
                        }
                    }
                }
            });
        }
        Q8Mat { q, s, cols }
    }
    #[inline(always)]
    unsafe fn dot(&self, row: usize, xq: *const i8, xsb: *const f32) -> f32 {
        unsafe {
            dot_q8_i8(self.q.as_ptr().add(row * self.cols), self.s.as_ptr().add(row * (self.cols / 32)), xq, xsb, self.cols)
        }
    }
}

#[derive(Clone, Copy)] struct SendMutU8(*mut u8); unsafe impl Send for SendMutU8 {} unsafe impl Sync for SendMutU8 {}
impl SendMutU8 { fn get(self) -> *mut u8 { self.0 } }
#[derive(Clone, Copy)] struct SendMutI8(*mut i8); unsafe impl Send for SendMutI8 {} unsafe impl Sync for SendMutI8 {}
impl SendMutI8 { fn get(self) -> *mut i8 { self.0 } }
#[derive(Clone, Copy)] struct SendMutF32(*mut f32); unsafe impl Send for SendMutF32 {} unsafe impl Sync for SendMutF32 {}
impl SendMutF32 { fn get(self) -> *mut f32 { self.0 } }
#[derive(Clone, Copy)] struct SendU16(*const u16); unsafe impl Send for SendU16 {} unsafe impl Sync for SendU16 {}
impl SendU16 { fn get(self) -> *const u16 { self.0 } }
#[derive(Clone, Copy)] struct SendU8(*const u8); unsafe impl Send for SendU8 {} unsafe impl Sync for SendU8 {}
impl SendU8 { fn get(self) -> *const u8 { self.0 } }

/// load-time parallel helper (scoped threads; worker pool not yet running)
fn par_load<F: Fn(usize, usize) + Sync>(n: usize, f: F) {
    let nt = NT().max(12);
    let chunk = (n + nt - 1) / nt;
    std::thread::scope(|s| {
        for t in 0..nt {
            let a = t * chunk;
            if a >= n { break; }
            let b = (a + chunk).min(n);
            let f = &f;
            s.spawn(move || f(a, b));
        }
    });
}

/// YaRN RoPE (freq_base 150000, factor 32, orig ctx 4096, beta 32/1, ext_factor 1).
/// cos/sin (mscale folded) precomputed per (pos, i) at startup with the exact
/// engine.rs expressions -> bit-identical rotation, no powf/cos/sin per token
/// (engine.rs burns ~0.5 ms/tok on 55k powf calls in its rope stage).
static ROPE_TAB: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
fn rope_init() {
    let half = cfg().hd / 2;
    let mut tab = vec![0f32; MAXSEQ * half * 2];
    let mut v = vec![0f32; cfg().hd];
    for pos in 0..MAXSEQ {
        // rotate the (1,0) basis with the exact slow path: v[i]=cos*mscale, v[half+i]=sin*mscale
        for i in 0..half { v[i] = 1.0; v[half + i] = 0.0; }
        rope_yarn_slow(v.as_mut_ptr(), pos);
        for i in 0..half {
            tab[(pos * half + i) * 2] = v[i];
            tab[(pos * half + i) * 2 + 1] = v[half + i];
        }
    }
    ROPE_TAB.set(tab).ok();
}
fn rope_yarn_slow(v: *mut f32, pos: usize) {
    unsafe {
        let half = cfg().hd / 2;
        let base = cfg().rope_base;
        let fd = |beta: f32| cfg().hd as f32 * (4096.0 / (beta * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
        let low = fd(32.0).floor().max(0.0);
        let high = fd(1.0).ceil().min(cfg().hd as f32 - 1.0);
        let mscale = 1.0 + 0.1 * 32f32.ln();
        for i in 0..half {
            let theta_ex = pos as f32 * base.powf(-(2.0 * i as f32) / cfg().hd as f32);
            let theta_in = theta_ex / 32.0;
            let y = (i as f32 - low) / (high - low).max(0.001);
            let mix = 1.0 - y.clamp(0.0, 1.0);
            let theta = theta_in * (1.0 - mix) + theta_ex * mix;
            let (c, s) = ((theta.cos() * mscale), (theta.sin() * mscale));
            let a = *v.add(i);
            let b = *v.add(half + i);
            *v.add(i) = a * c - b * s;
            *v.add(half + i) = b * c + a * s;
        }
    }
}

#[inline(always)]
fn rope_yarn(v: *mut f32, pos: usize) {
    unsafe {
        let half = cfg().hd / 2;
        let tab = ROPE_TAB.get().unwrap().as_ptr().add(pos * half * 2);
        for i in 0..half {
            let (c, s) = (*tab.add(i * 2), *tab.add(i * 2 + 1));
            let a = *v.add(i);
            let b = *v.add(half + i);
            *v.add(i) = a * c - b * s;
            *v.add(half + i) = b * c + a * s;
        }
    }
}

// ---------------- per-layer weights ----------------
struct Layer {
    attn_norm: Vec<f32>, ffn_norm: Vec<f32>,
    wq: Q8Mat, wk: Q8Mat, wv: Q8Mat, wo: Q8Mat,
    bq: Vec<f32>, bk: Vec<f32>, bv: Vec<f32>, bo: Vec<f32>,
    sinks: Vec<f32>,
    ginp: Vec<f32>, ginp_b: Vec<f32>,
    gate: QuadMx, up: QuadMx, down: QuadMx,
    gate_b: Vec<f32>, up_b: Vec<f32>, down_b: Vec<f32>,
}

struct Model { blob: memmap2::Mmap, idx: HashMap<String, (u32, usize, usize)> }
impl Model {
    fn f32v(&self, name: &str) -> Vec<f32> {
        let &(t, off, n) = self.idx.get(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(t, 0, "{name}");
        self.blob[off..off + n * 4].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn bf16(&self, name: &str) -> *const u16 {
        let &(t, off, _) = self.idx.get(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(t, 30, "{name}");
        unsafe { self.blob.as_ptr().add(off) as *const u16 }
    }
    fn mxfp4(&self, name: &str) -> *const u8 {
        let &(t, off, _) = self.idx.get(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(t, 39, "{name}");
        unsafe { self.blob.as_ptr().add(off) }
    }
}

/// worker's contiguous chunk of n items (unit-weighted: E-core workers get EW/5 of a P slice)
#[inline(always)]
fn sl(wid: usize, n: usize) -> (usize, usize) {
    let tu = total_units();
    let (su, w) = wunits(wid.min(NT() - 1));
    let a = (n as u64 * su / tu) as usize;
    let b = (n as u64 * (su + w) / tu) as usize;
    (a, b)
}

struct Shared {
    // control
    tseq: AtomicU64, ready: AtomicU64, cur_tok: AtomicU32, cur_pos: AtomicUsize,
    want_logits: AtomicBool, result: AtomicU32,
    // barrier
    bcount: AtomicUsize, bsense: AtomicBool,
    // park (between tokens only)
    m: std::sync::Mutex<()>, cv: std::sync::Condvar,
    dm: std::sync::Mutex<()>, dcv: std::sync::Condvar,
    // model
    layers: &'static [Layer], out_norm: &'static [f32], head: &'static Q8Mat, tok_embd: *const u16,
    // activation buffers
    x: *mut f32, xn: *mut f32, xq: *mut i8, xsb: *mut f32,
    q: *mut f32, k: *mut f32, v: *mut f32, ao: *mut f32,
    aq: *mut i8, asb: *mut f32,
    gu: *mut f32, hq: *mut i8, hsb: *mut f32,
    rlogits: *mut f32, logits: *mut f32,
    kc: *mut f32, vc: *mut f32,   // [nl][MAXSEQ*nkv*hd] flattened
}
unsafe impl Send for Shared {} unsafe impl Sync for Shared {}

#[inline(always)]
fn bar(sh: &Shared, ls: &mut bool) {
    *ls = !*ls;
    if sh.bcount.fetch_add(1, Ordering::AcqRel) == NT() - 1 {
        sh.bcount.store(0, Ordering::Relaxed);
        sh.bsense.store(*ls, Ordering::Release);
    } else {
        let mut n = 0u32;
        while sh.bsense.load(Ordering::Acquire) != *ls {
            n += 1;
            if n & 1023 != 0 { std::hint::spin_loop(); } else { std::thread::yield_now(); }
        }
    }
}

/// Per-32-block int8 quantization of one block (Q8_0-style, matches engine.rs quant_i8).
#[inline(always)]
fn qblock32(src: *const f32, xq: *mut i8, xsb: *mut f32, blk: usize) {
    unsafe {
        let mut amax = 1e-12f32;
        for i in 0..32 { amax = amax.max((*src.add(blk * 32 + i)).abs()); }
        let sc = amax / 127.0;
        *xsb.add(blk) = sc;
        for i in 0..32 { *xq.add(blk * 32 + i) = (*src.add(blk * 32 + i) / sc).round().clamp(-127.0, 127.0) as i8; }
    }
}

/// Redundant full serial sum-of-squares per worker: same fp order as engine.rs's
/// serial rmsnorm -> bit-identical normalization (parallel partials reorder the sum
/// and flip near-tie argmaxes). d=2880 -> ~1us redundant work, same as a barrier.
#[inline(always)]
fn rmsnorm_par(sh: &Shared, wid: usize, ls: &mut bool, x: *mut f32, out: *mut f32, w: *const f32, d: usize) {
    unsafe {
        let mut tot = 0f32; for i in 0..d { let v = *x.add(i); tot += v * v; }
        let inv = 1.0 / (tot / d as f32 + cfg().rms_eps).sqrt();
        let (a, b) = sl(wid, d);
        for i in a..b { *out.add(i) = *x.add(i) * inv * *w.add(i); }
        bar(sh, ls);
    }
}

fn forward_worker(sh: &Shared, wid: usize, ls: &mut bool) {
    let c = *cfg();
    let (d, nh, nkv, hd, nb) = (c.d, c.nh, c.nkv, c.hd, c.blocks);
    let topk = c.topk;
    let dq = d / 4;
    let pos = sh.cur_pos.load(Ordering::Relaxed);
    let tokid = sh.cur_tok.load(Ordering::Relaxed) as usize;
    unsafe {
        // ---- embed (worker slice of d, bf16 -> f32) ----
        {
            let (a, b) = sl(wid, d);
            for j in a..b { *sh.x.add(j) = f32::from_bits((*sh.tok_embd.add(tokid * d + j) as u32) << 16); }
        }
        bar(sh, ls);
        if wid == 0 && dbg_on() { eprintln!("DBG embed sum {:.6}", dsum(sh.x, d)); }

        for il in 0..c.nl {
            let ly = &sh.layers[il];
            let kcl = sh.kc.add(il * MAXSEQ * nkv * hd);
            let vcl = sh.vc.add(il * MAXSEQ * nkv * hd);
            let _ta = if wid==0 { Some(Instant::now()) } else { None };
            // ---- attn rmsnorm + quant ----
            rmsnorm_par(sh, wid, ls, sh.x, sh.xn, ly.attn_norm.as_ptr(), d);
            { let (a, b) = sl(wid, nb); for bl in a..b { qblock32(sh.xn, sh.xq, sh.xsb, bl); } }
            bar(sh, ls);
            // qkv matvec + bias
            { let nr = nh * hd + 2 * nkv * hd; let (a, b) = sl(wid, nr);
              for r in a..b {
                if r < nh * hd { *sh.q.add(r) = ly.wq.dot(r, sh.xq, sh.xsb) + ly.bq[r]; }
                else if r < nh * hd + nkv * hd { let rr = r - nh * hd; *sh.k.add(rr) = ly.wk.dot(rr, sh.xq, sh.xsb) + ly.bk[rr]; }
                else { let rr = r - nh * hd - nkv * hd; *sh.v.add(rr) = ly.wv.dot(rr, sh.xq, sh.xsb) + ly.bv[rr]; }
              } }
            bar(sh, ls);
            if wid == 0 && il == 0 && dbg_on() { eprintln!("DBG L0 qkv q {:.6} k {:.6} v {:.6}", dsum(sh.q, nh*hd), dsum(sh.k, nkv*hd), dsum(sh.v, nkv*hd)); }
            // rope + kv-cache write (parallel over nh+nkv heads; v has no rope)
            { let (a, b) = sl(wid, nh + nkv);
              for h in a..b {
                if h < nh { rope_yarn(sh.q.add(h * hd), pos); }
                else { let hh = h - nh;
                    rope_yarn(sh.k.add(hh * hd), pos);
                    for j in 0..hd { *kcl.add((pos * nkv + hh) * hd + j) = *sh.k.add(hh * hd + j); *vcl.add((pos * nkv + hh) * hd + j) = *sh.v.add(hh * hd + j); }
                }
              } }
            bar(sh, ls);
            // attention (parallel over nh q-heads); sinks in denom, SWA on even layers
            { let scale = 1.0 / (hd as f32).sqrt();
              let start = if (il + 1) % 2 != 0 { pos.saturating_sub(c.swa - 1) } else { 0 };
              let (a, b) = sl(wid, nh);
              for h in a..b {
                let kvh = h / (nh / nkv);
                let mut sc = [0f32; MAXSEQ];
                let qh = sh.q.add(h * hd);
                for t in start..=pos {
                    let kp = kcl.add((t * nkv + kvh) * hd);
                    let mut a0 = vdupq_n_f32(0.0); let mut a1 = vdupq_n_f32(0.0);
                    let mut a2 = vdupq_n_f32(0.0); let mut a3 = vdupq_n_f32(0.0);
                    let mut j = 0;
                    while j < hd {
                        a0 = vfmaq_f32(a0, vld1q_f32(kp.add(j)), vld1q_f32(qh.add(j)));
                        a1 = vfmaq_f32(a1, vld1q_f32(kp.add(j + 4)), vld1q_f32(qh.add(j + 4)));
                        a2 = vfmaq_f32(a2, vld1q_f32(kp.add(j + 8)), vld1q_f32(qh.add(j + 8)));
                        a3 = vfmaq_f32(a3, vld1q_f32(kp.add(j + 12)), vld1q_f32(qh.add(j + 12)));
                        j += 16;
                    }
                    sc[t - start] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * scale;
                }
                let n = pos - start + 1;
                let mx = sc[..n].iter().cloned().fold(ly.sinks[h], f32::max);
                let mut den = (ly.sinks[h] - mx).exp();
                for t in 0..n { sc[t] = (sc[t] - mx).exp(); den += sc[t]; }
                let aop = sh.ao.add(h * hd);
                let mut acc = [vdupq_n_f32(0.0); 16];
                for (ti, t) in (start..=pos).enumerate() {
                    let vp = vcl.add((t * nkv + kvh) * hd); let sct = sc[ti];
                    for u in 0..16 { acc[u] = vfmaq_n_f32(acc[u], vld1q_f32(vp.add(u * 4)), sct); }
                }
                let inv = 1.0 / den;
                for u in 0..16 { vst1q_f32(aop.add(u * 4), vmulq_n_f32(acc[u], inv)); }
                // hd=64 -> this head owns exactly blocks 2h, 2h+1: quantize in place, no extra barrier
                qblock32(sh.ao, sh.aq, sh.asb, 2 * h);
                qblock32(sh.ao, sh.aq, sh.asb, 2 * h + 1);
              } }
            bar(sh, ls);
            // o-proj + bias + residual
            { let (a, b) = sl(wid, d); for r in a..b { *sh.x.add(r) += ly.wo.dot(r, sh.aq, sh.asb) + ly.bo[r]; } }
            bar(sh, ls);
            if wid == 0 && il == 0 && dbg_on() { eprintln!("DBG L0 ao {:.6} x-after-attn {:.6}", dsum(sh.ao, nh*hd), dsum(sh.x, d)); }
            if let Some(t)=_ta { TM[0].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
            let _tr = if wid==0 { Some(Instant::now()) } else { None };
            // ---- MoE FFN ----
            rmsnorm_par(sh, wid, ls, sh.x, sh.xn, ly.ffn_norm.as_ptr(), d);
            { let (a, b) = sl(wid, nb); for bl in a..b { qblock32(sh.xn, sh.xq, sh.xsb, bl); } }
            // router logits (parallel over experts) — reads xn, safe alongside quant
            { let (a, b) = sl(wid, c.ne);
              for e in a..b { let mut acc = ly.ginp_b[e]; for j in 0..d { acc += ly.ginp[e * d + j] * *sh.xn.add(j); } *sh.rlogits.add(e) = acc; } }
            bar(sh, ls);
            // top-k (redundant per worker, cheap; no barrier)
            let logits = std::slice::from_raw_parts(sh.rlogits, c.ne);
            let mut order: Vec<usize> = (0..c.ne).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let lmax = (0..topk).map(|i| logits[order[i]]).fold(f32::MIN, f32::max);
            let mut wsum = 0f32; let mut ex = [0f32; 8];
            for i in 0..topk { ex[i] = (logits[order[i]] - lmax).exp(); wsum += ex[i]; }
            let top = &order[..topk];
            let mut wts = [0f32; 8];
            for i in 0..topk { wts[i] = ex[i] / wsum; }
            if let Some(t)=_tr { TM[4].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
            let _tg = if wid==0 { Some(Instant::now()) } else { None };
            // gate/up (parallel over topk * 2 * dq quads, quad-interleaved MXFP4 + bias)
            { let (a, b) = sl(wid, topk * 2 * dq);
              for it in a..b {
                let ei = it / (2 * dq);
                let e = top[ei];
                let rem = it % (2 * dq);
                let (w, bias, qd, obase) = if rem < dq { (&ly.gate, &ly.gate_b, rem, ei * 2 * d) }
                                           else { (&ly.up, &ly.up_b, rem - dq, ei * 2 * d + d) };
                let qg = e * dq + qd;
                let acc = dot4_mx_i8(w.nib.as_ptr().add(qg * nb * 64), w.exps.as_ptr().add(qg * nb * 4), sh.xq, sh.xsb);
                for i in 0..4 { *sh.gu.add(obase + qd * 4 + i) = acc[i] + bias[e * d + qd * 4 + i]; }
              } }
            bar(sh, ls);
            if let Some(t)=_tg { TM[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
            let _td = if wid==0 { Some(Instant::now()) } else { None };
            // clamped SwiGLU-OAI act + Q8 quant (parallel over topk*nb blocks)
            { const ALPHA: f32 = 1.702; const LIM: f32 = 7.0;
              let (a, b) = sl(wid, topk * nb);
              for it in a..b {
                let (ei, blk) = (it / nb, it % nb);
                let g = sh.gu.add(ei * 2 * d + blk * 32);
                let u = sh.gu.add(ei * 2 * d + d + blk * 32);
                let mut h = [0f32; 32];
                let mut amax = 1e-12f32;
                for k in 0..32 {
                    let xg = (*g.add(k)).min(LIM);
                    let yu = (*u.add(k)).clamp(-LIM, LIM);
                    let v = (xg / (1.0 + (-ALPHA * xg).exp())) * (yu + 1.0);
                    h[k] = v;
                    amax = amax.max(v.abs());
                }
                let sc = amax / 127.0;
                *sh.hsb.add(ei * nb + blk) = sc;
                let qp = sh.hq.add(ei * d + blk * 32);
                for k in 0..32 { *qp.add(k) = (h[k] / sc).round().clamp(-127.0, 127.0) as i8; }
              } }
            bar(sh, ls);
            // down-proj + weighted residual, fused (parallel over dq quads, all topk
            // experts per quad -> accumulate straight into x, no ffn4 pass/barrier;
            // per-element ei order matches engine.rs's serial accumulate exactly)
            { let (a, b) = sl(wid, dq);
              for qd in a..b {
                let mut s = [0f32; 4];
                for i in 0..4 { s[i] = *sh.x.add(qd * 4 + i); }
                for ei in 0..topk {
                    let e = top[ei];
                    let qg = e * dq + qd;
                    let acc = dot4_mx_i8(ly.down.nib.as_ptr().add(qg * nb * 64), ly.down.exps.as_ptr().add(qg * nb * 4),
                                         sh.hq.add(ei * d), sh.hsb.add(ei * nb));
                    for i in 0..4 { s[i] += wts[ei] * (acc[i] + ly.down_b[e * d + qd * 4 + i]); }
                }
                for i in 0..4 { *sh.x.add(qd * 4 + i) = s[i]; }
              } }
            bar(sh, ls);
            if wid == 0 && dbg_on() { eprintln!("DBG L{il} top {:?} wts {:?} x-after-ffn {:.6}", &top[..], &wts[..topk], dsum(sh.x, d)); }
            if wid == 0 && pos <= 2 && dump_on() {
                let n = (0..d).map(|i| (*sh.x.add(i) as f64).powi(2)).sum::<f64>().sqrt();
                eprintln!("p{pos} layer {il:2} |x| {n:10.4}");
            }
            if let Some(t)=_td { TM[2].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
        }
        if !sh.want_logits.load(Ordering::Relaxed) { return; }
        let _th = if wid==0 { Some(Instant::now()) } else { None };
        // ---- head ----
        rmsnorm_par(sh, wid, ls, sh.x, sh.xn, sh.out_norm.as_ptr(), d);
        { let (a, b) = sl(wid, nb); for bl in a..b { qblock32(sh.xn, sh.xq, sh.xsb, bl); } }
        bar(sh, ls);
        { let (a, b) = sl(wid, c.nvocab);
          for r in a..b { *sh.logits.add(r) = sh.head.dot(r, sh.xq, sh.xsb); } }
        bar(sh, ls);
        if wid == 0 {
            let logits = std::slice::from_raw_parts(sh.logits, c.nvocab);
            let chosen = sample_logits(logits);
            if dbg_on() { eprintln!("DBG head chosen {} logit {:.4}", chosen, logits[chosen as usize]); }
            sh.result.store(chosen, Ordering::Relaxed);
        }
        if let Some(t)=_th { TM[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir, blobp) = (&args[1], &args[2]);
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Why is the sky blue?".into());
    load_cfg(dir);
    rope_init();
    let c = *cfg();

    // tokenizer
    let tb = fs::read(format!("{dir}/tokens.bin")).unwrap();
    let ntok = u32::from_le_bytes(tb[0..4].try_into().unwrap()) as usize;
    let mut toks: Vec<&[u8]> = Vec::with_capacity(ntok);
    let mut p = 4;
    for _ in 0..ntok { let l = u16::from_le_bytes(tb[p..p + 2].try_into().unwrap()) as usize; toks.push(&tb[p + 2..p + 2 + l]); p += 2 + l; }
    let mut lookup: HashMap<&[u8], u32> = HashMap::new();
    for (i, t) in toks.iter().enumerate() { if !t.is_empty() && !t.starts_with(b"<|") { lookup.entry(*t).or_insert(i as u32); } }
    let encode = |s: &str| -> Vec<u32> {
        let b = s.as_bytes(); let mut ids = Vec::new(); let mut i = 0;
        while i < b.len() { let mut l = (b.len() - i).min(32);
            loop { if let Some(&id) = lookup.get(&b[i..i + l]) { ids.push(id); i += l; break; } l -= 1; assert!(l > 0, "unencodable byte {}", b[i]); } }
        ids
    };

    // model
    eprintln!("mmapping model blob...");
    let t0 = Instant::now();
    let blob = unsafe { memmap2::Mmap::map(&fs::File::open(blobp).unwrap()).unwrap() };
    let mut idx = HashMap::new();
    let man = fs::read_to_string(format!("{dir}/manifest.txt")).unwrap();
    let mut lines = man.lines();
    let ds: usize = lines.next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    for l in lines { let f: Vec<&str> = l.split_whitespace().collect();
        idx.insert(f[0].to_string(), (f[1].parse().unwrap(), ds + f[2].parse::<usize>().unwrap(), f[3].parse().unwrap())); }
    let m = Model { blob, idx };
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    eprintln!("quantizing attention + head to Q8, repacking MXFP4...");
    let tq = Instant::now();
    let layers: Vec<Layer> = (0..c.nl).map(|i| Layer {
        attn_norm: m.f32v(&format!("blk.{i}.attn_norm.weight")),
        ffn_norm: m.f32v(&format!("blk.{i}.ffn_norm.weight")),
        wq: Q8Mat::from_bf16(m.bf16(&format!("blk.{i}.attn_q.weight")), c.nh * c.hd, c.d),
        wk: Q8Mat::from_bf16(m.bf16(&format!("blk.{i}.attn_k.weight")), c.nkv * c.hd, c.d),
        wv: Q8Mat::from_bf16(m.bf16(&format!("blk.{i}.attn_v.weight")), c.nkv * c.hd, c.d),
        wo: Q8Mat::from_bf16(m.bf16(&format!("blk.{i}.attn_out.weight")), c.d, c.nh * c.hd),
        bq: m.f32v(&format!("blk.{i}.attn_q.bias")),
        bk: m.f32v(&format!("blk.{i}.attn_k.bias")),
        bv: m.f32v(&format!("blk.{i}.attn_v.bias")),
        bo: m.f32v(&format!("blk.{i}.attn_out.bias")),
        sinks: m.f32v(&format!("blk.{i}.attn_sinks")),
        ginp: m.f32v(&format!("blk.{i}.ffn_gate_inp.weight")),
        ginp_b: m.f32v(&format!("blk.{i}.ffn_gate_inp.bias")),
        gate: repack_mx(m.mxfp4(&format!("blk.{i}.ffn_gate_exps.weight")), c.ne * c.d),
        up: repack_mx(m.mxfp4(&format!("blk.{i}.ffn_up_exps.weight")), c.ne * c.d),
        down: repack_mx(m.mxfp4(&format!("blk.{i}.ffn_down_exps.weight")), c.ne * c.d),
        gate_b: m.f32v(&format!("blk.{i}.ffn_gate_exps.bias")),
        up_b: m.f32v(&format!("blk.{i}.ffn_up_exps.bias")),
        down_b: m.f32v(&format!("blk.{i}.ffn_down_exps.bias")),
    }).collect();
    let tok_embd = m.bf16("token_embd.weight");
    let out_norm = m.f32v("output_norm.weight");
    let head = Q8Mat::from_bf16(m.bf16("output.weight"), c.nvocab, c.d);
    eprintln!("prepared in {:.1}s", tq.elapsed().as_secs_f64());

    // leak model to 'static
    let layers: &'static [Layer] = Box::leak(layers.into_boxed_slice());
    let out_norm: &'static [f32] = Box::leak(out_norm.into_boxed_slice());
    let head: &'static Q8Mat = Box::leak(Box::new(head));
    let _m: &'static Model = Box::leak(Box::new(m)); // keep mmap alive

    // allocate activation buffers (leaked, reused across tokens)
    let mk_f = |n: usize| Box::leak(vec![0f32; n].into_boxed_slice()).as_mut_ptr();
    let mk_i8 = |n: usize| Box::leak(vec![0i8; n].into_boxed_slice()).as_mut_ptr();
    let (d, nh, nkv, hd) = (c.d, c.nh, c.nkv, c.hd);
    let sh: &'static Shared = Box::leak(Box::new(Shared {
        tseq: AtomicU64::new(0), ready: AtomicU64::new(0), cur_tok: AtomicU32::new(0), cur_pos: AtomicUsize::new(0),
        want_logits: AtomicBool::new(true), result: AtomicU32::new(0),
        bcount: AtomicUsize::new(0), bsense: AtomicBool::new(false),
        m: std::sync::Mutex::new(()), cv: std::sync::Condvar::new(),
        dm: std::sync::Mutex::new(()), dcv: std::sync::Condvar::new(),
        layers, out_norm, head, tok_embd,
        x: mk_f(d), xn: mk_f(d), xq: mk_i8(d), xsb: mk_f(c.blocks),
        q: mk_f(nh * hd), k: mk_f(nkv * hd), v: mk_f(nkv * hd), ao: mk_f(nh * hd),
        aq: mk_i8(nh * hd), asb: mk_f((nh * hd) / 32),
        gu: mk_f(c.topk * 2 * d), hq: mk_i8(c.topk * d), hsb: mk_f(c.topk * c.blocks),
        rlogits: mk_f(c.ne), logits: mk_f(c.nvocab),
        kc: mk_f(c.nl * MAXSEQ * nkv * hd), vc: mk_f(c.nl * MAXSEQ * nkv * hd),
    }));

    // spawn workers
    for wid in 0..NT() {
        std::thread::spawn(move || {
            unsafe extern "C" { fn pthread_set_qos_class_self_np(q: u32, p: i32) -> i32; }
            // P workers -> USER_INTERACTIVE (P-cores); extra workers -> UTILITY (E-cores)
            unsafe { pthread_set_qos_class_self_np(if wid < NP { 0x21 } else { 0x11 }, 0); }
            let mut ls = false;
            let mut seen = 0u64;
            loop {
                { let mut g = sh.m.lock().unwrap();
                  while sh.tseq.load(Ordering::Acquire) == seen { g = sh.cv.wait(g).unwrap(); }
                  seen = sh.tseq.load(Ordering::Acquire); }
                forward_worker(sh, wid, &mut ls);
                bar(sh, &mut ls); // all workers finish the token before signaling done
                if wid == 0 { let _g = sh.dm.lock().unwrap(); sh.ready.store(seen, Ordering::Release); sh.dcv.notify_one(); }
            }
        });
    }
    let run = |t: u32, pos: usize, want: bool| -> u32 {
        sh.cur_tok.store(t, Ordering::Relaxed);
        sh.cur_pos.store(pos, Ordering::Relaxed);
        sh.want_logits.store(want, Ordering::Relaxed);
        let s = sh.tseq.load(Ordering::Relaxed) + 1;
        { let _g = sh.m.lock().unwrap(); sh.tseq.store(s, Ordering::Release); sh.cv.notify_all(); }
        { let mut g = sh.dm.lock().unwrap(); while sh.ready.load(Ordering::Acquire) < s { g = sh.dcv.wait(g).unwrap(); } }
        sh.result.load(Ordering::Relaxed)
    };

    let serve = prompt == "--serve";
    let cli_ngen: Option<usize> = args.get(4).and_then(|s| s.parse().ok());

    // one full generate pass; streams tokens to stdout, returns (n, secs, prefill_s, prompt_toks)
    let generate = |ptext: &str, ngen: usize| -> (usize, f64, f64, usize) {
        // full harmony prompt (matches Ollama's template for a plain chat turn)
        let system = "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-07-03\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.";
        let mut ids: Vec<u32> = vec![200006];
        ids.extend(encode("system"));
        ids.push(200008);
        ids.extend(encode(system));
        ids.push(200007);
        ids.push(200006);
        ids.extend(encode("user"));
        ids.push(200008);
        ids.extend(encode(ptext));
        ids.push(200007);
        ids.push(200006);
        ids.extend(encode("assistant"));
        let t1 = Instant::now();
        let mut next = 0u32;
        for (i, &t) in ids.iter().enumerate() { next = run(t, i, i == ids.len() - 1); }
        let pf = t1.elapsed().as_secs_f64();
        // reset stage counters so the printed profile reflects DECODE only
        for s in &TM { s.store(0, Ordering::Relaxed); }
        let t2 = Instant::now();
        let mut pos = ids.len();
        let mut n = 0;
        for _ in 0..ngen {
            if next == 200002 || next == 199999 || pos + 1 >= MAXSEQ { break; }
            print!("{}", String::from_utf8_lossy(toks[next as usize]));
            use std::io::Write; std::io::stdout().flush().ok();
            next = run(next, pos, true); pos += 1; n += 1;
        }
        (n, t2.elapsed().as_secs_f64(), pf, ids.len())
    };

    if serve {
        // engine.rs serve protocol: [READY] on stderr; per line either a bare prompt
        // or TSV `temp \t seed \t ngen \t prompt` (prompt \n,\t backslash-escaped);
        // streamed tokens on stdout, then "[STATS] ..." + "[DONE]" lines.
        eprintln!("[READY]");
        let stdin = std::io::stdin();
        let mut line = String::new();
        loop {
            line.clear();
            use std::io::BufRead;
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 { break; }
            let pline = line.trim();
            if pline.is_empty() { continue; }
            let (req_ngen, ptext) = {
                let parts: Vec<&str> = pline.splitn(4, '\t').collect();
                if parts.len() == 4 {
                    if let (Ok(temp), Ok(seed), Ok(ngen)) = (parts[0].parse::<f32>(), parts[1].parse::<u64>(), parts[2].parse::<usize>()) {
                        TEMP_MILLI.store((temp.max(0.0) * 1000.0) as u32, Ordering::Relaxed);
                        // splitmix64 scramble so adjacent seeds diverge; never zero
                        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
                        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                        RNG_STATE.store((z ^ (z >> 31)) | 1, Ordering::Relaxed);
                        let mut unescaped = String::with_capacity(parts[3].len());
                        let mut it = parts[3].chars();
                        while let Some(ch) = it.next() {
                            if ch == '\\' {
                                match it.next() {
                                    Some('n') => unescaped.push('\n'),
                                    Some('t') => unescaped.push('\t'),
                                    Some('\\') => unescaped.push('\\'),
                                    Some(o) => { unescaped.push('\\'); unescaped.push(o) }
                                    None => unescaped.push('\\'),
                                }
                            } else { unescaped.push(ch); }
                        }
                        (ngen.clamp(1, MAXSEQ), unescaped)
                    } else { (0, pline.to_string()) }
                } else { (0, pline.to_string()) }
            };
            let ngen = if req_ngen > 0 { req_ngen } else { 256 };
            let (n, dt, pf, ptoks) = generate(&ptext, ngen);
            use std::io::Write;
            print!("\n[STATS] prefill_tok={ptoks} prefill_s={pf:.2} decode_tok={n} decode_s={dt:.2} tok_s={:.1}\n[DONE]\n", n as f64 / dt);
            std::io::stdout().flush().ok();
        }
    } else {
        let (n, dt, pf, ptoks) = generate(&prompt, cli_ngen.unwrap_or(96));
        println!("\n[STATS] prefill_tok={ptoks} prefill_s={pf:.2} decode_tok={n} decode_s={dt:.2} tok_s={:.1}", n as f64 / dt);
        let nm = ["attn-blk", "gate/up", "down", "head", "ffnorm+rtr"];
        for i in [0, 4, 1, 2, 3] { println!("  {:<10} {:.2} ms/tok", nm[i], TM[i].load(Ordering::Relaxed) as f64 / 1e6 / n.max(1) as f64); }
    }
}
