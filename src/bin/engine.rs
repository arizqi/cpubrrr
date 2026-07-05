//! E13: full gpt-oss:20b generation engine. Real weights straight from the Ollama
//! GGUF blob, all verified pieces wired: bf16 attention (E12 math, YaRN RoPE),
//! int8-activation MXFP4 experts (E9/E11), router f32, greedy decode.
//! Usage: engine <gptoss dir with tokens.bin+manifest.txt> <gguf blob> [prompt]

use std::arch::aarch64::*;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Cfg { d: usize, nh: usize, nkv: usize, hd: usize, nl: usize, ne: usize,
    topk: usize, ffexp: usize, blocks: usize, bpr: usize, nvocab: usize, swa: usize,
    rope_base: f32, rms_eps: f32 }
static CFG: std::sync::OnceLock<Cfg> = std::sync::OnceLock::new();
#[inline(always)]
fn cfg() -> &'static Cfg { CFG.get().unwrap() }
fn load_cfg(dir: &str) {
    let mut m = std::collections::HashMap::new();
    for l in fs::read_to_string(format!("{dir}/config.txt")).unwrap().lines() {
        let mut it = l.split_whitespace();
        if let (Some(k), Some(v)) = (it.next(), it.next()) { m.insert(k.to_string(), v.to_string()); }
    }
    let gi = |k: &str| m.get(k).unwrap_or_else(|| panic!("config missing {k}")).parse::<usize>().unwrap();
    let gf = |k: &str| m.get(k).unwrap().parse::<f32>().unwrap();
    let d = gi("n_embd");
    CFG.set(Cfg {
        d, nh: gi("n_head"), nkv: gi("n_head_kv"), hd: gi("head_dim"),
        nl: gi("n_layer"), ne: gi("n_expert"), topk: gi("n_expert_used"),
        ffexp: gi("n_ff_exp"), blocks: d / 32, bpr: (d / 32) * 17,
        nvocab: gi("n_vocab"), swa: m.get("sliding_window").map(|v| v.parse().unwrap()).unwrap_or(1 << 30),
        rope_base: gf("rope_freq_base"), rms_eps: gf("rms_eps"),
    }).ok();
}
const MAXSEQ: usize = 4096;

// ---- sampling controls (set per request in --serve mode; greedy when TEMP_MILLI=0) ----
static TEMP_MILLI: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static TOPK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(40);
static RNG_STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x9E3779B97F4A7C15);

fn rng_next() -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    let mut x = RNG_STATE.load(Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    RNG_STATE.store(x, Relaxed);
    x
}

/// Sample from logits with temperature + top-k (k<=64). Greedy if temp==0.
fn sample_logits(logits: &[f32]) -> u32 {
    let tm = TEMP_MILLI.load(std::sync::atomic::Ordering::Relaxed);
    if tm == 0 {
        let mut m = (f32::MIN, 0u32);
        for (i, &l) in logits.iter().enumerate() {
            if l > m.0 {
                m = (l, i as u32);
            }
        }
        return m.1;
    }
    let temp = tm as f32 / 1000.0;
    let k = TOPK.load(std::sync::atomic::Ordering::Relaxed).clamp(1, 64);
    // partial top-k: keep k best (small k -> linear scan insert is fine)
    let mut top: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (i, &l) in logits.iter().enumerate() {
        if top.len() < k || l > top.last().unwrap().0 {
            let pos = top.partition_point(|&(v, _)| v > l);
            top.insert(pos, (l, i as u32));
            if top.len() > k {
                top.pop();
            }
        }
    }
    let mx = top[0].0;
    let mut probs: Vec<f32> = top.iter().map(|&(l, _)| ((l - mx) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let r = (rng_next() >> 11) as f32 / (1u64 << 53) as f32;
    let mut acc = 0.0f32;
    for (p, &(_, idx)) in probs.iter().zip(top.iter()) {
        acc += p;
        if r <= acc {
            return idx;
        }
    }
    top.last().unwrap().1
}
const KVI: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
const NT: usize = 12; // worker threads
static STAGE_NS: [std::sync::atomic::AtomicU64; 8] = [
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0),
];
const STAGE_NAMES: [&str; 8] = ["norm+quant", "qkv", "rope+cache", "attention", "o-proj", "router", "gate/up+act", "down"];
struct Tick(std::time::Instant, usize);
impl Tick {
    fn new(i: usize) -> Self { Tick(std::time::Instant::now(), i) }
}
impl Drop for Tick {
    fn drop(&mut self) {
        STAGE_NS[self.1].fetch_add(self.0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

// ---------------- tensor access ----------------
struct Model {
    blob: memmap2::Mmap, // file-backed, evictable (avoids OOM on large models)
    idx: HashMap<String, (u32, usize, usize)>, // type, abs offset, nelems
}
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

unsafe fn dot_bf16(w: *const u16, x: *const f32, n: usize) -> f32 {
    unsafe {
        let zero = vdupq_n_u16(0);
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= n {
            let wv = vld1q_u16(w.add(i));
            let lo = vreinterpretq_f32_u16(vzip1q_u16(zero, wv));
            let hi = vreinterpretq_f32_u16(vzip2q_u16(zero, wv));
            a0 = vfmaq_f32(a0, lo, vld1q_f32(x.add(i)));
            a1 = vfmaq_f32(a1, hi, vld1q_f32(x.add(i + 4)));
            i += 8;
        }
        vaddvq_f32(vaddq_f32(a0, a1))
    }
}

/// MXFP4 row (native GGUF 17B blocks) · block-quantized int8 x -> f32.
/// xsb = per-32-block x scales (Q8_0-style: outliers only poison their own block).
unsafe fn dot_mxfp4_i8(row: *const u8, xq: *const i8, xsb: *const f32) -> f32 {
    unsafe {
        let kv = vld1q_s8(KVI.as_ptr());
        let mask = vdupq_n_u8(0x0F);
        let mut accf = vdupq_n_f32(0.0);
        for b in 0..cfg().blocks {
            let blk = row.add(b * 17);
            let scale = 2f32.powi(*blk as i32 - 128) * *xsb.add(b);
            let w = vld1q_u8(blk.add(1));
            let lo = vqtbl1q_s8(kv, vandq_u8(w, mask));
            let hi = vqtbl1q_s8(kv, vshrq_n_u8::<4>(w));
            let d = sdot(sdot(vdupq_n_s32(0), lo, vld1q_s8(xq.add(b * 32))), hi, vld1q_s8(xq.add(b * 32 + 16)));
            accf = vfmaq_n_f32(accf, vcvtq_f32_s32(d), scale);
        }
        vaddvq_f32(accf)
    }
}

/// Quad-interleaved MXFP4 (E6 lesson 8 layout): one sequential stream per thread.
/// nib: [quad][block][row 0..4][16B], scale: [quad][block][4] (2^(e-128) prefolded).
struct QuadMx {
    nib: Vec<u8>,
    scale: Vec<f32>,
}
fn repack_mx(raw: *const u8, rows: usize) -> QuadMx {
    let mut nib = vec![0u8; rows * cfg().blocks * 16];
    let mut scale = vec![0f32; rows * cfg().blocks];
    {
        let np = SendPtrU8(nib.as_mut_ptr());
        let sp = SendPtr(scale.as_mut_ptr());
        let rp = SendU8(raw as *mut u8 as *const u8);
        par_rows(rows / 4, |a, b| {
            for q in a..b {
                for blk in 0..cfg().blocks {
                    for r in 0..4 {
                        unsafe {
                            let src = rp.get().add(((q * 4 + r) * cfg().blocks + blk) * 17);
                            *sp.get().add((q * cfg().blocks + blk) * 4 + r) = 2f32.powi(*src as i32 - 128);
                            std::ptr::copy_nonoverlapping(src.add(1), np.get().add((q * cfg().blocks + blk) * 64 + r * 16), 16);
                        }
                    }
                }
            }
        });
    }
    QuadMx { nib, scale }
}

#[derive(Clone, Copy)]
struct SendPtrU8(*mut u8);
unsafe impl Send for SendPtrU8 {}
unsafe impl Sync for SendPtrU8 {}
impl SendPtrU8 {
    fn get(self) -> *mut u8 {
        self.0
    }
}

/// 4 rows x cfg().d against block-quantized int8 x, quad-interleaved MXFP4.
unsafe fn dot4_mx_i8(nib: *const u8, sc: *const f32, xq: *const i8, xsb: *const f32) -> [f32; 4] {
    unsafe {
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
            let sv = vmulq_n_f32(vld1q_f32(sp), *xsb.add(b));
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

/// Per-32-block int8 quantization (Q8_0-style).
fn quant_i8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let nb = x.len() / 32;
    let mut q = vec![0i8; x.len()];
    let mut s = vec![0f32; nb];
    for b in 0..nb {
        let blk = &x[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
        let sc = amax / 127.0;
        s[b] = sc;
        for (i, &v) in blk.iter().enumerate() {
            q[b * 32 + i] = (v / sc).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, s)
}

/// Q8 weight row · Q8 activation: int sdot per 32-block, scales multiplied in.
unsafe fn dot_q8_i8(wq: *const i8, ws: *const f32, xq: *const i8, xsb: *const f32, cols: usize) -> f32 {
    unsafe {
        let mut accf = vdupq_n_f32(0.0);
        for b in 0..cols / 32 {
            let w0 = vld1q_s8(wq.add(b * 32));
            let w1 = vld1q_s8(wq.add(b * 32 + 16));
            let d = sdot(sdot(vdupq_n_s32(0), w0, vld1q_s8(xq.add(b * 32))), w1, vld1q_s8(xq.add(b * 32 + 16)));
            accf = vfmaq_n_f32(accf, vcvtq_f32_s32(d), *ws.add(b) * *xsb.add(b));
        }
        vaddvq_f32(accf)
    }
}

/// Q8_0 weight matrix converted from bf16 at load (halves traffic vs bf16).
struct Q8Mat {
    q: Vec<i8>,
    s: Vec<f32>,
    cols: usize,
}
impl Q8Mat {
    fn from_bf16(w: *const u16, rows: usize, cols: usize) -> Self {
        let nb = cols / 32;
        let mut q = vec![0i8; rows * cols];
        let mut s = vec![0f32; rows * nb];
        {
            let qp = SendPtrI8(q.as_mut_ptr());
            let sp = SendPtr(s.as_mut_ptr());
            let wp = SendU16(w);
            par_rows(rows, |a, b| {
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
                            unsafe {
                                *qp.get().add(r * cols + blk * 32 + i) =
                                    (vals[i] / sc).round().clamp(-127.0, 127.0) as i8
                            };
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
            dot_q8_i8(
                self.q.as_ptr().add(row * self.cols),
                self.s.as_ptr().add(row * (self.cols / 32)),
                xq,
                xsb,
                self.cols,
            )
        }
    }
}

#[derive(Clone, Copy)]
struct SendPtrI8(*mut i8);
unsafe impl Send for SendPtrI8 {}
unsafe impl Sync for SendPtrI8 {}
impl SendPtrI8 {
    fn get(self) -> *mut i8 {
        self.0
    }
}

fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + cfg().rms_eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

/// YaRN RoPE (freq_base 150000, factor 32, orig ctx 4096, beta 32/1, ext_factor 1).
fn rope_yarn(v: &mut [f32], pos: usize) {
    if std::env::var("NO_ROPE").is_ok() { return; }
    let half = cfg().hd / 2;
    let base = cfg().rope_base;
    if std::env::var("PLAIN_ROPE").is_ok() {
        for i in 0..half {
            let theta = pos as f32 * base.powf(-(2.0 * i as f32) / cfg().hd as f32);
            let (c, s) = (theta.cos(), theta.sin());
            let a = v[i];
            let b = v[half + i];
            v[i] = a * c - b * s;
            v[half + i] = b * c + a * s;
        }
        return;
    }
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
        let a = v[i];
        let b = v[half + i];
        v[i] = a * c - b * s;
        v[half + i] = b * c + a * s;
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
#[derive(Clone, Copy)]
struct SendU16(*const u16);
unsafe impl Send for SendU16 {}
unsafe impl Sync for SendU16 {}
impl SendU16 {
    fn get(self) -> *const u16 {
        self.0
    }
}
#[derive(Clone, Copy)]
struct SendU8(*const u8);
unsafe impl Send for SendU8 {}
unsafe impl Sync for SendU8 {}
impl SendU8 {
    fn get(self) -> *const u8 {
        self.0
    }
}

// ---------------- persistent worker pool ----------------
// Threads park on a condvar; each job is a lifetime-erased &dyn Fn(worker_id).
// run() blocks until all workers finish, so the borrow stays valid.
struct PoolShared {
    m: std::sync::Mutex<PoolState>,
    cv_go: std::sync::Condvar,
    cv_done: std::sync::Condvar,
}
struct PoolState {
    seq: u64,
    task: [usize; 2],
    remaining: usize,
}
static POOL: std::sync::OnceLock<&'static PoolShared> = std::sync::OnceLock::new();

fn pool_init() {
    let shared: &'static PoolShared = Box::leak(Box::new(PoolShared {
        m: std::sync::Mutex::new(PoolState { seq: 0, task: [0, 0], remaining: 0 }),
        cv_go: std::sync::Condvar::new(),
        cv_done: std::sync::Condvar::new(),
    }));
    for wid in 0..NT {
        std::thread::spawn(move || {
            let mut seen = 0u64;
            loop {
                let task;
                {
                    let mut st = shared.m.lock().unwrap();
                    while st.seq == seen {
                        st = shared.cv_go.wait(st).unwrap();
                    }
                    seen = st.seq;
                    task = st.task;
                }
                let f: &(dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(task) };
                f(wid);
                let mut st = shared.m.lock().unwrap();
                st.remaining -= 1;
                if st.remaining == 0 {
                    shared.cv_done.notify_one();
                }
            }
        });
    }
    POOL.set(shared).ok();
}

fn pool_run(f: &(dyn Fn(usize) + Sync)) {
    let shared = POOL.get().unwrap();
    let fat: [usize; 2] = unsafe { std::mem::transmute(f as *const (dyn Fn(usize) + Sync)) };
    let mut st = shared.m.lock().unwrap();
    st.task = fat;
    st.seq += 1;
    st.remaining = NT;
    shared.cv_go.notify_all();
    while st.remaining > 0 {
        st = shared.cv_done.wait(st).unwrap();
    }
}

fn par_rows<F: Fn(usize, usize) + Sync>(n: usize, f: F) {
    if POOL.get().is_none() {
        // pre-init fallback (used during weight quantization at load)
        let chunk = (n + NT - 1) / NT;
        std::thread::scope(|s| {
            for t in 0..NT {
                let a = t * chunk;
                if a >= n {
                    break;
                }
                let b = (a + chunk).min(n);
                let f = &f;
                s.spawn(move || f(a, b));
            }
        });
        return;
    }
    let chunk = (n + NT - 1) / NT;
    pool_run(&|t: usize| {
        let a = t * chunk;
        if a >= n {
            return;
        }
        let b = (a + chunk).min(n);
        f(a, b);
    });
}

// ---------------- per-layer weights ----------------
struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq: Q8Mat,
    wk: Q8Mat,
    wv: Q8Mat,
    wo: Q8Mat,
    bq: Vec<f32>,
    bk: Vec<f32>,
    bv: Vec<f32>,
    bo: Vec<f32>,
    sinks: Vec<f32>,
    ginp: Vec<f32>,
    ginp_b: Vec<f32>,
    gate: QuadMx,
    up: QuadMx,
    down: QuadMx,
    gate_b: Vec<f32>,
    up_b: Vec<f32>,
    down_b: Vec<f32>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let blobp = &args[2];
    load_cfg(dir);
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Why is the sky blue?".into());

    // tokenizer
    let tb = fs::read(format!("{dir}/tokens.bin")).unwrap();
    let ntok = u32::from_le_bytes(tb[0..4].try_into().unwrap()) as usize;
    let mut toks: Vec<&[u8]> = Vec::with_capacity(ntok);
    let mut p = 4;
    for _ in 0..ntok {
        let l = u16::from_le_bytes(tb[p..p + 2].try_into().unwrap()) as usize;
        toks.push(&tb[p + 2..p + 2 + l]);
        p += 2 + l;
    }
    let mut lookup: HashMap<&[u8], u32> = HashMap::new();
    for (i, t) in toks.iter().enumerate() {
        if !t.is_empty() && !t.starts_with(b"<|") {
            lookup.entry(*t).or_insert(i as u32);
        }
    }
    let encode = |s: &str| -> Vec<u32> {
        let b = s.as_bytes();
        let mut ids = Vec::new();
        let mut i = 0;
        while i < b.len() {
            let mut l = (b.len() - i).min(32);
            loop {
                if let Some(&id) = lookup.get(&b[i..i + l]) {
                    ids.push(id);
                    i += l;
                    break;
                }
                l -= 1;
                assert!(l > 0, "unencodable byte {}", b[i]);
            }
        }
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
    for l in lines {
        let f: Vec<&str> = l.split_whitespace().collect();
        idx.insert(f[0].to_string(), (f[1].parse().unwrap(), ds + f[2].parse::<usize>().unwrap(), f[3].parse().unwrap()));
    }
    let m = Model { blob, idx };
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let layers: Vec<Layer> = (0..cfg().nl)
        .map(|i| Layer {
            attn_norm: m.f32v(&format!("blk.{i}.attn_norm.weight")),
            ffn_norm: m.f32v(&format!("blk.{i}.ffn_norm.weight")),
            wq: Q8Mat::from_bf16(m.bf16(&format!("blk.{i}.attn_q.weight")), cfg().nh * cfg().hd, cfg().d),
            wk: Q8Mat::from_bf16(m.bf16(&format!("blk.{i}.attn_k.weight")), cfg().nkv * cfg().hd, cfg().d),
            wv: Q8Mat::from_bf16(m.bf16(&format!("blk.{i}.attn_v.weight")), cfg().nkv * cfg().hd, cfg().d),
            wo: Q8Mat::from_bf16(m.bf16(&format!("blk.{i}.attn_out.weight")), cfg().d, cfg().nh * cfg().hd),
            bq: m.f32v(&format!("blk.{i}.attn_q.bias")),
            bk: m.f32v(&format!("blk.{i}.attn_k.bias")),
            bv: m.f32v(&format!("blk.{i}.attn_v.bias")),
            bo: m.f32v(&format!("blk.{i}.attn_out.bias")),
            sinks: m.f32v(&format!("blk.{i}.attn_sinks")),
            ginp: m.f32v(&format!("blk.{i}.ffn_gate_inp.weight")),
            ginp_b: m.f32v(&format!("blk.{i}.ffn_gate_inp.bias")),
            gate: repack_mx(m.mxfp4(&format!("blk.{i}.ffn_gate_exps.weight")), cfg().ne * cfg().d),
            up: repack_mx(m.mxfp4(&format!("blk.{i}.ffn_up_exps.weight")), cfg().ne * cfg().d),
            down: repack_mx(m.mxfp4(&format!("blk.{i}.ffn_down_exps.weight")), cfg().ne * cfg().d),
            gate_b: m.f32v(&format!("blk.{i}.ffn_gate_exps.bias")),
            up_b: m.f32v(&format!("blk.{i}.ffn_up_exps.bias")),
            down_b: m.f32v(&format!("blk.{i}.ffn_down_exps.bias")),
        })
        .collect();
    let tok_embd = SendU16(m.bf16("token_embd.weight"));
    let out_norm = m.f32v("output_norm.weight");
    eprintln!("quantizing attention + head to Q8...");
    let tq = Instant::now();
    let head = Q8Mat::from_bf16(m.bf16("output.weight"), cfg().nvocab, cfg().d);
    eprintln!("quantized in {:.1}s", tq.elapsed().as_secs_f64());

    // KV caches
    let mut kc = vec![vec![0f32; MAXSEQ * cfg().nkv * cfg().hd]; cfg().nl];
    let mut vc = vec![vec![0f32; MAXSEQ * cfg().nkv * cfg().hd]; cfg().nl];

    // forward one token; returns logits argmax if want_logits
    let mut forward = |tok: u32, pos: usize, want: bool, kc: &mut Vec<Vec<f32>>, vc: &mut Vec<Vec<f32>>| -> u32 {
        let mut x: Vec<f32> = (0..cfg().d)
            .map(|j| {
                let bits = (unsafe { *tok_embd.0.add(tok as usize * cfg().d + j) } as u32) << 16;
                f32::from_bits(bits)
            })
            .collect();
        for (il, ly) in layers.iter().enumerate() {
            // ---- attention ----
            let _t = Tick::new(0);
            let xn = rmsnorm(&x, &ly.attn_norm);
            let (xnq, xns) = quant_i8(&xn);
            drop(_t);
            let _t = Tick::new(1);
            let mut qkv = vec![0f32; (cfg().nh + 2 * cfg().nkv) * cfg().hd];
            {
                let out = SendPtr(qkv.as_mut_ptr());
                let (xnq, xns) = (&xnq, &xns);
                par_rows(cfg().nh * cfg().hd + 2 * cfg().nkv * cfg().hd, |a, b| {
                    for r in a..b {
                        let (w, bias, base) = if r < cfg().nh * cfg().hd {
                            (&ly.wq, &ly.bq, 0)
                        } else if r < cfg().nh * cfg().hd + cfg().nkv * cfg().hd {
                            (&ly.wk, &ly.bk, cfg().nh * cfg().hd)
                        } else {
                            (&ly.wv, &ly.bv, cfg().nh * cfg().hd + cfg().nkv * cfg().hd)
                        };
                        let rr = r - base;
                        unsafe {
                            *out.get().add(r) = w.dot(rr, xnq.as_ptr(), xns.as_ptr()) + bias[rr];
                        }
                    }
                });
            }
            drop(_t);
            let _t = Tick::new(2);
            let (q, rest) = qkv.split_at_mut(cfg().nh * cfg().hd);
            let (k, v) = rest.split_at_mut(cfg().nkv * cfg().hd);
            for h in 0..cfg().nh {
                rope_yarn(&mut q[h * cfg().hd..(h + 1) * cfg().hd], pos);
            }
            for h in 0..cfg().nkv {
                rope_yarn(&mut k[h * cfg().hd..(h + 1) * cfg().hd], pos);
            }
            kc[il][pos * cfg().nkv * cfg().hd..(pos + 1) * cfg().nkv * cfg().hd].copy_from_slice(k);
            vc[il][pos * cfg().nkv * cfg().hd..(pos + 1) * cfg().nkv * cfg().hd].copy_from_slice(v);
            // sliding window on even layers
            let start = if (il + 1) % 2 != 0 { pos.saturating_sub(cfg().swa - 1) } else { 0 };
            drop(_t);
            let _t = Tick::new(3);
            let scale = 1.0 / (cfg().hd as f32).sqrt();
            let mut attnout = vec![0f32; cfg().nh * cfg().hd];
            {
                let kcl: &[f32] = &kc[il];
                let vcl: &[f32] = &vc[il];
                let q: &[f32] = q;
                let out = SendPtr(attnout.as_mut_ptr());
                par_rows(cfg().nh, |ha, hb| {
                    for h in ha..hb {
                        let kvh = h / (cfg().nh / cfg().nkv);
                        let qp = q.as_ptr();
                        let mut scores: Vec<f32> = (start..=pos)
                            .map(|t| unsafe {
                                let kp = kcl.as_ptr().add((t * cfg().nkv + kvh) * cfg().hd);
                                let qh = qp.add(h * cfg().hd);
                                let mut a0 = vdupq_n_f32(0.0);
                                let mut a1 = vdupq_n_f32(0.0);
                                let mut a2 = vdupq_n_f32(0.0);
                                let mut a3 = vdupq_n_f32(0.0);
                                let mut j = 0;
                                while j < cfg().hd {
                                    a0 = vfmaq_f32(a0, vld1q_f32(kp.add(j)), vld1q_f32(qh.add(j)));
                                    a1 = vfmaq_f32(a1, vld1q_f32(kp.add(j + 4)), vld1q_f32(qh.add(j + 4)));
                                    a2 = vfmaq_f32(a2, vld1q_f32(kp.add(j + 8)), vld1q_f32(qh.add(j + 8)));
                                    a3 = vfmaq_f32(a3, vld1q_f32(kp.add(j + 12)), vld1q_f32(qh.add(j + 12)));
                                    j += 16;
                                }
                                vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * scale
                            })
                            .collect();
                        let mx = scores.iter().cloned().fold(ly.sinks[h], f32::max);
                        let mut denom = (ly.sinks[h] - mx).exp();
                        for s in scores.iter_mut() {
                            *s = (*s - mx).exp();
                            denom += *s;
                        }
                        unsafe {
                            let mut acc = [vdupq_n_f32(0.0); 16];
                            for (ti, t) in (start..=pos).enumerate() {
                                let vp = vcl.as_ptr().add((t * cfg().nkv + kvh) * cfg().hd);
                                let sv = scores[ti];
                                for u in 0..16 {
                                    acc[u] = vfmaq_n_f32(acc[u], vld1q_f32(vp.add(u * 4)), sv);
                                }
                            }
                            let inv = 1.0 / denom;
                            for u in 0..16 {
                                vst1q_f32(out.get().add(h * cfg().hd + u * 4), vmulq_n_f32(acc[u], inv));
                            }
                        }
                    }
                });
            }
            drop(_t);
            let _t = Tick::new(4);
            {
                let (aoq, aos) = quant_i8(&attnout);
                let out = SendPtr(x.as_mut_ptr());
                let (aoq, aos) = (&aoq, &aos);
                par_rows(cfg().d, |a, b| {
                    for r in a..b {
                        unsafe {
                            *out.get().add(r) += ly.wo.dot(r, aoq.as_ptr(), aos.as_ptr()) + ly.bo[r];
                        }
                    }
                });
            }
            drop(_t);
            let _t = Tick::new(5);
            // ---- MoE FFN ----
            let xn2 = rmsnorm(&x, &ly.ffn_norm);
            let logits: Vec<f32> = (0..cfg().ne)
                .map(|e| {
                    let mut a = ly.ginp_b[e];
                    for j in 0..cfg().d {
                        a += ly.ginp[e * cfg().d + j] * xn2[j];
                    }
                    a
                })
                .collect();
            let mut order: Vec<usize> = (0..cfg().ne).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let top = &order[..cfg().topk];
            let mx = top.iter().map(|&e| logits[e]).fold(f32::MIN, f32::max);
            let exps: Vec<f32> = top.iter().map(|&e| (logits[e] - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let wts: Vec<f32> = exps.iter().map(|v| v / sum).collect();

            drop(_t);
            let _t = Tick::new(6);
            let (xq, xs) = quant_i8(&xn2);
            let mut gu = vec![0f32; cfg().topk * 2 * cfg().d];
            {
                let out = SendPtr(gu.as_mut_ptr());
                let (xq, xs, top) = (&xq, &xs, &top);
                let dq = cfg().d / 4;
                par_rows(cfg().topk * 2 * dq, |a, b| {
                    for it in a..b {
                        let ei = it / (2 * dq);
                        let e = top[ei];
                        let rem = it % (2 * dq);
                        let (w, bias, qd, obase) = if rem < dq {
                            (&ly.gate, &ly.gate_b, rem, ei * 2 * cfg().d)
                        } else {
                            (&ly.up, &ly.up_b, rem - dq, ei * 2 * cfg().d + cfg().d)
                        };
                        let qg = e * dq + qd;
                        let acc = unsafe {
                            dot4_mx_i8(
                                w.nib.as_ptr().add(qg * cfg().blocks * 64),
                                w.scale.as_ptr().add(qg * cfg().blocks * 4),
                                xq.as_ptr(),
                                xs.as_ptr(),
                            )
                        };
                        for i in 0..4 {
                            unsafe { *out.get().add(obase + qd * 4 + i) = acc[i] + bias[e * cfg().d + qd * 4 + i] };
                        }
                    }
                });
            }
            const ALPHA: f32 = 1.702;
            const LIM: f32 = 7.0;
            let mut hq = vec![0i8; cfg().topk * cfg().d];
            let mut hs = vec![0f32; cfg().topk * cfg().blocks];
            for ei in 0..cfg().topk {
                let g = &gu[ei * 2 * cfg().d..ei * 2 * cfg().d + cfg().d];
                let u = &gu[ei * 2 * cfg().d + cfg().d..ei * 2 * cfg().d + 2 * cfg().d];
                let h: Vec<f32> = (0..cfg().d)
                    .map(|k| {
                        let xg = g[k].min(LIM);
                        let yu = u[k].clamp(-LIM, LIM);
                        (xg / (1.0 + (-ALPHA * xg).exp())) * (yu + 1.0)
                    })
                    .collect();
                let (q8, s8) = quant_i8(&h);
                hq[ei * cfg().d..(ei + 1) * cfg().d].copy_from_slice(&q8);
                hs[ei * cfg().blocks..(ei + 1) * cfg().blocks].copy_from_slice(&s8);
            }
            drop(_t);
            let _t = Tick::new(7);
            // down-proj: single fused pass, 4 experts to disjoint buffers, weighted
            let mut ffn4 = vec![0f32; cfg().topk * cfg().d];
            {
                let contrib = SendPtr(ffn4.as_mut_ptr());
                let (hq, hs, top, wts) = (&hq, &hs, &top, &wts);
                let dq = cfg().d / 4;
                par_rows(cfg().topk * dq, |a, b| {
                    for it in a..b {
                        let ei = it / dq;
                        let e = top[ei];
                        let qd = it % dq;
                        let qg = e * dq + qd;
                        let acc = unsafe {
                            dot4_mx_i8(
                                ly.down.nib.as_ptr().add(qg * cfg().blocks * 64),
                                ly.down.scale.as_ptr().add(qg * cfg().blocks * 4),
                                hq.as_ptr().add(ei * cfg().d),
                                hs.as_ptr().add(ei * cfg().blocks),
                            )
                        };
                        for i in 0..4 {
                            unsafe {
                                *contrib.get().add(ei * cfg().d + qd * 4 + i) = wts[ei] * (acc[i] + ly.down_b[e * cfg().d + qd * 4 + i])
                            };
                        }
                    }
                });
            }
            for ei in 0..cfg().topk {
                for kx in 0..cfg().d {
                    x[kx] += ffn4[ei * cfg().d + kx];
                }
            }
            drop(_t);
            if pos <= 2 && std::env::var("DUMP").is_ok() {
                let n = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
                eprintln!("p{pos} layer {il:2} |x| {n:10.4}");
            }
        }
        if std::env::var("DUMP_POS").is_ok() {
            let n = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            eprintln!("pos {pos} tok {tok}: |x| {n:.2}");
        } else if !want {
            return 0;
        }
        let xn = rmsnorm(&x, &out_norm);
        let (xq, xs) = quant_i8(&xn);
        let mut logits = vec![0f32; cfg().nvocab];
        {
            let lp = logits.as_mut_ptr() as usize;
            let chunk = (cfg().nvocab + NT - 1) / NT;
            std::thread::scope(|s| {
                for t in 0..NT {
                    let a = t * chunk;
                    let b = ((t + 1) * chunk).min(cfg().nvocab);
                    let (xq, xs, head) = (&xq, &xs, &head);
                    s.spawn(move || {
                        for r in a..b {
                            let l = unsafe { head.dot(r, xq.as_ptr(), xs.as_ptr()) };
                            unsafe { *(lp as *mut f32).add(r) = l };
                        }
                    });
                }
            });
        }
        let chosen = sample_logits(&logits);
        if std::env::var("DUMP_POS").is_ok() {
            eprintln!("  pos {pos} chosen {} logit {:.3}", chosen, logits[chosen as usize]);
        }
        chosen
    };

    let qkvd: usize = (cfg().nh + 2 * cfg().nkv) * cfg().hd;
    const TILE: usize = 8; // tokens per tile: 8 x 2880 i8 activations stay L1-resident
    // Batched prefill: one pass over weights serves the whole prompt (rows stay L1-hot
    // across tokens). Causal: token b attends cache 0..=pos0+b only.
    let forward_batch = |toks: &[u32], pos0: usize, kc: &mut Vec<Vec<f32>>, vc: &mut Vec<Vec<f32>>| -> u32 {
        let bsz = toks.len();
        let mut xs: Vec<Vec<f32>> = toks
            .iter()
            .map(|&t| {
                (0..cfg().d)
                    .map(|j| f32::from_bits((unsafe { *tok_embd.get().add(t as usize * cfg().d + j) } as u32) << 16))
                    .collect()
            })
            .collect();
        for (il, ly) in layers.iter().enumerate() {
            let xnq: Vec<(Vec<i8>, Vec<f32>)> = xs
                .iter()
                .map(|x| {
                    let xn = rmsnorm(x, &ly.attn_norm);
                    quant_i8(&xn)
                })
                .collect();
            let mut qkvs = vec![0f32; bsz * qkvd];
            for t0 in (0..bsz).step_by(TILE) {
                let t1e = (t0 + TILE).min(bsz);
                let out = SendPtr(qkvs.as_mut_ptr());
                let xnq = &xnq;
                par_rows(qkvd, |a, b2| {
                    for r in a..b2 {
                        let (w, bias, base) = if r < cfg().nh * cfg().hd {
                            (&ly.wq, &ly.bq, 0)
                        } else if r < cfg().nh * cfg().hd + cfg().nkv * cfg().hd {
                            (&ly.wk, &ly.bk, cfg().nh * cfg().hd)
                        } else {
                            (&ly.wv, &ly.bv, cfg().nh * cfg().hd + cfg().nkv * cfg().hd)
                        };
                        let rr = r - base;
                        for b in t0..t1e {
                            unsafe {
                                *out.get().add(b * qkvd + r) =
                                    w.dot(rr, xnq[b].0.as_ptr(), xnq[b].1.as_ptr()) + bias[rr];
                            }
                        }
                    }
                });
            }
            for b in 0..bsz {
                let pos = pos0 + b;
                let base = b * qkvd;
                for h in 0..cfg().nh {
                    rope_yarn(&mut qkvs[base + h * cfg().hd..base + (h + 1) * cfg().hd], pos);
                }
                for h in 0..cfg().nkv {
                    let o = base + cfg().nh * cfg().hd + h * cfg().hd;
                    rope_yarn(&mut qkvs[o..o + cfg().hd], pos);
                }
                kc[il][pos * cfg().nkv * cfg().hd..(pos + 1) * cfg().nkv * cfg().hd]
                    .copy_from_slice(&qkvs[base + cfg().nh * cfg().hd..base + cfg().nh * cfg().hd + cfg().nkv * cfg().hd]);
                vc[il][pos * cfg().nkv * cfg().hd..(pos + 1) * cfg().nkv * cfg().hd]
                    .copy_from_slice(&qkvs[base + cfg().nh * cfg().hd + cfg().nkv * cfg().hd..base + qkvd]);
            }
            let scale = 1.0 / (cfg().hd as f32).sqrt();
            let mut aouts = vec![0f32; bsz * cfg().nh * cfg().hd];
            {
                let kcl: &[f32] = &kc[il];
                let vcl: &[f32] = &vc[il];
                let qs: &[f32] = &qkvs;
                let out = SendPtr(aouts.as_mut_ptr());
                par_rows(bsz * cfg().nh, |a, b2| {
                    for i in a..b2 {
                        let b = i / cfg().nh;
                        let h = i % cfg().nh;
                        let pos = pos0 + b;
                        let start = if (il + 1) % 2 != 0 { pos.saturating_sub(cfg().swa - 1) } else { 0 };
                        let kvh = h / (cfg().nh / cfg().nkv);
                        let qh = unsafe { qs.as_ptr().add(b * qkvd + h * cfg().hd) };
                        let mut scores: Vec<f32> = (start..=pos)
                            .map(|t| unsafe {
                                let kp = kcl.as_ptr().add((t * cfg().nkv + kvh) * cfg().hd);
                                let mut a0 = vdupq_n_f32(0.0);
                                let mut a1 = vdupq_n_f32(0.0);
                                let mut a2 = vdupq_n_f32(0.0);
                                let mut a3 = vdupq_n_f32(0.0);
                                let mut j = 0;
                                while j < cfg().hd {
                                    a0 = vfmaq_f32(a0, vld1q_f32(kp.add(j)), vld1q_f32(qh.add(j)));
                                    a1 = vfmaq_f32(a1, vld1q_f32(kp.add(j + 4)), vld1q_f32(qh.add(j + 4)));
                                    a2 = vfmaq_f32(a2, vld1q_f32(kp.add(j + 8)), vld1q_f32(qh.add(j + 8)));
                                    a3 = vfmaq_f32(a3, vld1q_f32(kp.add(j + 12)), vld1q_f32(qh.add(j + 12)));
                                    j += 16;
                                }
                                vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * scale
                            })
                            .collect();
                        let mx = scores.iter().cloned().fold(ly.sinks[h], f32::max);
                        let mut denom = (ly.sinks[h] - mx).exp();
                        for sc in scores.iter_mut() {
                            *sc = (*sc - mx).exp();
                            denom += *sc;
                        }
                        unsafe {
                            let mut acc = [vdupq_n_f32(0.0); 16];
                            for (ti, t) in (start..=pos).enumerate() {
                                let vp = vcl.as_ptr().add((t * cfg().nkv + kvh) * cfg().hd);
                                let sv = scores[ti];
                                for u in 0..16 {
                                    acc[u] = vfmaq_n_f32(acc[u], vld1q_f32(vp.add(u * 4)), sv);
                                }
                            }
                            let inv = 1.0 / denom;
                            for u in 0..16 {
                                vst1q_f32(out.get().add(b * cfg().nh * cfg().hd + h * cfg().hd + u * 4), vmulq_n_f32(acc[u], inv));
                            }
                        }
                    }
                });
            }
            {
                let aq: Vec<(Vec<i8>, Vec<f32>)> =
                    (0..bsz).map(|b| quant_i8(&aouts[b * cfg().nh * cfg().hd..(b + 1) * cfg().nh * cfg().hd])).collect();
                let xps: Vec<SendPtr> = xs.iter_mut().map(|v| SendPtr(v.as_mut_ptr())).collect();
                let (aq, xps) = (&aq, &xps);
                for t0 in (0..bsz).step_by(TILE) {
                    let t1e = (t0 + TILE).min(bsz);
                    par_rows(cfg().d, |a, b2| {
                        for r in a..b2 {
                            for b in t0..t1e {
                                unsafe {
                                    *xps[b].get().add(r) += ly.wo.dot(r, aq[b].0.as_ptr(), aq[b].1.as_ptr()) + ly.bo[r];
                                }
                            }
                        }
                    });
                }
            }
            // ---- MoE FFN, expert-major batching ----
            let mut tops = vec![vec![0usize; cfg().topk]; bsz];
            let mut wtss = vec![vec![0f32; cfg().topk]; bsz];
            let xn2q: Vec<(Vec<i8>, Vec<f32>)> = (0..bsz)
                .map(|b| {
                    let xn2 = rmsnorm(&xs[b], &ly.ffn_norm);
                    let logits: Vec<f32> = (0..cfg().ne)
                        .map(|e| {
                            let mut acc = ly.ginp_b[e];
                            for j in 0..cfg().d {
                                acc += ly.ginp[e * cfg().d + j] * xn2[j];
                            }
                            acc
                        })
                        .collect();
                    let mut order: Vec<usize> = (0..cfg().ne).collect();
                    order.sort_by(|&p1, &p2| logits[p2].partial_cmp(&logits[p1]).unwrap());
                    let mx = (0..cfg().topk).map(|i| logits[order[i]]).fold(f32::MIN, f32::max);
                    let mut sum = 0.0;
                    let mut ws = vec![0f32; cfg().topk];
                    for i in 0..cfg().topk {
                        ws[i] = (logits[order[i]] - mx).exp();
                        sum += ws[i];
                    }
                    for i in 0..cfg().topk {
                        ws[i] /= sum;
                        tops[b][i] = order[i];
                    }
                    wtss[b] = ws;
                    quant_i8(&xn2)
                })
                .collect();
            let mut subs: Vec<Vec<(usize, usize)>> = vec![Vec::new(); cfg().ne];
            for b in 0..bsz {
                for ei in 0..cfg().topk {
                    subs[tops[b][ei]].push((b, ei));
                }
            }
            let active: Vec<usize> = (0..cfg().ne).filter(|&e| !subs[e].is_empty()).collect();
            let mut gu = vec![0f32; bsz * cfg().topk * 2 * cfg().d];
            {
                let out = SendPtr(gu.as_mut_ptr());
                let (subs, active, xn2q) = (&subs, &active, &xn2q);
                for t0 in (0..bsz).step_by(TILE) {
                let t1e = (t0 + TILE).min(bsz);
                par_rows(active.len() * cfg().d, |a, b2| {
                    for it in a..b2 {
                        let e = active[it / cfg().d];
                        let rr = it % cfg().d;
                        let qd = rr / 4;
                        if rr % 4 != 0 { continue; }
                        let qg = e * (cfg().d / 4) + qd;
                        for &(b, ei) in subs[e].iter().filter(|&&(b, _)| b >= t0 && b < t1e) {
                            let gs = unsafe {
                                dot4_mx_i8(ly.gate.nib.as_ptr().add(qg * cfg().blocks * 64), ly.gate.scale.as_ptr().add(qg * cfg().blocks * 4), xn2q[b].0.as_ptr(), xn2q[b].1.as_ptr())
                            };
                            let us = unsafe {
                                dot4_mx_i8(ly.up.nib.as_ptr().add(qg * cfg().blocks * 64), ly.up.scale.as_ptr().add(qg * cfg().blocks * 4), xn2q[b].0.as_ptr(), xn2q[b].1.as_ptr())
                            };
                            for i in 0..4 {
                                unsafe {
                                    *out.get().add((b * cfg().topk + ei) * 2 * cfg().d + qd * 4 + i) = gs[i] + ly.gate_b[e * cfg().d + qd * 4 + i];
                                    *out.get().add((b * cfg().topk + ei) * 2 * cfg().d + cfg().d + qd * 4 + i) = us[i] + ly.up_b[e * cfg().d + qd * 4 + i];
                                }
                            }
                        }
                    }
                });
                }
            }
            const ALPHA: f32 = 1.702;
            const LIM: f32 = 7.0;
            let mut hq = vec![0i8; bsz * cfg().topk * cfg().d];
            let mut hsb = vec![0f32; bsz * cfg().topk * cfg().blocks];
            for b in 0..bsz {
                for ei in 0..cfg().topk {
                    let o = (b * cfg().topk + ei) * 2 * cfg().d;
                    let hvec: Vec<f32> = (0..cfg().d)
                        .map(|k2| {
                            let xg = gu[o + k2].min(LIM);
                            let yu = gu[o + cfg().d + k2].clamp(-LIM, LIM);
                            (xg / (1.0 + (-ALPHA * xg).exp())) * (yu + 1.0)
                        })
                        .collect();
                    let (q8, s8) = quant_i8(&hvec);
                    hq[(b * cfg().topk + ei) * cfg().d..(b * cfg().topk + ei + 1) * cfg().d].copy_from_slice(&q8);
                    hsb[(b * cfg().topk + ei) * cfg().blocks..(b * cfg().topk + ei + 1) * cfg().blocks].copy_from_slice(&s8);
                }
            }
            let mut ffn4 = vec![0f32; bsz * cfg().topk * cfg().d];
            {
                let out = SendPtr(ffn4.as_mut_ptr());
                let (subs, active, hq, hsb) = (&subs, &active, &hq, &hsb);
                for t0 in (0..bsz).step_by(TILE) {
                let t1e = (t0 + TILE).min(bsz);
                par_rows(active.len() * cfg().d, |a, b2| {
                    for it in a..b2 {
                        let e = active[it / cfg().d];
                        let rr = it % cfg().d;
                        let qd = rr / 4;
                        if rr % 4 != 0 { continue; }
                        let qg = e * (cfg().d / 4) + qd;
                        for &(b, ei) in subs[e].iter().filter(|&&(b, _)| b >= t0 && b < t1e) {
                            let ds = unsafe {
                                dot4_mx_i8(ly.down.nib.as_ptr().add(qg * cfg().blocks * 64), ly.down.scale.as_ptr().add(qg * cfg().blocks * 4), hq.as_ptr().add((b * cfg().topk + ei) * cfg().d), hsb.as_ptr().add((b * cfg().topk + ei) * cfg().blocks))
                            };
                            for i in 0..4 {
                                unsafe { *out.get().add((b * cfg().topk + ei) * cfg().d + qd * 4 + i) = ds[i] + ly.down_b[e * cfg().d + qd * 4 + i] };
                            }
                        }
                    }
                });
                }
            }
            for b in 0..bsz {
                for ei in 0..cfg().topk {
                    let w = wtss[b][ei];
                    for kx in 0..cfg().d {
                        xs[b][kx] += w * ffn4[(b * cfg().topk + ei) * cfg().d + kx];
                    }
                }
            }
        }
        let xn = rmsnorm(&xs[bsz - 1], &out_norm);
        let (xq2, xs2) = quant_i8(&xn);
        let mut logits = vec![0f32; cfg().nvocab];
        {
            let lp = logits.as_mut_ptr() as usize;
            let chunk = (cfg().nvocab + NT - 1) / NT;
            std::thread::scope(|sc| {
                for t in 0..NT {
                    let a = t * chunk;
                    let b2 = ((t + 1) * chunk).min(cfg().nvocab);
                    let (xq2, xs2, head) = (&xq2, &xs2, &head);
                    sc.spawn(move || {
                        for r in a..b2 {
                            let l = unsafe { head.dot(r, xq2.as_ptr(), xs2.as_ptr()) };
                            unsafe { *(lp as *mut f32).add(r) = l };
                        }
                    });
                }
            });
        }
        sample_logits(&logits)
    };

    pool_init();
    let serve = prompt == "--serve";
    let prompts: Box<dyn Iterator<Item = String>> = if serve {
        eprintln!("[READY]");
        use std::io::BufRead;
        Box::new(std::io::stdin().lock().lines().map_while(Result::ok))
    } else {
        Box::new(std::iter::once(prompt.clone()))
    };
    for pline in prompts {
        if pline.trim().is_empty() {
            continue;
        }
        // Serve protocol: either a bare prompt line, or TSV `temp \t seed \t ngen \t prompt`
        // where prompt has newlines escaped as \n and tabs as \t (backslash-escaped).
        let (req_ngen, pline) = {
            let parts: Vec<&str> = pline.splitn(4, '\t').collect();
            if parts.len() == 4 {
                if let (Ok(temp), Ok(seed), Ok(ngen)) = (
                    parts[0].parse::<f32>(),
                    parts[1].parse::<u64>(),
                    parts[2].parse::<usize>(),
                ) {
                    use std::sync::atomic::Ordering::Relaxed;
                    TEMP_MILLI.store((temp.max(0.0) * 1000.0) as u32, Relaxed);
                    // splitmix64 scramble so adjacent seeds diverge; never zero
                    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                    RNG_STATE.store((z ^ (z >> 31)) | 1, Relaxed);
                    let mut unescaped = String::with_capacity(parts[3].len());
                    let mut it = parts[3].chars();
                    while let Some(c) = it.next() {
                        if c == '\\' {
                            match it.next() {
                                Some('n') => unescaped.push('\n'),
                                Some('t') => unescaped.push('\t'),
                                Some('\\') => unescaped.push('\\'),
                                Some(o) => {
                                    unescaped.push('\\');
                                    unescaped.push(o)
                                }
                                None => unescaped.push('\\'),
                            }
                        } else {
                            unescaped.push(c);
                        }
                    }
                    (ngen.clamp(1, MAXSEQ), unescaped)
                } else {
                    (0, pline.clone())
                }
            } else {
                (0, pline.clone())
            }
        };
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
        ids.extend(encode(&pline));
        ids.push(200007);
        ids.push(200006);
        ids.extend(encode("assistant"));

        let t1 = Instant::now();
        let mut next = if !std::env::var("BATCH_PREFILL").is_ok() {
            let mut n = 0u32;
            for (i, &t) in ids.iter().enumerate() {
                n = forward(t, i, i == ids.len() - 1, &mut kc, &mut vc);
            }
            n
        } else {
            forward_batch(&ids, 0, &mut kc, &mut vc)
        };
        let prefill_s = t1.elapsed().as_secs_f64();

        let ngen = if req_ngen > 0 { req_ngen } else if serve { 256 } else { 96 };
        let t2 = Instant::now();
        let mut pos = ids.len();
        let mut emitted = 0usize;
        for _ in 0..ngen {
            if next == 200002 || next == 199999 || pos + 1 >= MAXSEQ {
                break;
            }
            let disp = String::from_utf8_lossy(toks[next as usize]).into_owned();
            print!("{disp}");
            emitted += 1;
            use std::io::Write;
            std::io::stdout().flush().ok();
            next = forward(next, pos, true, &mut kc, &mut vc);
            pos += 1;
        }
        let dt = t2.elapsed().as_secs_f64();
        println!(
            "\n[STATS] prefill_tok={} prefill_s={:.2} decode_tok={} decode_s={:.2} tok_s={:.1}",
            ids.len(),
            prefill_s,
            emitted,
            dt,
            emitted as f64 / dt
        );
        if serve {
            println!("[DONE]");
            use std::io::Write;
            std::io::stdout().flush().ok();
        } else {
            let total_ns: u64 = STAGE_NS.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).sum();
            for (i, name) in STAGE_NAMES.iter().enumerate() {
                let ns = STAGE_NS[i].load(std::sync::atomic::Ordering::Relaxed);
                println!("  {name:<12} {:>7.2} ms/tok  {:>4.1}%", ns as f64 / 1e6 / emitted.max(1) as f64, 100.0 * ns as f64 / total_ns.max(1) as f64);
            }
        }
    }
}
