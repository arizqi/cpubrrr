//! Qwen3-MoE engine v2 — worker-driven execution model (rewrite for core saturation).
//! 12 persistent workers run the ENTIRE forward pass themselves, syncing at cheap
//! sense-reversing spin-barriers (~1us, all arrive together) instead of condvar
//! fork-join (~35us wakeup/stage). Workers park only ONCE per token, not per stage.
//! All "serial glue" (rmsnorm, quant, router) is parallelized too -> no idle cores.
//! Kernels (q4k_dot/q6k_dot) are the verified ones from engine_qwen/qk_mv_verify.
//! Usage: engine_qwen2 <data-dir> <blob> [prompt]

use std::arch::aarch64::*;
use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const NT: usize = 12;
const MAXSEQ: usize = 4096;
static TM: [AtomicU64; 5] = [AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0)];

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
        let mut f = [vdupq_n_f32(0.0); 4];
        let mut minacc = 0f32;
        for sb in 0..cols / 256 {
            let b = row.add(sb * 144);
            let d = half_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
            let dmin = half_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
            let scales = std::slice::from_raw_parts(b.add(4), 12);
            let qs = b.add(16);
            // precompute 8 (d*sc, dmin*m) once per superblock (hoist branchy extraction)
            let mut dsc = [0f32; 8]; let mut dm = [0f32; 8];
            for j in 0..8 { let (sc, m) = scale_min_k4(j, scales); dsc[j] = d * sc as f32; dm[j] = dmin * m as f32; }
            for jj in 0..4 {
                let (j0, j1) = (jj * 2, jj * 2 + 1);
                let (blk0, blk1) = (sb * 8 + j0, sb * 8 + j1);
                let qb = qs.add(jj * 32);
                let w0 = vld1q_u8(qb);
                let w1 = vld1q_u8(qb.add(16));
                let s0 = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(w0, mask)), vld1q_s8(xq.add(blk0 * 32))),
                              vreinterpretq_s8_u8(vandq_u8(w1, mask)), vld1q_s8(xq.add(blk0 * 32 + 16)));
                let s1 = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8::<4>(w0)), vld1q_s8(xq.add(blk1 * 32))),
                              vreinterpretq_s8_u8(vshrq_n_u8::<4>(w1)), vld1q_s8(xq.add(blk1 * 32 + 16)));
                let (xs0, xs1) = (*xs.add(blk0), *xs.add(blk1));
                f[0] = vfmaq_n_f32(f[0], vcvtq_f32_s32(s0), xs0 * dsc[j0]);
                f[1] = vfmaq_n_f32(f[1], vcvtq_f32_s32(s1), xs1 * dsc[j1]);
                minacc += xs0 * dm[j0] * *xsum.add(blk0) as f32 + xs1 * dm[j1] * *xsum.add(blk1) as f32;
            }
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(f[0], f[1]), vaddq_f32(f[2], f[3]))) - minacc
    }
}
/// Fused gate+up Q4_K: shared xq loads, 2 independent weight streams (ILP). -> (gate, up)
unsafe fn q4k_gu(g: *const u8, u: *const u8, xq: *const i8, xs: *const f32, xsum: *const i32, cols: usize) -> (f32, f32) {
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let mut fg = [vdupq_n_f32(0.0); 2];
        let mut fu = [vdupq_n_f32(0.0); 2];
        let (mut mg, mut mu) = (0f32, 0f32);
        for sb in 0..cols / 256 {
            let gb = g.add(sb * 144); let ub = u.add(sb * 144);
            let gd = half_to_f32(u16::from_le_bytes([*gb, *gb.add(1)])); let gdm = half_to_f32(u16::from_le_bytes([*gb.add(2), *gb.add(3)]));
            let ud = half_to_f32(u16::from_le_bytes([*ub, *ub.add(1)])); let udm = half_to_f32(u16::from_le_bytes([*ub.add(2), *ub.add(3)]));
            let gsc = std::slice::from_raw_parts(gb.add(4), 12); let usc = std::slice::from_raw_parts(ub.add(4), 12);
            let (gqs, uqs) = (gb.add(16), ub.add(16));
            for jj in 0..4 {
                let (j0, j1) = (jj * 2, jj * 2 + 1);
                let (blk0, blk1) = (sb * 8 + j0, sb * 8 + j1);
                let x0a = vld1q_s8(xq.add(blk0 * 32)); let x0b = vld1q_s8(xq.add(blk0 * 32 + 16));
                let x1a = vld1q_s8(xq.add(blk1 * 32)); let x1b = vld1q_s8(xq.add(blk1 * 32 + 16));
                let (xs0, xs1) = (*xs.add(blk0), *xs.add(blk1));
                let (xm0, xm1) = (*xsum.add(blk0) as f32, *xsum.add(blk1) as f32);
                // gate
                let (gs0, gm0) = scale_min_k4(j0, gsc); let (gs1, gm1) = scale_min_k4(j1, gsc);
                let gw0 = vld1q_u8(gqs.add(jj * 32)); let gw1 = vld1q_u8(gqs.add(jj * 32 + 16));
                let gd0 = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(gw0, mask)), x0a), vreinterpretq_s8_u8(vandq_u8(gw1, mask)), x0b);
                let gd1 = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8::<4>(gw0)), x1a), vreinterpretq_s8_u8(vshrq_n_u8::<4>(gw1)), x1b);
                fg[0] = vfmaq_n_f32(fg[0], vcvtq_f32_s32(gd0), xs0 * gd * gs0 as f32);
                fg[1] = vfmaq_n_f32(fg[1], vcvtq_f32_s32(gd1), xs1 * gd * gs1 as f32);
                mg += xs0 * gdm * gm0 as f32 * xm0 + xs1 * gdm * gm1 as f32 * xm1;
                // up
                let (us0, um0) = scale_min_k4(j0, usc); let (us1, um1) = scale_min_k4(j1, usc);
                let uw0 = vld1q_u8(uqs.add(jj * 32)); let uw1 = vld1q_u8(uqs.add(jj * 32 + 16));
                let ud0 = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(uw0, mask)), x0a), vreinterpretq_s8_u8(vandq_u8(uw1, mask)), x0b);
                let ud1 = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8::<4>(uw0)), x1a), vreinterpretq_s8_u8(vshrq_n_u8::<4>(uw1)), x1b);
                fu[0] = vfmaq_n_f32(fu[0], vcvtq_f32_s32(ud0), xs0 * ud * us0 as f32);
                fu[1] = vfmaq_n_f32(fu[1], vcvtq_f32_s32(ud1), xs1 * ud * us1 as f32);
                mu += xs0 * udm * um0 as f32 * xm0 + xs1 * udm * um1 as f32 * xm1;
            }
        }
        (vaddvq_f32(vaddq_f32(fg[0], fg[1])) - mg, vaddvq_f32(vaddq_f32(fu[0], fu[1])) - mu)
    }
}
unsafe fn q6k_dot(row: *const u8, xq: *const i8, xs: *const f32, cols: usize) -> f32 {
    unsafe {
        let mut f = [vdupq_n_f32(0.0); 4];
        for sb in 0..cols / 256 {
            let b = row.add(sb * 210);
            let (ql, qh, sc) = (b, b.add(128), b.add(192));
            let d = half_to_f32(u16::from_le_bytes([*b.add(208), *b.add(209)]));
            let mut t = [0i8; 256];
            let m0f = vdupq_n_u8(0x0F); let m3 = vdupq_n_u8(0x03); let n32 = vdupq_n_s8(32);
            for h in 0..2 {
                let (qlh, qhh) = (ql.add(h * 64), qh.add(h * 32));
                let tb = t.as_mut_ptr().add(h * 128);
                for l16 in 0..2 {
                    let off = l16 * 16;
                    let qla = vld1q_u8(qlh.add(off));
                    let qlb = vld1q_u8(qlh.add(off + 32));
                    let qhv = vld1q_u8(qhh.add(off));
                    let h0 = vshlq_n_u8::<4>(vandq_u8(qhv, m3));
                    let h1 = vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qhv), m3));
                    let h2 = vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qhv), m3));
                    let h3 = vshlq_n_u8::<4>(vshrq_n_u8::<6>(qhv));
                    vst1q_s8(tb.add(off), vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qla, m0f), h0)), n32));
                    vst1q_s8(tb.add(off + 32), vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qlb, m0f), h1)), n32));
                    vst1q_s8(tb.add(off + 64), vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(qla), h2)), n32));
                    vst1q_s8(tb.add(off + 96), vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(qlb), h3)), n32));
                }
            }
            for s16 in 0..16 {
                let blk = sb * 8 + s16 / 2;
                let sd = sdot(vdupq_n_s32(0), vld1q_s8(t.as_ptr().add(s16 * 16)), vld1q_s8(xq.add(blk * 32 + (s16 % 2) * 16)));
                f[s16 & 3] = vfmaq_n_f32(f[s16 & 3], vcvtq_f32_s32(sd), *xs.add(blk) * d * (*sc.add(s16) as i8 as f32));
            }
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(f[0], f[1]), vaddq_f32(f[2], f[3])))
    }
}

#[derive(Clone, Copy)]
struct KMat { ptr: *const u8, ttype: u32, cols: usize, bpr: usize }
unsafe impl Send for KMat {} unsafe impl Sync for KMat {}
impl KMat {
    #[inline(always)]
    unsafe fn dot(&self, row: usize, xq: *const i8, xs: *const f32, xsum: *const i32) -> f32 {
        unsafe {
            let r = self.ptr.add(row * self.bpr);
            match self.ttype { 12 => q4k_dot(r, xq, xs, xsum, self.cols), 14 => q6k_dot(r, xq, xs, self.cols), _ => 0.0 }
        }
    }
}

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

// ---- raw shared buffers (workers write disjoint slices) ----
#[derive(Clone, Copy)]
struct Buf { f: *mut f32, i: *mut i8, w: *mut i32 }
unsafe impl Send for Buf {} unsafe impl Sync for Buf {}
struct F(*mut f32); unsafe impl Send for F {} unsafe impl Sync for F {}
struct I8(*mut i8); unsafe impl Send for I8 {} unsafe impl Sync for I8 {}
struct I32(*mut i32); unsafe impl Send for I32 {} unsafe impl Sync for I32 {}

/// worker's contiguous chunk of n items
#[inline(always)]
fn sl(wid: usize, n: usize) -> (usize, usize) {
    let base = n / NT;
    let rem = n % NT;
    let a = wid * base + wid.min(rem);
    let sz = base + if wid < rem { 1 } else { 0 };
    (a, a + sz)
}

struct Shared {
    // control
    tseq: AtomicU64, ready: AtomicU64, cur_tok: AtomicU32, cur_pos: AtomicUsize, result: AtomicU32,
    // barrier
    bcount: AtomicUsize, bsense: AtomicBool,
    // park (between tokens only)
    m: std::sync::Mutex<()>, cv: std::sync::Condvar, parked: AtomicUsize,
    dm: std::sync::Mutex<()>, dcv: std::sync::Condvar,
    // model
    layers: &'static [Layer], out_norm: &'static [f32], head: KMat, tok_embd_off: usize, blob: *const u8,
    // activation buffers
    x: *mut f32, xn: *mut f32, xq: *mut i8, xs: *mut f32, xsum: *mut i32,
    q: *mut f32, k: *mut f32, v: *mut f32, ao: *mut f32,
    aq: *mut i8, asc: *mut f32, asm: *mut i32,
    hbuf: *mut f32, hq: *mut i8, hs: *mut f32, hm: *mut i32,
    logits: *mut f32, partials: *mut f32, apart: *mut u64,
    kc: *mut f32, vc: *mut f32,   // [nl][MAXSEQ*nkv*hd] flattened
}
unsafe impl Send for Shared {} unsafe impl Sync for Shared {}

#[inline(always)]
fn bar(sh: &Shared, ls: &mut bool) {
    *ls = !*ls;
    if sh.bcount.fetch_add(1, Ordering::AcqRel) == NT - 1 {
        sh.bcount.store(0, Ordering::Relaxed);
        sh.bsense.store(*ls, Ordering::Release);
    } else {
        while sh.bsense.load(Ordering::Acquire) != *ls { std::hint::spin_loop(); }
    }
}

fn qblock(xn: *const f32, xq: *mut i8, xs: *mut f32, xsum: *mut i32, bl: usize) {
    unsafe {
        let mut amax = 1e-12f32;
        for i in 0..32 { amax = amax.max((*xn.add(bl * 32 + i)).abs()); }
        let sc = amax / 127.0;
        *xs.add(bl) = sc;
        let mut acc = 0i32;
        for i in 0..32 { let qi = (*xn.add(bl * 32 + i) / sc).round().clamp(-127.0, 127.0) as i8; *xq.add(bl * 32 + i) = qi; acc += qi as i32; }
        *xsum.add(bl) = acc;
    }
}

fn forward_worker(sh: &Shared, wid: usize, ls: &mut bool) {
    let c = *cfg();
    let (d, nh, nkv, hd, ff, ne, topk) = (c.d, c.nh, c.nkv, c.hd, c.ffexp, c.ne, c.topk);
    let nb = d / 32;
    let pos = sh.cur_pos.load(Ordering::Relaxed);
    let tokid = sh.cur_tok.load(Ordering::Relaxed);
    unsafe {
        // ---- embed (worker slice of d, dequant Q4_K embedding row) ----
        {
            let bpr = d / 256 * 144;
            let row = sh.blob.add(sh.tok_embd_off + tokid as usize * bpr);
            let (a, b) = sl(wid, d / 256);
            for sb in a..b {
                let bb = row.add(sb * 144);
                let dd = half_to_f32(u16::from_le_bytes([*bb, *bb.add(1)]));
                let dmin = half_to_f32(u16::from_le_bytes([*bb.add(2), *bb.add(3)]));
                let scales = std::slice::from_raw_parts(bb.add(4), 12);
                let qs = bb.add(16);
                let (mut y, mut is, mut qo) = (0usize, 0usize, 0usize);
                while y < 256 {
                    let (s1, m1) = scale_min_k4(is, scales);
                    let (s2, m2) = scale_min_k4(is + 1, scales);
                    for l in 0..32 { *sh.x.add(sb * 256 + y + l) = dd * s1 as f32 * (*qs.add(qo + l) & 0xF) as f32 - dmin * m1 as f32; }
                    for l in 0..32 { *sh.x.add(sb * 256 + y + 32 + l) = dd * s2 as f32 * (*qs.add(qo + l) >> 4) as f32 - dmin * m2 as f32; }
                    y += 64; is += 2; qo += 32;
                }
            }
        }
        bar(sh, ls);

        for il in 0..c.nl {
            let ly = &sh.layers[il];
            let kcl = sh.kc.add(il * MAXSEQ * nkv * hd);
            let vcl = sh.vc.add(il * MAXSEQ * nkv * hd);
            // ---- attn rmsnorm: parallel reduction then normalize ----
            rmsnorm_par(sh, wid, ls, sh.x, sh.xn, ly.attn_norm.as_ptr(), d);
            // quant xn
            { let (a, b) = sl(wid, nb); for bl in a..b { qblock(sh.xn, sh.xq, sh.xs, sh.xsum, bl); } }
            bar(sh, ls);
            // qkv matvec
            { let nr = nh * hd + 2 * nkv * hd; let (a, b) = sl(wid, nr);
              for r in a..b {
                if r < nh * hd { *sh.q.add(r) = ly.wq.dot(r, sh.xq, sh.xs, sh.xsum); }
                else if r < nh * hd + nkv * hd { let rr = r - nh * hd; *sh.k.add(rr) = ly.wk.dot(rr, sh.xq, sh.xs, sh.xsum); }
                else { let rr = r - nh * hd - nkv * hd; *sh.v.add(rr) = ly.wv.dot(rr, sh.xq, sh.xs, sh.xsum); }
              } }
            bar(sh, ls);
            // qk-norm + rope + kv-cache write (parallel over nh+nkv heads)
            { let (a, b) = sl(wid, nh + nkv);
              for h in a..b {
                if h < nh { head_norm_rope(sh.q.add(h * hd), ly.q_norm.as_ptr(), hd, pos, c.rope_base, c.rms_eps); }
                else { let hh = h - nh;
                    head_norm_rope(sh.k.add(hh * hd), ly.k_norm.as_ptr(), hd, pos, c.rope_base, c.rms_eps);
                    for j in 0..hd { *kcl.add((pos * nkv + hh) * hd + j) = *sh.k.add(hh * hd + j); *vcl.add((pos * nkv + hh) * hd + j) = *sh.v.add(hh * hd + j); }
                }
              } }
            bar(sh, ls);
            // attention (parallel over nh q-heads)
            { let scale = 1.0 / (hd as f32).sqrt(); let (a, b) = sl(wid, nh);
              for h in a..b {
                let kvh = h / (nh / nkv);
                let mut sc = [0f32; MAXSEQ];
                for t in 0..=pos { let mut acc = 0.0; for j in 0..hd { acc += *kcl.add((t * nkv + kvh) * hd + j) * *sh.q.add(h * hd + j); } sc[t] = acc * scale; }
                let mx = sc[..=pos].iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0.0; for t in 0..=pos { sc[t] = (sc[t] - mx).exp(); den += sc[t]; }
                for j in 0..hd { let mut acc = 0.0; for t in 0..=pos { acc += sc[t] * *vcl.add((t * nkv + kvh) * hd + j); } *sh.ao.add(h * hd + j) = acc / den; }
              } }
            bar(sh, ls);
            // quant ao
            { let (a, b) = sl(wid, (nh * hd) / 32); for bl in a..b { qblock(sh.ao, sh.aq, sh.asc, sh.asm, bl); } }
            bar(sh, ls);
            // o-proj + residual
            { let (a, b) = sl(wid, d); for r in a..b { *sh.x.add(r) += ly.wo.dot(r, sh.aq, sh.asc, sh.asm); } }
            bar(sh, ls);
            // ffn rmsnorm
            rmsnorm_par(sh, wid, ls, sh.x, sh.xn, ly.ffn_norm.as_ptr(), d);
            { let (a, b) = sl(wid, nb); for bl in a..b { qblock(sh.xn, sh.xq, sh.xs, sh.xsum, bl); } }
            bar(sh, ls);
            // router logits (parallel over experts)
            { let (a, b) = sl(wid, ne); for e in a..b { let mut acc = 0.0; for j in 0..d { acc += ly.gate_inp[e * d + j] * *sh.xn.add(j); } *sh.logits.add(e) = acc; } }
            bar(sh, ls);
            // top-k (redundant per worker, cheap; no barrier)
            let logits = std::slice::from_raw_parts(sh.logits, ne);
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let lmax = (0..topk).map(|i| logits[order[i]]).fold(f32::MIN, f32::max);
            let mut wsum = 0f32; let mut ex = vec![0f32; topk];
            for i in 0..topk { ex[i] = (logits[order[i]] - lmax).exp(); wsum += ex[i]; }
            let top: Vec<usize> = order[..topk].to_vec();
            let wts: Vec<f32> = (0..topk).map(|i| ex[i] / wsum).collect();
            let _tg = if wid==0 { Some(Instant::now()) } else { None };
            // gate/up/silu (parallel over topk*ff)
            { let (a, b) = sl(wid, topk * ff);
              for it in a..b { let (ei, r) = (it / ff, it % ff); let gr = top[ei] * ff + r;
                let (g, u) = q4k_gu(ly.gate.ptr.add(gr * ly.gate.bpr), ly.up.ptr.add(gr * ly.up.bpr), sh.xq, sh.xs, sh.xsum, ly.gate.cols);
                *sh.hbuf.add(it) = (g / (1.0 + (-g).exp())) * u; } }
            bar(sh, ls);
            if let Some(t)=_tg { TM[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
            let _td = if wid==0 { Some(Instant::now()) } else { None };
            // quant hbuf per expert (parallel over topk*ff/32 blocks)
            { let (a, b) = sl(wid, topk * ff / 32); for bl in a..b { qblock(sh.hbuf, sh.hq, sh.hs, sh.hm, bl); } }
            bar(sh, ls);
            // down + weighted accumulate (expert-major for sequential row streaming)
            { let (a, b) = sl(wid, d);
              for r in a..b { *sh.x.add(r) += 0.0; } // no-op; accumulate below
              for ei in 0..topk {
                let (hq, hs, hm) = (sh.hq.add(ei * ff), sh.hs.add(ei * (ff / 32)), sh.hm.add(ei * (ff / 32)));
                let w = wts[ei]; let base = top[ei] * d;
                for r in a..b { *sh.x.add(r) += w * ly.down.dot(base + r, hq, hs, hm); }
              } }
            bar(sh, ls);
            if let Some(t)=_td { TM[2].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
        }
        let _th = if wid==0 { Some(Instant::now()) } else { None };
        // ---- head ----
        rmsnorm_par(sh, wid, ls, sh.x, sh.xn, sh.out_norm.as_ptr(), d);
        { let (a, b) = sl(wid, nb); for bl in a..b { qblock(sh.xn, sh.xq, sh.xs, sh.xsum, bl); } }
        bar(sh, ls);
        { let (a, b) = sl(wid, c.nvocab); let mut best = (f32::MIN, 0u32);
          for r in a..b { let l = sh.head.dot(r, sh.xq, sh.xs, sh.xsum); if l > best.0 { best = (l, r as u32); } }
          *sh.apart.add(wid) = ((best.0.to_bits() as u64) << 32) | best.1 as u64; }
        bar(sh, ls);
        if wid == 0 {
            let mut best = (f32::MIN, 0u32);
            for w in 0..NT { let v = *sh.apart.add(w); let sc = f32::from_bits((v >> 32) as u32); let id = v as u32; if sc > best.0 { best = (sc, id); } }
            sh.result.store(best.1, Ordering::Relaxed);
        }
        if let Some(t)=_th { TM[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    }
}

#[inline(always)]
fn rmsnorm_par(sh: &Shared, wid: usize, ls: &mut bool, x: *mut f32, out: *mut f32, w: *const f32, d: usize) {
    unsafe {
        let (a, b) = sl(wid, d);
        let mut ps = 0f32; for i in a..b { let v = *x.add(i); ps += v * v; }
        *sh.partials.add(wid) = ps;
        bar(sh, ls);
        let mut tot = 0f32; for k in 0..NT { tot += *sh.partials.add(k); }
        let inv = 1.0 / (tot / d as f32 + cfg().rms_eps).sqrt();
        for i in a..b { *out.add(i) = *x.add(i) * inv * *w.add(i); }
        bar(sh, ls);
    }
}
#[inline(always)]
fn head_norm_rope(v: *mut f32, w: *const f32, hd: usize, pos: usize, base: f32, eps: f32) {
    unsafe {
        let mut ms = 0f32; for i in 0..hd { ms += *v.add(i) * *v.add(i); }
        let inv = 1.0 / (ms / hd as f32 + eps).sqrt();
        for i in 0..hd { *v.add(i) = *v.add(i) * inv * *w.add(i); }
        let half = hd / 2;
        for i in 0..half {
            let theta = pos as f32 * base.powf(-(2.0 * i as f32) / hd as f32);
            let (co, si) = (theta.cos(), theta.sin());
            let (aa, bb) = (*v.add(i), *v.add(half + i));
            *v.add(i) = aa * co - bb * si;
            *v.add(half + i) = bb * co + aa * si;
        }
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
    let spm: HashMap<&[u8], u32> = toks.iter().enumerate().filter(|(_, t)| t.starts_with(b"<|")).map(|(i, t)| (*t, i as u32)).collect();
    let encode = |s: &str| -> Vec<u32> {
        let b = s.as_bytes(); let mut ids = Vec::new(); let mut i = 0;
        while i < b.len() { let mut l = (b.len() - i).min(48);
            loop { if let Some(&id) = lookup.get(&b[i..i + l]) { ids.push(id); i += l; break; } l -= 1; assert!(l > 0); } }
        ids
    };
    let tok = |t: &str| spm[t.as_bytes()];

    eprintln!("loading blob...");
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
        attn_norm: raw.f32v(&format!("{pfx}.attn_norm.weight")), ffn_norm: raw.f32v(&format!("{pfx}.ffn_norm.weight")),
        q_norm: raw.f32v(&format!("{pfx}.attn_q_norm.weight")), k_norm: raw.f32v(&format!("{pfx}.attn_k_norm.weight")),
        wq: raw.km(&format!("{pfx}.attn_q.weight"), d), wk: raw.km(&format!("{pfx}.attn_k.weight"), d),
        wv: raw.km(&format!("{pfx}.attn_v.weight"), d), wo: raw.km(&format!("{pfx}.attn_output.weight"), nh * hd),
        gate_inp: raw.f32v(&format!("{pfx}.ffn_gate_inp.weight")), gate: raw.km(&format!("{pfx}.ffn_gate_exps.weight"), d),
        up: raw.km(&format!("{pfx}.ffn_up_exps.weight"), d), down: raw.km(&format!("{pfx}.ffn_down_exps.weight"), ff),
    }}).collect();
    let tok_embd_off = raw.idx["token_embd.weight"].1;
    let out_norm = raw.f32v("output_norm.weight");
    let head = raw.km("output.weight", d);
    let blob_ptr = raw.blob.as_ptr();
    // leak model to 'static
    let layers: &'static [Layer] = Box::leak(layers.into_boxed_slice());
    let out_norm: &'static [f32] = Box::leak(out_norm.into_boxed_slice());
    let raw: &'static Raw = Box::leak(Box::new(raw));
    let _ = raw;
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // allocate activation buffers (leaked, reused across tokens)
    let mk_f = |n: usize| Box::leak(vec![0f32; n].into_boxed_slice()).as_mut_ptr();
    let mk_i8 = |n: usize| Box::leak(vec![0i8; n].into_boxed_slice()).as_mut_ptr();
    let mk_i32 = |n: usize| Box::leak(vec![0i32; n].into_boxed_slice()).as_mut_ptr();
    let mk_u64 = |n: usize| Box::leak(vec![0u64; n].into_boxed_slice()).as_mut_ptr();
    let nb = d / 32;
    let sh: &'static Shared = Box::leak(Box::new(Shared {
        tseq: AtomicU64::new(0), ready: AtomicU64::new(0), cur_tok: AtomicU32::new(0), cur_pos: AtomicUsize::new(0), result: AtomicU32::new(0),
        bcount: AtomicUsize::new(0), bsense: AtomicBool::new(false),
        m: std::sync::Mutex::new(()), cv: std::sync::Condvar::new(), parked: AtomicUsize::new(0),
        dm: std::sync::Mutex::new(()), dcv: std::sync::Condvar::new(),
        layers, out_norm, head, tok_embd_off, blob: blob_ptr,
        x: mk_f(d), xn: mk_f(d), xq: mk_i8(d), xs: mk_f(nb), xsum: mk_i32(nb),
        q: mk_f(nh * hd), k: mk_f(nkv * hd), v: mk_f(nkv * hd), ao: mk_f(nh * hd),
        aq: mk_i8(nh * hd), asc: mk_f((nh * hd) / 32), asm: mk_i32((nh * hd) / 32),
        hbuf: mk_f(c.topk * ff), hq: mk_i8(c.topk * ff), hs: mk_f(c.topk * ff / 32), hm: mk_i32(c.topk * ff / 32),
        logits: mk_f(c.ne), partials: mk_f(NT), apart: mk_u64(NT),
        kc: mk_f(c.nl * MAXSEQ * nkv * hd), vc: mk_f(c.nl * MAXSEQ * nkv * hd),
    }));

    // spawn workers
    for wid in 0..NT {
        std::thread::spawn(move || {
            unsafe extern "C" { fn pthread_set_qos_class_self_np(q: u32, p: i32) -> i32; }
            unsafe { pthread_set_qos_class_self_np(0x21, 0); } // USER_INTERACTIVE -> P-cores
            let mut ls = false;
            let mut seen = 0u64;
            loop {
                { let mut g = sh.m.lock().unwrap();
                  while sh.tseq.load(Ordering::Acquire) == seen { g = sh.cv.wait(g).unwrap(); }
                  seen = sh.tseq.load(Ordering::Acquire); }
                forward_worker(sh, wid, &mut ls);
                if wid == 0 { let _g = sh.dm.lock().unwrap(); sh.ready.store(seen, Ordering::Release); sh.dcv.notify_one(); }
            }
        });
    }
    let run = |t: u32, pos: usize| -> u32 {
        sh.cur_tok.store(t, Ordering::Relaxed);
        sh.cur_pos.store(pos, Ordering::Relaxed);
        let s = sh.tseq.load(Ordering::Relaxed) + 1;
        { let _g = sh.m.lock().unwrap(); sh.tseq.store(s, Ordering::Release); sh.cv.notify_all(); }
        { let mut g = sh.dm.lock().unwrap(); while sh.ready.load(Ordering::Acquire) < s { g = sh.dcv.wait(g).unwrap(); } }
        sh.result.load(Ordering::Relaxed)
    };

    let mut ids: Vec<u32> = Vec::new();
    ids.push(tok("<|im_start|>")); ids.extend(encode("system\nYou are a helpful assistant.")); ids.push(tok("<|im_end|>"));
    ids.push(tok("<|im_start|>")); ids.extend(encode(&format!("\nuser\n{prompt}"))); ids.push(tok("<|im_end|>"));
    ids.push(tok("<|im_start|>")); ids.extend(encode("\nassistant\n"));
    eprintln!("prompt: {} tokens", ids.len());
    let im_end = tok("<|im_end|>");
    let t1 = Instant::now();
    let mut next = 0u32;
    for (i, &t) in ids.iter().enumerate() { next = run(t, i); let _ = i; }
    eprintln!("prefill {} tok in {:.2}s", ids.len(), t1.elapsed().as_secs_f64());
    let ngen = 64;
    let t2 = Instant::now();
    let mut pos = ids.len();
    let mut n = 0;
    for _ in 0..ngen {
        if next == im_end || next as usize >= ntok { break; }
        print!("{}", String::from_utf8_lossy(toks[next as usize]));
        use std::io::Write; std::io::stdout().flush().ok();
        next = run(next, pos); pos += 1; n += 1;
    }
    let dt = t2.elapsed().as_secs_f64();
    println!("\n---\ndecode: {n} tokens in {dt:.2}s = {:.1} tok/s", n as f64 / dt);
    let nm=["","gate/up","down","head",""];
    for i in [1,2,3] { println!("  {:<8} {:.2} ms/tok", nm[i], TM[i].load(Ordering::Relaxed) as f64/1e6/n as f64); }
}
