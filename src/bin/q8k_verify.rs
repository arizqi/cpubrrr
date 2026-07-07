//! Verify the integer-accumulation Q4_K/Q6_K kernels (Q8_K-style per-256 activation
//! scale, int scale-mults, 2 float ops per superblock) against dequant-f64 reference.
//! This is llama.cpp's algorithmic trick, adopted. Also verifies the i8mm (smmla)
//! 2-row variant. Reference: /tmp/q8k_ref.json (real qwen weights).

use std::arch::aarch64::*;
use std::fs;

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
#[inline(always)]
unsafe fn unpack_k4(scales: *const u8) -> ([u8; 8], [u8; 8]) {
    unsafe {
        let mut u = [0u32; 4];
        std::ptr::copy_nonoverlapping(scales, u.as_mut_ptr() as *mut u8, 12);
        let (k1, k2, k3) = (0x3f3f3f3fu32, 0x0f0f0f0fu32, 0x03030303u32);
        u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
        let uaux = u[1] & k1;
        u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
        u[2] = uaux;
        u[0] &= k1;
        let mut sc = [0u8; 8]; let mut m = [0u8; 8];
        std::ptr::copy_nonoverlapping(u.as_ptr() as *const u8, sc.as_mut_ptr(), 8);
        std::ptr::copy_nonoverlapping((u.as_ptr() as *const u8).add(8), m.as_mut_ptr(), 8);
        (sc, m)
    }
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
/// smmla: 2x2 int32 matrix += (2x8 i8) x (8x2 i8): d[i][j] += sum_k a[i*8+k]*b[j*8+k]
#[inline(always)]
unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    unsafe {
        let mut r = acc;
        std::arch::asm!("smmla {r:v}.4s, {a:v}.16b, {b:v}.16b",
            r = inout(vreg) r, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
        r
    }
}

/// H1: integer-accumulation Q4_K dot. ys = per-256 activation scale; xsum32 = per-32 sums.
unsafe fn q4k_dot_int(row: *const u8, xq: *const i8, ys: *const f32, xsum32: *const i32, cols: usize) -> f32 {
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let mut sumf = 0f32;
        for sb in 0..cols / 256 {
            let b = row.add(sb * 144);
            let d = half_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
            let dmin = half_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
            let (sc, mn) = unpack_k4(b.add(4));
            let qs = b.add(16);
            let mut acc = vdupq_n_s32(0);
            for jj in 0..4 {
                let (j0, j1) = (jj * 2, jj * 2 + 1);
                let (blk0, blk1) = (sb * 8 + j0, sb * 8 + j1);
                let w0 = vld1q_u8(qs.add(jj * 32));
                let w1 = vld1q_u8(qs.add(jj * 32 + 16));
                let s0 = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(w0, mask)), vld1q_s8(xq.add(blk0 * 32))),
                              vreinterpretq_s8_u8(vandq_u8(w1, mask)), vld1q_s8(xq.add(blk0 * 32 + 16)));
                let s1 = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8::<4>(w0)), vld1q_s8(xq.add(blk1 * 32))),
                              vreinterpretq_s8_u8(vshrq_n_u8::<4>(w1)), vld1q_s8(xq.add(blk1 * 32 + 16)));
                acc = vmlaq_n_s32(acc, s0, sc[j0] as i32);
                acc = vmlaq_n_s32(acc, s1, sc[j1] as i32);
            }
            let mut mint = 0i32;
            for j in 0..8 { mint += mn[j] as i32 * *xsum32.add(sb * 8 + j); }
            let ysb = *ys.add(sb);
            sumf += ysb * d * vaddvq_s32(acc) as f32 - ysb * dmin * mint as f32;
        }
        sumf
    }
}

/// H1 for Q6_K: integer accumulation with per-16 int8 scales.
unsafe fn q6k_dot_int(row: *const u8, xq: *const i8, ys: *const f32, cols: usize) -> f32 {
    unsafe {
        let mut sumf = 0f32;
        for sb in 0..cols / 256 {
            let b = row.add(sb * 210);
            let (ql, qh, scp) = (b, b.add(128), b.add(192));
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
            let mut acc = vdupq_n_s32(0);
            for s16 in 0..16 {
                let sd = sdot(vdupq_n_s32(0), vld1q_s8(t.as_ptr().add(s16 * 16)),
                    vld1q_s8(xq.add(sb * 256 + s16 * 16)));
                acc = vmlaq_n_s32(acc, sd, *scp.add(s16) as i8 as i32);
            }
            sumf += *ys.add(sb) * d * vaddvq_s32(acc) as f32;
        }
        sumf
    }
}

/// H2: 2-row Q4_K with smmla. Processes rows r0,r1 against shared xq.
/// smmla input layout: a = [row0 8B | row1 8B], b = [x 8B | x 8B] -> acc[0]=r0·x, acc[3]=r1·x...
/// We use b = [x_lo8 | x_hi8] and a = [w_r0_lo8|w_r0_hi8]? Simpler mapping:
/// a = [w_r0(8B) , w_r1(8B)], b = [x(8B), x_next(8B)] gives cross terms; we need SAME x for both rows:
/// b = [x0_8, x1_8] (two halves of a 16-byte x chunk), a = [r0_first8, r0_second8]??
/// Cleanest: acc2x2 with a=[r0_lo8, r1_lo8], b=[x_lo8, x_lo8] wastes half.
/// Standard trick (llama.cpp): a=[r0 8B, r1 8B], b=[x 8B, x' 8B] where x' is the NEXT 8 bytes;
/// then acc = [[r0·x, r0·x'],[r1·x, r1·x']] — all 4 products useful! 32 useful MACs/instr.
unsafe fn q4k_dot2_smmla(r0: *const u8, r1: *const u8, xq: *const i8, ys: *const f32, xsum32: *const i32, cols: usize) -> (f32, f32) {
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let (mut sumf0, mut sumf1) = (0f32, 0f32);
        for sb in 0..cols / 256 {
            let b0 = r0.add(sb * 144);
            let b1 = r1.add(sb * 144);
            let d0 = half_to_f32(u16::from_le_bytes([*b0, *b0.add(1)]));
            let dm0 = half_to_f32(u16::from_le_bytes([*b0.add(2), *b0.add(3)]));
            let d1 = half_to_f32(u16::from_le_bytes([*b1, *b1.add(1)]));
            let dm1 = half_to_f32(u16::from_le_bytes([*b1.add(2), *b1.add(3)]));
            let (sc0, mn0) = unpack_k4(b0.add(4));
            let (sc1, mn1) = unpack_k4(b1.add(4));
            let (qs0, qs1) = (b0.add(16), b1.add(16));
            // per-subblock int sums for both rows via smmla
            let mut acc0 = vdupq_n_s32(0); // row0: sc-weighted
            let mut acc1 = vdupq_n_s32(0); // row1
            for jj in 0..4 {
                let (j0, j1) = (jj * 2, jj * 2 + 1);
                let (blk0, blk1) = (sb * 8 + j0, sb * 8 + j1);
                let w0a = vld1q_u8(qs0.add(jj * 32)); let w0b = vld1q_u8(qs0.add(jj * 32 + 16));
                let w1a = vld1q_u8(qs1.add(jj * 32)); let w1b = vld1q_u8(qs1.add(jj * 32 + 16));
                // subblock j0 (lo nibbles): rows interleaved for smmla
                // a-reg = [r0_lo(first 8 of 16B), r1_lo(first 8)], need zip of 8-byte halves
                let r0lo_a = vandq_u8(w0a, mask); let r0lo_b = vandq_u8(w0b, mask);
                let r1lo_a = vandq_u8(w1a, mask); let r1lo_b = vandq_u8(w1b, mask);
                let x0 = vld1q_s8(xq.add(blk0 * 32));
                let x0b = vld1q_s8(xq.add(blk0 * 32 + 16));
                // m0 = [r0lo_a lo8 | r1lo_a lo8]; mb pairs with [x lo8 | x hi8]
                let a01 = vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(r0lo_a), vget_low_u8(r1lo_a)));
                let a23 = vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(r0lo_a), vget_high_u8(r1lo_a)));
                let a45 = vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(r0lo_b), vget_low_u8(r1lo_b)));
                let a67 = vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(r0lo_b), vget_high_u8(r1lo_b)));
                let bx0 = vreinterpretq_s8_s64(vdupq_n_s64(vgetq_lane_s64(vreinterpretq_s64_s8(x0), 0)));
                let bx1 = vreinterpretq_s8_s64(vdupq_n_s64(vgetq_lane_s64(vreinterpretq_s64_s8(x0), 1)));
                let bx2 = vreinterpretq_s8_s64(vdupq_n_s64(vgetq_lane_s64(vreinterpretq_s64_s8(x0b), 0)));
                let bx3 = vreinterpretq_s8_s64(vdupq_n_s64(vgetq_lane_s64(vreinterpretq_s64_s8(x0b), 1)));
                let mut m = vdupq_n_s32(0);
                m = smmla(m, a01, bx0);
                m = smmla(m, a23, bx1);
                m = smmla(m, a45, bx2);
                m = smmla(m, a67, bx3);
                // m = [r0·x, r0·x, r1·x, r1·x]? layout: [ [a_row0·b_col0, a_row0·b_col1],[a_row1·b_col0,...] ]
                // with b duplicated cols, lanes 0,1 = r0 dot; 2,3 = r1 dot (each = half sums)
                let r0s = vgetq_lane_s32(m, 0) + vgetq_lane_s32(m, 1);
                let r1s = vgetq_lane_s32(m, 2) + vgetq_lane_s32(m, 3);
                acc0 = vsetq_lane_s32(vgetq_lane_s32(acc0, 0) + r0s * sc0[j0] as i32, acc0, 0);
                acc1 = vsetq_lane_s32(vgetq_lane_s32(acc1, 0) + r1s * sc1[j0] as i32, acc1, 0);
                // subblock j1 (hi nibbles)
                let r0hi_a = vshrq_n_u8::<4>(w0a); let r0hi_b = vshrq_n_u8::<4>(w0b);
                let r1hi_a = vshrq_n_u8::<4>(w1a); let r1hi_b = vshrq_n_u8::<4>(w1b);
                let x1 = vld1q_s8(xq.add(blk1 * 32));
                let x1b = vld1q_s8(xq.add(blk1 * 32 + 16));
                let c01 = vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(r0hi_a), vget_low_u8(r1hi_a)));
                let c23 = vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(r0hi_a), vget_high_u8(r1hi_a)));
                let c45 = vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(r0hi_b), vget_low_u8(r1hi_b)));
                let c67 = vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(r0hi_b), vget_high_u8(r1hi_b)));
                let cx0 = vreinterpretq_s8_s64(vdupq_n_s64(vgetq_lane_s64(vreinterpretq_s64_s8(x1), 0)));
                let cx1 = vreinterpretq_s8_s64(vdupq_n_s64(vgetq_lane_s64(vreinterpretq_s64_s8(x1), 1)));
                let cx2 = vreinterpretq_s8_s64(vdupq_n_s64(vgetq_lane_s64(vreinterpretq_s64_s8(x1b), 0)));
                let cx3 = vreinterpretq_s8_s64(vdupq_n_s64(vgetq_lane_s64(vreinterpretq_s64_s8(x1b), 1)));
                let mut m2 = vdupq_n_s32(0);
                m2 = smmla(m2, c01, cx0);
                m2 = smmla(m2, c23, cx1);
                m2 = smmla(m2, c45, cx2);
                m2 = smmla(m2, c67, cx3);
                let r0s2 = vgetq_lane_s32(m2, 0) + vgetq_lane_s32(m2, 1);
                let r1s2 = vgetq_lane_s32(m2, 2) + vgetq_lane_s32(m2, 3);
                acc0 = vsetq_lane_s32(vgetq_lane_s32(acc0, 0) + r0s2 * sc0[j1] as i32, acc0, 0);
                acc1 = vsetq_lane_s32(vgetq_lane_s32(acc1, 0) + r1s2 * sc1[j1] as i32, acc1, 0);
            }
            let (mut mint0, mut mint1) = (0i32, 0i32);
            for j in 0..8 {
                let xs = *xsum32.add(sb * 8 + j);
                mint0 += mn0[j] as i32 * xs;
                mint1 += mn1[j] as i32 * xs;
            }
            let ysb = *ys.add(sb);
            sumf0 += ysb * d0 * vgetq_lane_s32(acc0, 0) as f32 - ysb * dm0 * mint0 as f32;
            sumf1 += ysb * d1 * vgetq_lane_s32(acc1, 0) as f32 - ysb * dm1 * mint1 as f32;
        }
        (sumf0, sumf1)
    }
}

fn hexb(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}
fn arr_f32(o: &str, k: &str) -> Vec<f32> {
    let key = format!("\"{k}\": [");
    let st = o.find(&key).unwrap() + key.len();
    let e = o[st..].find(']').unwrap();
    o[st..st + e].split(',').filter_map(|s| s.trim().parse().ok()).collect()
}
fn arr_i(o: &str, k: &str) -> Vec<i32> {
    let key = format!("\"{k}\": [");
    let st = o.find(&key).unwrap() + key.len();
    let e = o[st..].find(']').unwrap();
    o[st..st + e].split(',').filter_map(|s| s.trim().parse().ok()).collect()
}
fn rows(o: &str) -> Vec<Vec<u8>> {
    let key = "\"rows_hex\": [";
    let st = o.find(key).unwrap() + key.len();
    let e = o[st..].find(']').unwrap();
    o[st..st + e].split(',').map(|s| hexb(s.trim().trim_matches('"'))).collect()
}

fn main() {
    let js = fs::read_to_string("/tmp/q8k_ref.json").unwrap();
    let q4 = &js[js.find("\"q4k\":").unwrap()..js.find("\"q6k\":").unwrap()];
    let q6 = &js[js.find("\"q6k\":").unwrap()..];

    for (name, obj, is4) in [("Q4_K int-accum", q4, true), ("Q6_K int-accum", q6, false)] {
        let xq: Vec<i8> = arr_i(obj, "xq").iter().map(|&v| v as i8).collect();
        let ys = arr_f32(obj, "ys");
        let xsum = arr_i(obj, "xsum32");
        let refs = arr_f32(obj, "ref");
        let rws = rows(obj);
        let cols = 2048;
        let got = unsafe {
            if is4 { q4k_dot_int(rws[0].as_ptr(), xq.as_ptr(), ys.as_ptr(), xsum.as_ptr(), cols) }
            else { q6k_dot_int(rws[0].as_ptr(), xq.as_ptr(), ys.as_ptr(), cols) }
        };
        let rel = ((got - refs[0]) / refs[0].abs().max(1e-6)).abs();
        println!("{name}: got {got:.5} ref {:.5} rel {rel:.2e} {}", refs[0], if rel < 1e-3 { "PASS" } else { "FAIL" });
    }
    // smmla 2-row
    {
        let xq: Vec<i8> = arr_i(q4, "xq").iter().map(|&v| v as i8).collect();
        let ys = arr_f32(q4, "ys");
        let xsum = arr_i(q4, "xsum32");
        let refs = arr_f32(q4, "ref");
        let rws = rows(q4);
        let (g0, g1) = unsafe { q4k_dot2_smmla(rws[0].as_ptr(), rws[1].as_ptr(), xq.as_ptr(), ys.as_ptr(), xsum.as_ptr(), 2048) };
        let r0 = ((g0 - refs[0]) / refs[0].abs().max(1e-6)).abs();
        let r1 = ((g1 - refs[1]) / refs[1].abs().max(1e-6)).abs();
        println!("Q4_K smmla-2row: r0 {g0:.5}/{:.5} rel {r0:.2e} | r1 {g1:.5}/{:.5} rel {r1:.2e} {}",
            refs[0], refs[1], if r0 < 1e-3 && r1 < 1e-3 { "PASS" } else { "FAIL" });
    }
}
