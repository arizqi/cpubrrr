//! Verify native Q4_K / Q6_K matvec kernels (4-bit·int8, dequant-inline) against the
//! dequant-then-dot reference (which uses the gguf-verified dequant). Transitively
//! verifies the fast kernels vs the official oracle before they enter the engine.

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

/// Native Q4_K row · int8 x. xsum[blk] = sum of xq over 32-block blk (precomputed).
unsafe fn q4k_dot(row: *const u8, xq: *const i8, xs: *const f32, xsum: *const i32, cols: usize) -> f32 {
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        let mut acc = 0f32;
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
                let (n0, n1) = if j % 2 == 0 {
                    (vandq_u8(w0, mask), vandq_u8(w1, mask))
                } else {
                    (vshrq_n_u8::<4>(w0), vshrq_n_u8::<4>(w1))
                };
                let x0 = vld1q_s8(xq.add(blk * 32));
                let x1 = vld1q_s8(xq.add(blk * 32 + 16));
                let s = sdot(sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(n0), x0), vreinterpretq_s8_u8(n1), x1);
                let sumqx = vaddvq_s32(s) as f32;
                acc += *xs.add(blk) * (d * sc as f32 * sumqx - dmin * m as f32 * (*xsum.add(blk)) as f32);
            }
        }
        acc
    }
}

/// Native Q6_K row · int8 x.
unsafe fn q6k_dot(row: *const u8, xq: *const i8, xs: *const f32, cols: usize) -> f32 {
    unsafe {
        let mut acc = 0f32;
        for sb in 0..cols / 256 {
            let b = row.add(sb * 210);
            let ql = b;
            let qh = b.add(128);
            let sc = b.add(192); // i8
            let d = half_to_f32(u16::from_le_bytes([*b.add(208), *b.add(209)]));
            let mut temp = [0i8; 256];
            for half in 0..2 {
                let qlh = ql.add(half * 64);
                let qhh = qh.add(half * 32);
                for l in 0..32 {
                    let a = *qlh.add(l);
                    let a32 = *qlh.add(l + 32);
                    let ah = *qhh.add(l);
                    temp[half * 128 + l] = (((a & 0xF) | (((ah >> 0) & 3) << 4)) as i8) - 32;
                    temp[half * 128 + l + 32] = (((a32 & 0xF) | (((ah >> 2) & 3) << 4)) as i8) - 32;
                    temp[half * 128 + l + 64] = (((a >> 4) | (((ah >> 4) & 3) << 4)) as i8) - 32;
                    temp[half * 128 + l + 96] = (((a32 >> 4) | (((ah >> 6) & 3) << 4)) as i8) - 32;
                }
            }
            for s16 in 0..16 {
                let blk32 = sb * 8 + s16 / 2;
                let t = vld1q_s8(temp.as_ptr().add(s16 * 16));
                let x = vld1q_s8(xq.add(blk32 * 32 + (s16 % 2) * 16));
                let sm = vaddvq_s32(sdot(vdupq_n_s32(0), t, x)) as f32;
                acc += *xs.add(blk32) * d * (*sc.add(s16) as i8 as f32) * sm;
            }
        }
        acc
    }
}

fn hexb(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn main() {
    let js = fs::read_to_string("/tmp/qk_mv.json").unwrap();
    let get = |obj: &str, key: &str| -> String {
        let k = format!("\"{key}\":");
        let st = obj.find(&k).unwrap() + k.len();
        obj[st..].trim_start().to_string()
    };
    let arr_f32 = |obj: &str, key: &str| -> Vec<f32> {
        let k = format!("\"{key}\": [");
        let st = obj.find(&k).unwrap() + k.len();
        let e = obj[st..].find(']').unwrap();
        obj[st..st + e].split(',').filter_map(|s| s.trim().parse().ok()).collect()
    };
    let arr_i8 = |obj: &str, key: &str| -> Vec<i8> {
        let k = format!("\"{key}\": [");
        let st = obj.find(&k).unwrap() + k.len();
        let e = obj[st..].find(']').unwrap();
        obj[st..st + e].split(',').filter_map(|s| s.trim().parse().ok()).collect()
    };
    let first_hex = |obj: &str| -> Vec<u8> {
        let k = "\"rows_hex\": [";
        let st = obj.find(k).unwrap() + k.len();
        let seg = &obj[st..];
        let a = seg.find('"').unwrap() + 1;
        let b = seg[a..].find('"').unwrap();
        hexb(&seg[a..a + b])
    };

    let q4 = &js[js.find("\"q4k\":").unwrap()..js.find("\"q6k\":").unwrap()];
    let q6 = &js[js.find("\"q6k\":").unwrap()..];

    for (name, obj, is4) in [("Q4_K", q4, true), ("Q6_K", q6, false)] {
        let cols: usize = get(obj, "cols").split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap();
        let xq = arr_i8(obj, "xq");
        let xs = arr_f32(obj, "xs");
        let refv = arr_f32(obj, "ref");
        let row0 = first_hex(obj);
        let xsum: Vec<i32> = (0..cols / 32).map(|b| xq[b * 32..b * 32 + 32].iter().map(|&v| v as i32).sum()).collect();
        let got = unsafe {
            if is4 { q4k_dot(row0.as_ptr(), xq.as_ptr(), xs.as_ptr(), xsum.as_ptr(), cols) }
            else { q6k_dot(row0.as_ptr(), xq.as_ptr(), xs.as_ptr(), cols) }
        };
        let rel = ((got - refv[0]) / refv[0].abs().max(1e-6)).abs();
        println!("{name} native dot: got {got:.5}  ref {:.5}  rel {rel:.2e}  {}",
            refv[0], if rel < 1e-3 { "PASS" } else { "FAIL" });
    }
}
