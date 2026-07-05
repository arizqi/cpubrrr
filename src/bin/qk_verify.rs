//! Verify cpubrrr's Q4_K / Q6_K dequant against the official `gguf` library oracle.
//! Reference blocks + expected values exported to /tmp/qk_ref.json by a gguf python call.

use std::fs;

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = -1i32;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x400 != 0 {
                    break;
                }
            }
            let mant = (m & 0x3ff) << 13;
            (sign << 31) | (((127 - 15 - e) as u32) << 23) | mant
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequant one Q4_K superblock (144 bytes) -> 256 f32.
fn dequant_q4k(b: &[u8], out: &mut [f32]) {
    let d = half_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let dmin = half_to_f32(u16::from_le_bytes([b[2], b[3]]));
    let scales = &b[4..16];
    let qs = &b[16..144];
    let mut y = 0;
    let mut is = 0;
    let mut qoff = 0;
    while y < 256 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
        for l in 0..32 {
            out[y + l] = d1 * (qs[qoff + l] & 0xF) as f32 - mm1;
        }
        for l in 0..32 {
            out[y + 32 + l] = d2 * (qs[qoff + l] >> 4) as f32 - mm2;
        }
        y += 64;
        is += 2;
        qoff += 32;
    }
}

/// Dequant one Q6_K superblock (210 bytes) -> 256 f32.
fn dequant_q6k(b: &[u8], out: &mut [f32]) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208]; // int8
    let d = half_to_f32(u16::from_le_bytes([b[208], b[209]]));
    let sci = |i: usize| sc[i] as i8 as f32;
    let mut base = 0;
    let mut yb = 0;
    while yb < 256 {
        let ql = &ql[base..];
        let qh = &qh[base / 2..];
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
        base += 64;
        yb += 128;
    }
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn field<'a>(json: &'a str, key: &str) -> &'a str {
    let k = format!("\"{key}\":");
    let start = json.find(&k).unwrap() + k.len();
    let rest = json[start..].trim_start();
    if rest.starts_with('"') {
        let e = rest[1..].find('"').unwrap();
        &rest[1..1 + e]
    } else {
        let e = rest.find(|c| c == ',' || c == '}' || c == ']').unwrap();
        &rest[..e]
    }
}

fn nums(json: &str, key: &str) -> Vec<f32> {
    let k = format!("\"{key}\": [");
    let start = json.find(&k).unwrap() + k.len();
    let e = json[start..].find(']').unwrap();
    json[start..start + e].split(',').filter_map(|s| s.trim().parse().ok()).collect()
}

fn main() {
    let js = fs::read_to_string("/tmp/qk_ref.json").unwrap();
    // crude split into the two objects
    let q4 = &js[js.find("\"q4k\":").unwrap()..js.find("\"q6k\":").unwrap()];
    let q6 = &js[js.find("\"q6k\":").unwrap()..];

    let mut out = vec![0f32; 256];
    let b4 = hex_to_bytes(field(q4, "first_block_hex"));
    dequant_q4k(&b4, &mut out);
    let ref4 = nums(q4, "deq_first32");
    let d4 = (0..32).map(|i| (out[i] - ref4[i]).abs()).fold(0f32, f32::max);
    println!("Q4_K max abs diff vs gguf (first 32): {d4:.2e}  {}", if d4 < 1e-5 { "PASS" } else { "FAIL" });
    println!("  ours[:4]  {:?}", &out[..4].iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
    println!("  gguf[:4]  {:?}", &ref4[..4].iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());

    let b6 = hex_to_bytes(field(q6, "first_block_hex"));
    dequant_q6k(&b6, &mut out);
    let ref6 = nums(q6, "deq_first32");
    let d6 = (0..32).map(|i| (out[i] - ref6[i]).abs()).fold(0f32, f32::max);
    println!("Q6_K max abs diff vs gguf (first 32): {d6:.2e}  {}", if d6 < 1e-5 { "PASS" } else { "FAIL" });
    println!("  ours[:4]  {:?}", &out[..4].iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
    println!("  gguf[:4]  {:?}", &ref6[..4].iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
}
