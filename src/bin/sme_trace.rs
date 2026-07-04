//! Instruction-level trace of one real SME matmul: C(16x16) = A(16x2) * B(2x16).
//!
//! Runs the actual instructions on the SME unit and dumps the hardware ZA tile
//! state after each fmopa, so you can watch the outer products accumulate.
//!
//! Inputs chosen so every intermediate is recognizable by eye:
//!   A column 0 = [0,1,2,...,15]   B row 0 = [1,1,1,...,1]
//!   A column 1 = [2,2,2,..., 2]   B row 1 = [0,1,2,...,15]
//! => after fmopa #1: C[i][j] = i          (row gradient)
//! => after fmopa #2: C[i][j] = i + 2*j    (row + column gradient)

use std::arch::asm;

/// One 16x16x2 matmul, dumping ZA after each of the two fmopa instructions.
/// a: [k=2][16] column-major-per-k, b: [k=2][16].
unsafe fn traced_matmul(a: *const f32, b: *const f32, dump1: *mut f32, dump2: *mut f32) {
    unsafe {
        asm!(
            ".arch armv9.2-a+sme2",
            "smstart",                                //  enter streaming mode, ZA live
            "ptrue p0.s",                             //  predicate: all 16 lanes on
            "zero {{za}}",                            //  clear accumulator tile
            // ---- k = 0 ----
            "ld1w {{z0.s}}, p0/z, [{a}]",             //  z0 <- A column 0 (16 floats)
            "ld1w {{z1.s}}, p0/z, [{b}]",             //  z1 <- B row 0    (16 floats)
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",    //  za0 += z0 (x) z1  [256 FMAs]
            // dump ZA tile -> dump1 (16 slices of 16 floats)
            "mov w12, #0",
            "mov {t}, {d1}",
            "2:",
            "mov z4.s, p0/m, za0h.s[w12, 0]",         //  extract horizontal slice w12
            "st1w {{z4.s}}, p0, [{t}]",
            "add {t}, {t}, #64",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 2b",
            // ---- k = 1 ----
            "ld1w {{z0.s}}, p0/z, [{a}, #1, mul vl]", //  z0 <- A column 1
            "ld1w {{z1.s}}, p0/z, [{b}, #1, mul vl]", //  z1 <- B row 1
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",    //  za0 += z0 (x) z1
            // dump ZA tile -> dump2
            "mov w12, #0",
            "mov {t}, {d2}",
            "3:",
            "mov z4.s, p0/m, za0h.s[w12, 0]",
            "st1w {{z4.s}}, p0, [{t}]",
            "add {t}, {t}, #64",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 3b",
            "smstop",                                 //  leave streaming mode
            a = in(reg) a,
            b = in(reg) b,
            d1 = in(reg) dump1,
            d2 = in(reg) dump2,
            t = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("x12") _,
            options(nostack),
        );
    }
}

fn print_mat(title: &str, m: &[f32], rows: usize, cols: usize) {
    println!("{title}");
    for i in 0..rows {
        let row: Vec<String> = (0..cols).map(|j| format!("{:>3.0}", m[i * cols + j])).collect();
        println!("  {}", row.join(" "));
    }
    println!();
}

fn main() {
    // packed A: [k][16 rows], packed B: [k][16 cols]
    let mut a = [0.0f32; 32];
    let mut b = [0.0f32; 32];
    for i in 0..16 {
        a[i] = i as f32; //      A column 0
        a[16 + i] = 2.0; //      A column 1
        b[i] = 1.0; //           B row 0
        b[16 + i] = i as f32; // B row 1
    }
    let mut d1 = [0.0f32; 256];
    let mut d2 = [0.0f32; 256];
    unsafe { traced_matmul(a.as_ptr(), b.as_ptr(), d1.as_mut_ptr(), d2.as_mut_ptr()) };

    print_mat("A column 0 (z0 after 1st ld1w):", &a[..16], 1, 16);
    print_mat("B row 0    (z1 after 1st ld1w):", &b[..16], 1, 16);
    print_mat("ZA tile after fmopa #1  (= col0 x row0, C[i][j] = i):", &d1, 16, 16);
    print_mat("A column 1 (z0 after 2nd ld1w):", &a[16..], 1, 16);
    print_mat("B row 1    (z1 after 2nd ld1w):", &b[16..], 1, 16);
    print_mat("ZA tile after fmopa #2  (accumulated, C[i][j] = i + 2j):", &d2, 16, 16);

    // verify against scalar reference
    let mut ok = true;
    for i in 0..16 {
        for j in 0..16 {
            let expect = i as f32 + 2.0 * j as f32;
            if (d2[i * 16 + j] - expect).abs() > 1e-6 {
                ok = false;
            }
        }
    }
    println!("verified against scalar reference: {}", if ok { "EXACT" } else { "MISMATCH" });
}
