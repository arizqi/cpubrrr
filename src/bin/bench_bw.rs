//! Streaming-bandwidth ceiling probe, shaped like the engine's real access pattern.
//! bench_moe answered "how fast can N threads stream int4 expert weights in a tight
//! loop"; this answers the question the engine actually cares about: what does the
//! machine sustain when 12 persistent workers each walk their own sequential run,
//! at the working-set size and thread count decode really uses, WITH a barrier
//! between passes (so we can price the barrier/pipeline-drain cost directly).
//! Usage: bench_bw [total_MB] [threads]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

fn main() {
    let mb: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let nt: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    let bytes = mb * 1_000_000;
    println!("== streaming ceiling: {mb} MB working set, {nt} threads ==");
    let buf: &'static [u8] = Box::leak(vec![1u8; bytes].into_boxed_slice());

    struct S { bcount: AtomicUsize, bsense: AtomicBool, go: AtomicUsize, acc: [AtomicUsize; 32] }
    const Z: AtomicUsize = AtomicUsize::new(0);
    let sh: &'static S = Box::leak(Box::new(S {
        bcount: AtomicUsize::new(0), bsense: AtomicBool::new(false), go: AtomicUsize::new(0), acc: [Z; 32],
    }));

    // `stages` passes per iteration, each separated by a barrier: mimics the engine's
    // 336-barriers-per-token structure so the drain cost shows up in the number.
    for stages in [1usize, 14, 336] {
        for w in 0..nt {
            let _ = w;
        }
        let iters = 6usize;
        let handles: Vec<_> = (0..nt).map(|w| {
            std::thread::spawn(move || {
                unsafe extern "C" { fn pthread_set_qos_class_self_np(q: u32, p: i32) -> i32; }
                unsafe { pthread_set_qos_class_self_np(0x21, 0); }
                let mut ls = false;
                let mut seen = 0usize;
                loop {
                    while sh.go.load(Ordering::Acquire) == seen { std::hint::spin_loop(); }
                    seen = sh.go.load(Ordering::Acquire);
                    if seen == usize::MAX { return; }
                    let mut sum = 0usize;
                    let per = bytes / stages;
                    for st in 0..stages {
                        // each worker walks its own contiguous run of this stage's slice
                        let base = st * per;
                        let chunk = per / nt;
                        let a = base + w * chunk;
                        let b = a + chunk;
                        let mut i = a;
                        while i + 64 <= b {
                            sum += buf[i] as usize + buf[i + 32] as usize;
                            i += 64;
                        }
                        // barrier
                        ls = !ls;
                        if sh.bcount.fetch_add(1, Ordering::AcqRel) == nt - 1 {
                            sh.bcount.store(0, Ordering::Relaxed);
                            sh.bsense.store(ls, Ordering::Release);
                        } else {
                            let mut n = 0u32;
                            while sh.bsense.load(Ordering::Acquire) != ls {
                                n += 1;
                                if n & 1023 != 0 { std::hint::spin_loop(); } else { std::thread::yield_now(); }
                            }
                        }
                    }
                    sh.acc[w].store(sum, Ordering::Relaxed);
                    if w == 0 { sh.acc[31].fetch_add(1, Ordering::Release); }
                }
            })
        }).collect();

        let mut best = f64::MAX;
        for it in 0..iters {
            let done0 = sh.acc[31].load(Ordering::Acquire);
            let t0 = Instant::now();
            sh.go.fetch_add(1, Ordering::Release);
            while sh.acc[31].load(Ordering::Acquire) == done0 { std::hint::spin_loop(); }
            let dt = t0.elapsed().as_secs_f64();
            if it > 0 { best = best.min(dt); }
        }
        sh.go.store(usize::MAX, Ordering::Release);
        for h in handles { let _ = h.join(); }
        println!("  stages={stages:<4} (barriers/pass) {:6.1} GB/s   {:.2} ms/pass",
                 bytes as f64 / best / 1e9, best * 1e3);
        // reset for next config
        sh.go.store(0, Ordering::Release);
        sh.acc[31].store(0, Ordering::Release);
        sh.bcount.store(0, Ordering::Release);
        sh.bsense.store(false, Ordering::Release);
    }
}
