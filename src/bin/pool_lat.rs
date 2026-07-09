//! Measure the engine's fork-join pool dispatch cost (condvar broadcast + join)
//! with a no-op task — replicates engine.rs pool_run exactly.
//! Decode does ~8 par_rows/layer x 24 layers = ~192 dispatches/token, so at X us
//! per dispatch the fixed overhead is 192*X us/token.
use std::sync::{Condvar, Mutex};
use std::time::Instant;

struct PoolState { seq: u64, task: [usize; 2], remaining: usize }
struct PoolShared { m: Mutex<PoolState>, cv_go: Condvar, cv_done: Condvar }

fn main() {
    let nt: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let shared: &'static PoolShared = Box::leak(Box::new(PoolShared {
        m: Mutex::new(PoolState { seq: 0, task: [0, 0], remaining: 0 }),
        cv_go: Condvar::new(),
        cv_done: Condvar::new(),
    }));
    for _wid in 0..nt {
        std::thread::spawn(move || {
            unsafe extern "C" { fn pthread_set_qos_class_self_np(q: u32, p: i32) -> i32; }
            unsafe { pthread_set_qos_class_self_np(0x21, 0); }
            let mut seen = 0u64;
            loop {
                let task;
                {
                    let mut st = shared.m.lock().unwrap();
                    while st.seq == seen { st = shared.cv_go.wait(st).unwrap(); }
                    seen = st.seq;
                    task = st.task;
                }
                let f: &(dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(task) };
                f(0);
                let mut st = shared.m.lock().unwrap();
                st.remaining -= 1;
                if st.remaining == 0 { shared.cv_done.notify_one(); }
            }
        });
    }
    let run = |f: &(dyn Fn(usize) + Sync)| {
        let task: [usize; 2] = unsafe { std::mem::transmute(f) };
        let mut st = shared.m.lock().unwrap();
        st.task = task;
        st.remaining = nt;
        st.seq += 1;
        shared.cv_go.notify_all();
        while st.remaining > 0 { st = shared.cv_done.wait(st).unwrap(); }
    };
    let nop = |_t: usize| {};
    // warmup
    for _ in 0..100 { run(&nop); }
    let iters = 2000;
    let t0 = Instant::now();
    for _ in 0..iters { run(&nop); }
    let us = t0.elapsed().as_micros() as f64 / iters as f64;
    println!("pool dispatch (nt={nt}): {us:.1} us/dispatch -> {:.1} ms/token at 192 dispatches", us * 192.0 / 1000.0);
}
