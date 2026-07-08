// sync_lat2.m — G2 retry: persistent-kernel mailbox with RMW atomics (coherency probe).
// Failure of v1: GPU atomic_load spin never saw CPU stores (or vice versa) — Metal only
// promises shared-mode visibility at command-buffer boundaries; plain loads can hit a
// stale GPU cache line forever. Hypothesis: RMW atomics (fetch_add/exchange) are
// performed at a coherent point (LLC) and DO observe CPU writes mid-kernel.
// This version: GPU spins with atomic_fetch_add(&m[0],0) (RMW read), responds with
// atomic_exchange; adds GPU heartbeat + bounded spins so nothing can hang.
// Build: clang -O2 -fobjc-arc sync_lat2.m -framework Metal -framework Foundation -o sync_lat2
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdatomic.h>
#include <mach/mach_time.h>

static double now_us(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1000.0;
}
static int cmpd(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    printf("device: %s\n", dev.name.UTF8String);

    // mailbox: [0] req  [1] resp  [2] iters  [3] heartbeat  [4] bailed  [5] spins-hi
    NSString *src = @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void kmailbox(volatile device atomic_uint* m [[buffer(0)]],\n"
        "                     uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid != 0) return;\n"
        "    uint iters = atomic_fetch_add_explicit(&m[2], 0u, memory_order_relaxed);\n"
        "    uint last = 0;\n"
        "    for (uint i = 0; i < iters; i++) {\n"
        "        uint req; ulong spins = 0;\n"
        "        do {\n"
        "            req = atomic_fetch_add_explicit(&m[0], 0u, memory_order_relaxed);  // RMW read\n"
        "            if ((++spins & 0xFFFF) == 0) atomic_fetch_add_explicit(&m[3], 1u, memory_order_relaxed); // heartbeat\n"
        "            if (spins > 400000000UL) { atomic_store_explicit(&m[4], 1u, memory_order_relaxed); return; } // bail\n"
        "        } while (req == last);\n"
        "        last = req;\n"
        "        atomic_exchange_explicit(&m[1], req, memory_order_relaxed);            // RMW write\n"
        "    }\n"
        "}\n";
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { printf("compile error: %s\n", err.description.UTF8String); return 1; }
    id<MTLComputePipelineState> pmb = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kmailbox"] error:&err];
    if (!pmb) { printf("pso error: %s\n", err.description.UTF8String); return 1; }

    id<MTLBuffer> mbox = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    volatile _Atomic uint32_t *mb = (volatile _Atomic uint32_t *)mbox.contents;
    const uint32_t ITERS = 20000;
    for (int i = 0; i < 8; i++) atomic_store(&mb[i], 0);
    atomic_store(&mb[2], ITERS);

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pmb];
    [e setBuffer:mbox offset:0 atIndex:0];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [e endEncoding];
    [cb commit];
    usleep(5000);
    printf("heartbeat after 5ms: %u (kernel alive: %s)\n", atomic_load(&mb[3]), atomic_load(&mb[3]) ? "YES" : "no");

    static double lat[20000];
    uint32_t done = 0;
    for (uint32_t i = 1; i <= ITERS; i++) {
        double t0 = now_us();
        atomic_store_explicit(&mb[0], i, memory_order_release);
        // CPU spin with 200ms bail-out
        while (atomic_load_explicit(&mb[1], memory_order_acquire) != i) {
            if (now_us() - t0 > 200000.0) goto out;
        }
        lat[i-1] = now_us() - t0;
        done = i;
    }
out:
    if (done < ITERS) {
        printf("STALLED at handshake %u/%u  heartbeat=%u bailed=%u resp=%u\n",
               done + 1, ITERS, atomic_load(&mb[3]), atomic_load(&mb[4]), atomic_load(&mb[1]));
        atomic_store(&mb[0], 0xFFFFFFFF);  // poke so GPU may progress; then bounded wait
    }
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError)
        printf("kernel ERROR: %s\n", cb.error.description.UTF8String);
    if (done > 100) {
        qsort(lat, done, sizeof(double), cmpd);
        printf("G2-RMW mailbox round-trip: median %6.2f us   p10 %6.2f   p90 %6.2f   p99 %6.2f  (n=%u)\n",
               lat[done/2], lat[done/10], lat[done*9/10], lat[done*99/100], done);
        printf("budget: 96 syncs/token x median = %5.3f ms/token\n", 96.0 * lat[done/2] / 1000.0);
    }
    return 0;
} }
