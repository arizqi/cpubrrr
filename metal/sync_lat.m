// sync_lat.m — G1+G2: CPU<->GPU sync latency on M4 Max (GPU phase).
// G1: classic dispatch round-trip (commit + waitUntilCompleted, 1-thread kernel)
// G2: persistent-kernel mailbox — GPU kernel spins on shared-memory word; CPU writes a
//     request seq, GPU echoes it; N handshakes inside ONE kernel launch.
// Hybrid single-stream decode needs ~48-96 syncs/token; budget <20us each.
// Build: clang -O2 -fobjc-arc sync_lat.m -framework Metal -framework Foundation -o sync_lat
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

    NSString *src = @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void knop(device uint* x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid == 0) x[0] += 1;\n"
        "}\n"
        "kernel void kmailbox(volatile device atomic_uint* m [[buffer(0)]],\n"
        "                     uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid != 0) return;\n"
        "    uint iters = atomic_load_explicit(&m[2], memory_order_relaxed);\n"
        "    uint last = 0;\n"
        "    for (uint i = 0; i < iters; i++) {\n"
        "        uint req;\n"
        "        do { req = atomic_load_explicit(&m[0], memory_order_relaxed); } while (req == last);\n"
        "        last = req;\n"
        "        atomic_store_explicit(&m[1], req, memory_order_relaxed);\n"
        "    }\n"
        "}\n";
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { printf("compile error: %s\n", err.description.UTF8String); return 1; }
    id<MTLComputePipelineState> nop = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"knop"] error:&err];
    id<MTLComputePipelineState> pmb = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kmailbox"] error:&err];
    id<MTLBuffer> tiny = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];

    // ---------- G1 ----------
    double t[400];
    for (int i = 0; i < 200; i++) {
        double t0 = now_us();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:nop];
        [e setBuffer:tiny offset:0 atIndex:0];
        [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        t[i] = now_us() - t0;
    }
    qsort(t, 200, sizeof(double), cmpd);
    printf("G1 dispatch round-trip: median %6.1f us   p10 %6.1f   p90 %6.1f\n", t[100], t[20], t[180]);

    // ---------- G2 ----------
    id<MTLBuffer> mbox = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    volatile _Atomic uint32_t *mb = (volatile _Atomic uint32_t *)mbox.contents;
    const uint32_t ITERS = 20000;
    atomic_store(&mb[0], 0); atomic_store(&mb[1], 0); atomic_store(&mb[2], ITERS);

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pmb];
    [e setBuffer:mbox offset:0 atIndex:0];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [e endEncoding];
    [cb commit];               // kernel now resident + spinning
    usleep(3000);

    static double lat[20000];
    for (uint32_t i = 1; i <= ITERS; i++) {
        double t0 = now_us();
        atomic_store_explicit(&mb[0], i, memory_order_relaxed);       // request
        while (atomic_load_explicit(&mb[1], memory_order_relaxed) != i) {} // echo
        lat[i-1] = now_us() - t0;
    }
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError)
        printf("G2 kernel ERROR: %s\n", cb.error.description.UTF8String);
    qsort(lat, ITERS, sizeof(double), cmpd);
    printf("G2 mailbox round-trip: median %6.2f us   p10 %6.2f   p90 %6.2f   p99 %6.2f  (n=%u)\n",
           lat[ITERS/2], lat[ITERS/10], lat[ITERS*9/10], lat[ITERS*99/100], ITERS);
    printf("budget: 96 syncs/token x median = %5.3f ms/token\n", 96.0 * lat[ITERS/2] / 1000.0);
    return 0;
} }
