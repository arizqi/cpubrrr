// sync_lat3.m — G2b: MTLSharedEvent ping-pong (the coherent path).
// v1/v2 finding: NO mid-kernel CPU<->GPU visibility on AGX (writes flush at command
// boundaries only) — persistent-kernel mailboxes are impossible on Metal.
// Plan B: pre-encode K (wait -> tiny kernel -> signal) triples in ONE command buffer.
//   GPU: waitForEvent(ev, 2k+1); kernel writes result; signalEvent(ev, 2k+2)
//   CPU: write input; ev.signaledValue = 2k+1; spin on ev.signaledValue >= 2k+2; read.
// Metal guarantees writes-before-signal are visible to the waiter. Question = latency.
// Also verifies DATA visibility: kernel adds 1000+k to a shared counter, CPU checks.
// Build: clang -O2 -fobjc-arc sync_lat3.m -framework Metal -framework Foundation -o sync_lat3
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
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
        "kernel void kstep(device uint* io [[buffer(0)]], constant uint& k [[buffer(1)]],\n"
        "                  uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid == 0) io[k] = io[1024 + k] + 1000u + k;  // read CPU input, write result\n"
        "}\n";
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { printf("compile error: %s\n", err.description.UTF8String); return 1; }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kstep"] error:&err];

    id<MTLBuffer> buf = [dev newBufferWithLength:1 << 16 options:MTLResourceStorageModeShared];
    uint32_t *bp = (uint32_t *)buf.contents;
    id<MTLSharedEvent> ev = [dev newSharedEvent];

    const int STEPS = 48;       // one "token" worth of layer syncs
    const int TOKENS = 200;     // repeat
    static double lat[48 * 200];
    int n = 0, dataErr = 0;
    uint64_t base = 0;

    for (int t = 0; t < TOKENS; t++) {
        // encode one command buffer with 48 wait->kernel->signal triples
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (uint32_t k = 0; k < STEPS; k++) {
            [cb encodeWaitForEvent:ev value:base + 2*k + 1];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:buf offset:0 atIndex:0];
            [e setBytes:&k length:4 atIndex:1];
            [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [e endEncoding];
            [cb encodeSignalEvent:ev value:base + 2*k + 2];
        }
        [cb commit];

        for (uint32_t k = 0; k < STEPS; k++) {
            double t0 = now_us();
            bp[1024 + k] = 7000 + k;                    // CPU "hidden state" input
            ev.signaledValue = base + 2*k + 1;          // release GPU
            while (ev.signaledValue < base + 2*k + 2) {} // spin for GPU done
            lat[n++] = now_us() - t0;
            if (bp[k] != 7000 + k + 1000 + k) dataErr++; // verify data visibility
        }
        [cb waitUntilCompleted];
        base += 2 * STEPS;
    }

    qsort(lat, n, sizeof(double), cmpd);
    printf("G2b shared-event round-trip (incl tiny kernel): median %6.2f us  p10 %6.2f  p90 %6.2f  p99 %6.2f  (n=%d)\n",
           lat[n/2], lat[n/10], lat[n*9/10], lat[n*99/100], n);
    printf("data visibility errors: %d/%d %s\n", dataErr, n, dataErr ? "!!! BROKEN" : "(all correct)");
    printf("budget: 48 syncs/token x median = %5.3f ms/token; 96x = %5.3f ms/token\n",
           48.0 * lat[n/2] / 1000.0, 96.0 * lat[n/2] / 1000.0);
    return 0;
} }
