// disp_cost.m — how much does a chain of N serial dispatches cost on AGX?
// Emulates the engine's per-token command buffer: N dependent dispatches on the same
// buffer (hazard-tracked, serial encoder). If N=580 empty dispatches ~= 15ms GPU time,
// dispatch serialization — not kernel math — is the engine's real bottleneck.
// Build: clang -O2 -fobjc-arc disp_cost.m -framework Metal -framework Foundation -o disp_cost
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void knop(device float* x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    x[gid] += 1.0f;\n"
        "}\n" options:nil error:&err];
    id<MTLComputePipelineState> nop = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"knop"] error:&err];
    id<MTLBuffer> buf = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];

    int chains[] = {1, 16, 128, 580};
    int sizes[] = {32, 2048, 65536};   // threads per dispatch
    for (int si = 0; si < 3; si++) {
        for (int ci = 0; ci < 4; ci++) {
            int N = chains[ci], T = sizes[si];
            // warmup + measure
            double best = 1e9;
            for (int it = 0; it < 5; it++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                for (int i = 0; i < N; i++) {
                    [e setComputePipelineState:nop];
                    [e setBuffer:buf offset:0 atIndex:0];
                    [e dispatchThreads:MTLSizeMake(T,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                }
                [e endEncoding];
                [cb commit]; [cb waitUntilCompleted];
                double g = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                if (it > 0 && g < best) best = g;
            }
            printf("threads %6d  chain %4d: gpu %7.3f ms  (%6.2f us/dispatch)\n",
                   T, N, best, best * 1000.0 / N);
        }
    }
    return 0;
} }
