// mmap_bw.m — GPU streaming-read bandwidth from a file-backed mmap wrapped with
// newBufferWithBytesNoCopy, vs a regular device-allocated buffer. If mmap reads are
// slow, the engine must copy weights into a device buffer at load (llama.cpp-style
// residency) instead of zero-copy.
// Build: clang -O2 -fobjc-arc mmap_bw.m -framework Metal -framework Foundation -o mmap_bw
// Run: ./mmap_bw <file> [GiB]
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

int main(int argc, char **argv) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    double gib = argc > 2 ? atof(argv[2]) : 2.0;
    size_t N = ((size_t)(gib * (1ull << 30))) & ~16383ull;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void ksum(device const float4* s [[buffer(0)]], device float* o [[buffer(1)]],\n"
        "                 constant uint& n4 [[buffer(2)]],\n"
        "                 uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {\n"
        "    float4 acc = 0;\n"
        "    for (uint i = gid; i < n4; i += gsz) acc += s[i];\n"
        "    if ((gid & 1023) == 0) o[gid >> 10] = acc.x + acc.y + acc.z + acc.w;\n"
        "}\n" options:nil error:&err];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ksum"] error:&err];
    id<MTLBuffer> out = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];

    void (^bench)(NSString *, id<MTLBuffer>, size_t) = ^(NSString *label, id<MTLBuffer> buf, size_t len) {
        uint32_t n4 = (uint32_t)(len / 16);
        double best = 0;
        for (int it = 0; it < 6; it++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:buf offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e setBytes:&n4 length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(1 << 20, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            double dt = cb.GPUEndTime - cb.GPUStartTime;
            double g = len / dt / 1e9;
            if (it > 0 && g > best) best = g;
        }
        printf("%-28s %7.1f GB/s\n", label.UTF8String, best);
    };

    // 1. regular shared device buffer
    id<MTLBuffer> reg = [dev newBufferWithLength:N options:MTLResourceStorageModeShared];
    memset(reg.contents, 1, N);
    bench(@"regular shared buffer", reg, N);

    // 2. anonymous mmap wrapped
    void *anon = mmap(NULL, N, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(anon, 1, N);
    id<MTLBuffer> ab = [dev newBufferWithBytesNoCopy:anon length:N options:MTLResourceStorageModeShared deallocator:nil];
    if (ab) bench(@"anon mmap bytesNoCopy", ab, N); else printf("anon mmap wrap FAILED\n");

    // 3. file-backed mmap wrapped (the engine's path)
    if (argc > 1) {
        int fd = open(argv[1], O_RDONLY);
        struct stat st; fstat(fd, &st);
        size_t len = ((size_t)st.st_size < N ? (size_t)st.st_size : N) & ~16383ull;
        void *fm = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
        // touch to warm page cache
        volatile char acc = 0;
        for (size_t i = 0; i < len; i += 16384) acc ^= ((char *)fm)[i];
        id<MTLBuffer> fb = [dev newBufferWithBytesNoCopy:fm length:len options:MTLResourceStorageModeShared deallocator:nil];
        if (fb) bench(@"file mmap bytesNoCopy (warm)", fb, len); else printf("file mmap wrap FAILED\n");
    }
    return 0;
} }
