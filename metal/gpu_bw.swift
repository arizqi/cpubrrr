// gpu_bw.swift — measure achievable M4 Max GPU memory bandwidth (E1, GPU phase).
// Two probes:
//   1. blit copy   (read N + write N  -> 2N bytes moved)  — near-peak DMA path
//   2. compute sum (read N, write ~0)                     — pure streaming read, like decode
//   3. compute copy(read N + write N)                     — shader-path copy
// Build: swiftc -O gpu_bw.swift -o gpu_bw   Run: ./gpu_bw [GiB]
import Metal
import Foundation

let gib = CommandLine.arguments.count > 1 ? Double(CommandLine.arguments[1]) ?? 2.0 : 2.0
let N = Int(gib * 1024 * 1024 * 1024) & ~4095   // bytes, page aligned
let iters = 8

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
print("device: \(dev.name)  buffer: \(String(format: "%.1f", Double(N)/1e9)) GB  iters: \(iters)")

guard let src = dev.makeBuffer(length: N, options: .storageModeShared),
      let dst = dev.makeBuffer(length: N, options: .storageModeShared),
      let out = dev.makeBuffer(length: 1 << 20, options: .storageModeShared) else { fatalError("alloc") }
// touch pages so first-use faults don't pollute timing
memset(src.contents(), 1, N); memset(dst.contents(), 2, N)

let msl = """
#include <metal_stdlib>
using namespace metal;
kernel void ksum(device const float4* s [[buffer(0)]],
                 device float* o [[buffer(1)]],
                 constant uint& n4 [[buffer(2)]],
                 uint gid [[thread_position_in_grid]],
                 uint gsz [[threads_per_grid]]) {
    float4 acc = 0;
    for (uint i = gid; i < n4; i += gsz) acc += s[i];
    if ((gid & 1023) == 0) o[gid >> 10] = acc.x + acc.y + acc.z + acc.w;
}
kernel void kcopy(device const float4* s [[buffer(0)]],
                  device float4* d [[buffer(1)]],
                  constant uint& n4 [[buffer(2)]],
                  uint gid [[thread_position_in_grid]],
                  uint gsz [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += gsz) d[i] = s[i];
}
"""
let lib = try dev.makeLibrary(source: msl, options: nil)
let q = dev.makeCommandQueue()!

func timeGPU(_ label: String, bytesPerIter: Double, _ encode: (MTLCommandBuffer) -> Void) {
    // one warmup
    var best = 0.0
    for it in 0...iters {
        let cb = q.makeCommandBuffer()!
        encode(cb)
        let t0 = DispatchTime.now()
        cb.commit(); cb.waitUntilCompleted()
        let dt = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        let gbps = bytesPerIter / dt / 1e9
        if it > 0 { best = max(best, gbps) }
    }
    print(String(format: "%-14s best %7.1f GB/s", (label as NSString).utf8String!, best))
}

// 1. blit copy: 2N bytes
timeGPU("blit copy", bytesPerIter: 2.0 * Double(N)) { cb in
    let b = cb.makeBlitCommandEncoder()!
    b.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: N)
    b.endEncoding()
}

// 2/3. compute kernels
var n4 = UInt32(N / 16)
let grid = 1 << 20   // 1M threads, each strides
for (name, fn, bytes) in [("compute read", "ksum", Double(N)),
                          ("compute copy", "kcopy", 2.0 * Double(N))] {
    let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: fn)!)
    timeGPU(name, bytesPerIter: bytes) { cb in
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso)
        e.setBuffer(src, offset: 0, index: 0)
        e.setBuffer(fn == "ksum" ? out : dst, offset: 0, index: 1)
        e.setBytes(&n4, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: grid, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        e.endEncoding()
    }
}
