// sync_lat.swift — G1+G2: CPU<->GPU synchronization latency on M4 Max (GPU phase).
// G1: classic dispatch round-trip (commandBuffer commit + waitUntilCompleted, empty kernel)
// G2: persistent-kernel mailbox — GPU kernel spins on a shared-memory atomic; CPU writes
//     a request seq, GPU echoes it back; N handshakes inside ONE kernel launch.
// The hybrid single-stream engine needs ~48-96 syncs/token; budget <20us each.
// Build: swiftc -O sync_lat.swift -o sync_lat   Run: ./sync_lat
import Metal
import Foundation

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal") }
let q = dev.makeCommandQueue()!
print("device: \(dev.name)")

// ---------- G1: classic dispatch round-trip ----------
let empty = """
#include <metal_stdlib>
using namespace metal;
kernel void knop(device uint* x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0) x[0] += 1;
}
"""
let lib1 = try dev.makeLibrary(source: empty, options: nil)
let nop = try dev.makeComputePipelineState(function: lib1.makeFunction(name: "knop")!)
let tiny = dev.makeBuffer(length: 64, options: .storageModeShared)!

var times: [Double] = []
for _ in 0..<200 {
    let t0 = DispatchTime.now().uptimeNanoseconds
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(nop)
    e.setBuffer(tiny, offset: 0, index: 0)
    e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    e.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000.0)
}
times.sort()
print(String(format: "G1 dispatch round-trip: median %6.1f us   p10 %6.1f   p90 %6.1f",
             times[100], times[20], times[180]))

// ---------- G2: persistent-kernel mailbox ----------
// mailbox layout (uint32 slots, storageModeShared):
//   [0] = request seq (CPU writes)   [1] = response seq (GPU writes)   [2] = iters
let mbox = dev.makeBuffer(length: 4096, options: .storageModeShared)!
let mb = mbox.contents().bindMemory(to: UInt32.self, capacity: 1024)

let persistent = """
#include <metal_stdlib>
using namespace metal;
kernel void kmailbox(volatile device atomic_uint* m [[buffer(0)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    uint iters = atomic_load_explicit(&m[2], memory_order_relaxed);
    uint last = 0;
    for (uint i = 0; i < iters; i++) {
        uint req;
        do { req = atomic_load_explicit(&m[0], memory_order_relaxed); } while (req == last);
        last = req;
        atomic_store_explicit(&m[1], req, memory_order_relaxed);
    }
}
"""
let lib2 = try dev.makeLibrary(source: persistent, options: nil)
let pmb = try dev.makeComputePipelineState(function: lib2.makeFunction(name: "kmailbox")!)

let ITERS: UInt32 = 20000
mb[0] = 0; mb[1] = 0; mb[2] = ITERS
let cb = q.makeCommandBuffer()!
let e = cb.makeComputeCommandEncoder()!
e.setComputePipelineState(pmb)
e.setBuffer(mbox, offset: 0, index: 0)
e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                  threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
e.endEncoding()
cb.commit()   // kernel now spinning on the mailbox

usleep(2000)  // let it arrive
var lat: [Double] = []
let volPtr = UnsafeMutablePointer<UInt32>(mutating: mb)
for i in 1...ITERS {
    let t0 = DispatchTime.now().uptimeNanoseconds
    // publish request
    OSAtomicCompareAndSwap32(Int32(bitPattern: volPtr[0]), Int32(bitPattern: i), UnsafeMutablePointer<Int32>(OpaquePointer(volPtr)))
    // spin for echo
    while volPtr[1] != i { /* spin */ }
    lat.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000.0)
}
cb.waitUntilCompleted()
if cb.status == .error { print("G2 kernel ERROR: \(String(describing: cb.error))") }
lat.sort()
let n = lat.count
print(String(format: "G2 mailbox round-trip: median %6.2f us   p10 %6.2f   p90 %6.2f   p99 %6.2f  (n=%d)",
             lat[n/2], lat[n/10], lat[n*9/10], lat[n*99/100], n))
print(String(format: "budget check: 96 syncs/token x median = %5.2f ms/token", 96.0 * lat[n/2] / 1000.0))
