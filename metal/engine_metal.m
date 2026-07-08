// engine_metal.m — cpubrrr full-GPU decode engine for qwen3moe (Q4_K/Q6_K), v1.
//
// Findings that shaped this design (see docs/RESEARCH_LOG_V2.md, GPU phase):
//   - No mid-kernel CPU<->GPU coherency on AGX -> persistent mailbox impossible.
//   - MTLSharedEvent sync = 63us -> per-layer CPU/GPU hybrid refuted.
//   - Therefore: the WHOLE forward pass lives on the GPU; ONE command buffer per
//     token (~580 dispatches, hazard-tracked); a single ~100us round-trip per token.
//   - Weights: mmap'd Ollama blob wrapped zero-copy (newBufferWithBytesNoCopy),
//     kernels index with byte offsets; same int-accumulation Q4_K/Q6_K math that
//     beat llama.cpp on CPU, ported to MSL.
// Targets: >102 tok/s (Ollama Metal, log-verified) toward the 487 GB/s ceiling ~155.
//
// Build: clang -O2 -fobjc-arc engine_metal.m -framework Metal -framework Foundation -o engine_metal
// Run:   ./engine_metal <data-dir> <blob-path> "<prompt>" [ngen]
//        ./engine_metal <data-dir> <blob-path> --verify
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <mach/mach_time.h>

// ---------------- config ----------------
typedef struct { int d, nh, nkv, hd, nl, ne, topk, ff, nvocab; float rope_base, rms_eps; } Cfg;
static Cfg C;
#define MAXSEQ 4096

static double now_s(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

// ---------------- MSL ----------------
static const char *MSL =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"typedef struct { ulong woff; uint rows, cols, bpr, nsb, xoff, soff, moff, ooff, add, pos, aux; } KP;\n"
"\n"
"static inline float half_at(device const uchar* p) { return float(*(device const half*)p); }\n"
"static inline void scale_min_k4(uint j, device const uchar* s, thread uint* sc, thread uint* mn) {\n"
"    if (j < 4) { *sc = s[j] & 63u; *mn = s[j+4] & 63u; }\n"
"    else { *sc = (s[j+4] & 0xFu) | ((s[j-4] >> 6) << 4); *mn = (s[j+4] >> 4) | ((s[j] >> 6) << 4); }\n"
"}\n"
"\n"
"// per-lane partial of q4k int-accum dot for one row; caller simd_sums.\n"
"// vectorized: uchar4/char4 loads, float4 fma (products <= 15*127 -> exact in f32,\n"
"// so this is still the integer-accumulation algorithm, just carried in floats).\n"
"static inline float q4k_lane(device const uchar* blob, ulong woff, uint row, uint bpr, uint nsb,\n"
"                             device const char* xq, device const float* xs, device const int* xsum,\n"
"                             uint xoff, uint soff, uint moff, uint lane) {\n"
"    float acc = 0.0f;\n"
"    uint units = nsb * 8u;\n"
"    for (uint u = lane; u < units; u += 32u) {\n"
"        uint sb = u >> 3, j = u & 7u;\n"
"        device const uchar* b = blob + woff + (ulong)row * bpr + sb * 144u;\n"
"        float d = half_at(b), dmin = half_at(b + 2);\n"
"        uint sc, mn; scale_min_k4(j, b + 4, &sc, &mn);\n"
"        uint c = j >> 1, nib = j & 1u;\n"
"        device const uchar4* q = (device const uchar4*)(b + 16 + c * 32u);\n"
"        device const char4* x = (device const char4*)(xq + xoff + sb * 256u + c * 64u + nib * 32u);\n"
"        float4 a4 = 0.0f;\n"
"        for (uint i = 0; i < 8u; i++) {\n"
"            uchar4 qb = q[i];\n"
"            uchar4 qv = nib ? (qb >> 4) : (qb & (uchar)0xF);\n"
"            a4 = fma(float4(qv), float4(x[i]), a4);\n"
"        }\n"
"        float s0 = a4.x + a4.y + a4.z + a4.w;\n"
"        acc += xs[soff + sb] * (d * float(sc) * s0 - dmin * float(int(mn) * xsum[moff + sb * 8u + j]));\n"
"    }\n"
"    return acc;\n"
"}\n"
"\n"
"// per-lane partial of q6k int-accum dot; caller simd_sums.\n"
"static inline float q6k_lane(device const uchar* blob, ulong woff, uint row, uint bpr, uint nsb,\n"
"                             device const char* xq, device const float* xs,\n"
"                             uint xoff, uint soff, uint lane) {\n"
"    float acc = 0.0f;\n"
"    uint units = nsb * 16u;\n"
"    for (uint u = lane; u < units; u += 32u) {\n"
"        uint sb = u >> 4, g = u & 15u;\n"
"        device const uchar* b = blob + woff + (ulong)row * bpr + sb * 210u;\n"
"        uint h = g >> 3, rem = g & 7u, qg = rem >> 1, is = rem & 1u;\n"
"        device const uchar* ql = b + h * 64u + (qg & 1u) * 32u;\n"
"        device const uchar* qh = b + 128 + h * 32u;\n"
"        int sc = int(*(device const char*)(b + 192 + g));\n"
"        float d = half_at(b + 208);\n"
"        uint lbase = is * 16u;\n"
"        device const char4* x = (device const char4*)(xq + xoff + sb * 256u + h * 128u + qg * 32u + lbase);\n"
"        device const uchar4* ql4 = (device const uchar4*)(ql + lbase);\n"
"        device const uchar4* qh4 = (device const uchar4*)(qh + lbase);\n"
"        float4 a4 = 0.0f;\n"
"        bool hiNib = (qg >= 2);\n"
"        uchar hshift = (uchar)(qg * 2u);\n"
"        for (uint i = 0; i < 4u; i++) {\n"
"            uchar4 lo = hiNib ? (ql4[i] >> 4) : (ql4[i] & (uchar)0xF);\n"
"            uchar4 hi = (qh4[i] >> hshift) & (uchar)3;\n"
"            int4 qv = int4(lo | (hi << 4)) - 32;\n"
"            a4 = fma(float4(qv), float4(x[i]), a4);\n"
"        }\n"
"        float s0 = a4.x + a4.y + a4.z + a4.w;\n"
"        acc += xs[soff + sb] * d * float(sc) * s0;\n"
"    }\n"
"    return acc;\n"
"}\n"
"\n"
"// 4-rows-per-simdgroup q4k matvec: activation loads amortized 4x, 4 independent\n"
"// weight streams per lane for ILP. tg = 32 lanes handles rows [4*tg, 4*tg+4).\n"
"kernel void kmv4x4(device const uchar* blob [[buffer(0)]], device const char* xq [[buffer(1)]],\n"
"                   device const float* xs [[buffer(2)]], device const int* xsum [[buffer(3)]],\n"
"                   device float* y [[buffer(4)]], constant KP& p [[buffer(29)]],\n"
"                   uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint row0 = tg * 4u;\n"
"    float4 acc = 0.0f;\n"
"    uint units = p.nsb * 8u;\n"
"    device const uchar* base = blob + p.woff + (ulong)row0 * p.bpr;\n"
"    for (uint u = lane; u < units; u += 32u) {\n"
"        uint sb = u >> 3, j = u & 7u;\n"
"        uint c = j >> 1, nib = j & 1u;\n"
"        device const char4* x = (device const char4*)(xq + p.xoff + sb * 256u + c * 64u + nib * 32u);\n"
"        float4 xf[8];\n"
"        for (uint i = 0; i < 8u; i++) xf[i] = float4(x[i]);\n"
"        int xsm = xsum[p.moff + sb * 8u + j];\n"
"        float ysb = xs[p.soff + sb];\n"
"        for (uint r = 0; r < 4u; r++) {\n"
"            device const uchar* b = base + (ulong)r * p.bpr + sb * 144u;\n"
"            float d = half_at(b), dmin = half_at(b + 2);\n"
"            uint sc, mn; scale_min_k4(j, b + 4, &sc, &mn);\n"
"            device const uchar4* q = (device const uchar4*)(b + 16 + c * 32u);\n"
"            float4 a4 = 0.0f;\n"
"            for (uint i = 0; i < 8u; i++) {\n"
"                uchar4 qb = q[i];\n"
"                uchar4 qv = nib ? (qb >> 4) : (qb & (uchar)0xF);\n"
"                a4 = fma(float4(qv), xf[i], a4);\n"
"            }\n"
"            float s0 = a4.x + a4.y + a4.z + a4.w;\n"
"            acc[r] += ysb * (d * float(sc) * s0 - dmin * float(int(mn) * xsm));\n"
"        }\n"
"    }\n"
"    for (uint r = 0; r < 4u; r++) {\n"
"        float v = simd_sum(acc[r]);\n"
"        if (lane == 0 && row0 + r < p.rows) { if (p.add) y[p.ooff + row0 + r] += v; else y[p.ooff + row0 + r] = v; }\n"
"    }\n"
"}\n"
"\n"
"kernel void kmv6x4(device const uchar* blob [[buffer(0)]], device const char* xq [[buffer(1)]],\n"
"                   device const float* xs [[buffer(2)]], device float* y [[buffer(4)]],\n"
"                   constant KP& p [[buffer(29)]],\n"
"                   uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint row0 = tg * 4u;\n"
"    float4 acc = 0.0f;\n"
"    uint units = p.nsb * 16u;\n"
"    device const uchar* base = blob + p.woff + (ulong)row0 * p.bpr;\n"
"    for (uint u = lane; u < units; u += 32u) {\n"
"        uint sb = u >> 4, g = u & 15u;\n"
"        uint h = g >> 3, rem = g & 7u, qg = rem >> 1, is = rem & 1u;\n"
"        uint lbase = is * 16u;\n"
"        device const char4* x = (device const char4*)(xq + p.xoff + sb * 256u + h * 128u + qg * 32u + lbase);\n"
"        float4 xf[4];\n"
"        for (uint i = 0; i < 4u; i++) xf[i] = float4(x[i]);\n"
"        float ysb = xs[p.soff + sb];\n"
"        bool hiNib = (qg >= 2);\n"
"        uchar hshift = (uchar)(qg * 2u);\n"
"        for (uint r = 0; r < 4u; r++) {\n"
"            device const uchar* b = base + (ulong)r * p.bpr + sb * 210u;\n"
"            device const uchar4* ql4 = (device const uchar4*)(b + h * 64u + (qg & 1u) * 32u + lbase);\n"
"            device const uchar4* qh4 = (device const uchar4*)(b + 128 + h * 32u + lbase);\n"
"            int sc = int(*(device const char*)(b + 192 + g));\n"
"            float d = half_at(b + 208);\n"
"            float4 a4 = 0.0f;\n"
"            for (uint i = 0; i < 4u; i++) {\n"
"                uchar4 lo = hiNib ? (ql4[i] >> 4) : (ql4[i] & (uchar)0xF);\n"
"                uchar4 hi = (qh4[i] >> hshift) & (uchar)3;\n"
"                int4 qv = int4(lo | (hi << 4)) - 32;\n"
"                a4 = fma(float4(qv), xf[i], a4);\n"
"            }\n"
"            float s0 = a4.x + a4.y + a4.z + a4.w;\n"
"            acc[r] += ysb * d * float(sc) * s0;\n"
"        }\n"
"    }\n"
"    for (uint r = 0; r < 4u; r++) {\n"
"        float v = simd_sum(acc[r]);\n"
"        if (lane == 0 && row0 + r < p.rows) { if (p.add) y[p.ooff + row0 + r] += v; else y[p.ooff + row0 + r] = v; }\n"
"    }\n"
"}\n"
"\n"
"kernel void kembed(device const uchar* blob [[buffer(0)]], device float* x [[buffer(1)]],\n"
"                   device const uint* toks [[buffer(2)]],\n"
"                   constant KP& p [[buffer(29)]], uint gid [[thread_position_in_grid]]) {\n"
"    // p.woff = token_embd offset, p.cols = d; one thread per element.\n"
"    // p.add==0: token id = p.aux (prefill). p.add==1: GPU-chained decode -> read\n"
"    // the PREVIOUS argmax from toks[p.pos-1]; no CPU round-trip on the critical path.\n"
"    if (gid >= p.cols) return;\n"
"    uint tok = p.add ? toks[p.pos - 1u] : p.aux;\n"
"    uint sb = gid >> 8, y = gid & 255u;\n"
"    device const uchar* b = blob + p.woff + (ulong)tok * p.bpr + sb * 144u;\n"
"    float d = half_at(b), dmin = half_at(b + 2);\n"
"    uint c = y >> 6, r = y & 63u, nib = r >> 5, l = r & 31u;\n"
"    uint sc, mn; scale_min_k4(c * 2u + nib, b + 4, &sc, &mn);\n"
"    uchar q = b[16 + c * 32u + l];\n"
"    uint qv = nib ? (q >> 4) : (q & 0xF);\n"
"    x[gid] = d * float(sc) * float(qv) - dmin * float(mn);\n"
"}\n"
"\n"
"kernel void krms(device const float* x [[buffer(0)]], device float* xn [[buffer(1)]],\n"
"                 device const float* w [[buffer(2)]], constant KP& p [[buffer(29)]],\n"
"                 uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],\n"
"                 uint sgid [[simdgroup_index_in_threadgroup]]) {\n"
"    // p.cols = d, p.woff = weight offset (floats) into w, aux = eps bits; tg 256\n"
"    threadgroup float tg[8];\n"
"    float ps = 0.0f;\n"
"    for (uint i = tid; i < p.cols; i += 256u) ps += x[i] * x[i];\n"
"    ps = simd_sum(ps);\n"
"    if (lane == 0) tg[sgid] = ps;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float tot = tg[0] + tg[1] + tg[2] + tg[3] + tg[4] + tg[5] + tg[6] + tg[7];\n"
"    float inv = rsqrt(tot / float(p.cols) + as_type<float>(p.aux));\n"
"    for (uint i = tid; i < p.cols; i += 256u) xn[i] = x[i] * inv * w[p.woff + i];\n"
"}\n"
"\n"
"// fused rmsnorm + Q8_K quant: one dispatch replaces krms+kqblock (small-op fusion).\n"
"// tg = 256 threads = 8 simdgroups; simdgroup sg handles 256-block sg; lane holds 8 elems.\n"
"kernel void krmsq(device const float* x [[buffer(0)]], device float* xn [[buffer(1)]],\n"
"                  device const float* w [[buffer(2)]], device char* qout [[buffer(3)]],\n"
"                  device float* s [[buffer(4)]], device int* sum [[buffer(5)]],\n"
"                  constant KP& p [[buffer(29)]],\n"
"                  uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],\n"
"                  uint sgid [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float tgm[8];\n"
"    float ps = 0.0f;\n"
"    for (uint i = tid; i < p.cols; i += 256u) ps += x[i] * x[i];\n"
"    ps = simd_sum(ps);\n"
"    if (lane == 0) tgm[sgid] = ps;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float tot = tgm[0] + tgm[1] + tgm[2] + tgm[3] + tgm[4] + tgm[5] + tgm[6] + tgm[7];\n"
"    float inv = rsqrt(tot / float(p.cols) + as_type<float>(p.aux));\n"
"    // per-simdgroup block quant (nsb blocks total = cols/256; 8 simdgroups stride)\n"
"    for (uint sb = sgid; sb < p.cols / 256u; sb += 8u) {\n"
"        float v[8]; float amax = 1e-12f;\n"
"        uint base = sb * 256u + lane * 8u;\n"
"        for (uint i = 0; i < 8u; i++) { v[i] = x[base + i] * inv * w[p.woff + base + i]; xn[base + i] = v[i]; amax = max(amax, fabs(v[i])); }\n"
"        amax = simd_max(amax);\n"
"        float sc = amax / 127.0f;\n"
"        if (lane == 0) s[sb] = sc;\n"
"        float psum = 0.0f;\n"
"        for (uint i = 0; i < 8u; i++) { int qi = int(clamp(rint(v[i] / sc), -127.0f, 127.0f)); qout[base + i] = char(qi); psum += float(qi); }\n"
"        // lanes l..l+3 hold one 32-chunk (l*8..l*8+32): pairwise shuffle-down sum\n"
"        psum += simd_shuffle_down(psum, 1);\n"
"        psum += simd_shuffle_down(psum, 2);\n"
"        if ((lane & 3u) == 0u) sum[sb * 8u + lane / 4u] = int(psum);\n"
"    }\n"
"}\n"
"\n"
"kernel void kqblock(device const float* src [[buffer(0)]], device char* q [[buffer(1)]],\n"
"                    device float* s [[buffer(2)]], device int* sum [[buffer(3)]],\n"
"                    constant KP& p [[buffer(29)]], uint tg [[threadgroup_position_in_grid]],\n"
"                    uint lane [[thread_index_in_simdgroup]]) {\n"
"    // one 256-block per threadgroup (32 lanes, 8 elems each)\n"
"    uint base = tg * 256u;\n"
"    float amax = 1e-12f;\n"
"    for (uint i = lane; i < 256u; i += 32u) amax = max(amax, fabs(src[base + i]));\n"
"    amax = simd_max(amax);\n"
"    float sc = amax / 127.0f;\n"
"    if (lane == 0) s[tg] = sc;\n"
"    // quantize: lane i handles elems where (i%32)==lane; per-32 sums via 4-lane groups\n"
"    for (uint j = 0; j < 8u; j++) {\n"
"        uint i = base + j * 32u + lane;\n"
"        int qi = int(clamp(rint(src[i] / sc), -127.0f, 127.0f));\n"
"        q[i] = char(qi);\n"
"        int t = simd_sum(qi);\n"
"        if (lane == 0) sum[tg * 8u + j] = t;\n"
"    }\n"
"}\n"
"\n"
"kernel void kmv4(device const uchar* blob [[buffer(0)]], device const char* xq [[buffer(1)]],\n"
"                 device const float* xs [[buffer(2)]], device const int* xsum [[buffer(3)]],\n"
"                 device float* y [[buffer(4)]], constant KP& p [[buffer(29)]],\n"
"                 uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    if (tg >= p.rows) return;\n"
"    float v = q4k_lane(blob, p.woff, tg, p.bpr, p.nsb, xq, xs, xsum, p.xoff, p.soff, p.moff, lane);\n"
"    v = simd_sum(v);\n"
"    if (lane == 0) { if (p.add) y[p.ooff + tg] += v; else y[p.ooff + tg] = v; }\n"
"}\n"
"\n"
"kernel void kmv6(device const uchar* blob [[buffer(0)]], device const char* xq [[buffer(1)]],\n"
"                 device const float* xs [[buffer(2)]], device float* y [[buffer(4)]],\n"
"                 constant KP& p [[buffer(29)]],\n"
"                 uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    if (tg >= p.rows) return;\n"
"    float v = q6k_lane(blob, p.woff, tg, p.bpr, p.nsb, xq, xs, p.xoff, p.soff, lane);\n"
"    v = simd_sum(v);\n"
"    if (lane == 0) { if (p.add) y[p.ooff + tg] += v; else y[p.ooff + tg] = v; }\n"
"}\n"
"\n"
"kernel void kqknorm(device float* q [[buffer(0)]], device float* k [[buffer(1)]],\n"
"                    device const float* v [[buffer(2)]], device const float* w [[buffer(3)]],\n"
"                    device float* kc [[buffer(4)]], device float* vc [[buffer(5)]],\n"
"                    constant KP& p [[buffer(29)]],\n"
"                    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    // p.rows=nh, p.cols=hd, p.aux=nkv, p.pos, p.woff=q_norm off (floats), p.xoff=k_norm off\n"
"    // p.soff = rope base bits, p.moff = eps bits; heads: [0,nh) = q, [nh, nh+nkv) = k\n"
"    uint h = tg; uint hd = p.cols; bool isq = h < p.rows;\n"
"    device float* vec = isq ? (q + h * hd) : (k + (h - p.rows) * hd);\n"
"    device const float* nw = w + (isq ? p.woff : p.xoff);\n"
"    float ps = 0.0f;\n"
"    for (uint i = lane; i < hd; i += 32u) ps += vec[i] * vec[i];\n"
"    float inv = rsqrt(simd_sum(ps) / float(hd) + as_type<float>(p.moff));\n"
"    for (uint i = lane; i < hd; i += 32u) vec[i] *= inv * nw[i];\n"
"    simdgroup_barrier(mem_flags::mem_device);\n"
"    uint half_ = hd / 2u;\n"
"    float base = as_type<float>(p.soff);\n"
"    for (uint i = lane; i < half_; i += 32u) {\n"
"        float theta = float(p.pos) * pow(base, -2.0f * float(i) / float(hd));\n"
"        float co = cos(theta), si = sin(theta);\n"
"        float a = vec[i], b = vec[half_ + i];\n"
"        vec[i] = a * co - b * si;\n"
"        vec[half_ + i] = b * co + a * si;\n"
"    }\n"
"    if (!isq) {\n"
"        uint hh = h - p.rows;\n"
"        simdgroup_barrier(mem_flags::mem_device);\n"
"        for (uint i = lane; i < hd; i += 32u) {\n"
"            kc[(p.pos * p.aux + hh) * hd + i] = vec[i];\n"
"            vc[(p.pos * p.aux + hh) * hd + i] = v[hh * hd + i];\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void kattn(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]],\n"
"                  device const float* vc [[buffer(2)]], device float* ao [[buffer(3)]],\n"
"                  constant KP& p [[buffer(29)]],\n"
"                  uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\n"
"                  uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) {\n"
"    // one q-head per tg of 128 threads, TWO-PASS: (1) all scores in parallel into\n"
"    // threadgroup memory + softmax, (2) dims-parallel weighted-V accumulate.\n"
"    // p.rows=nh, p.cols=hd, p.aux=nkv, p.pos. Requires pos+1 <= 4096.\n"
"    threadgroup float sc[4096];\n"
"    threadgroup float red[4];\n"
"    uint h = tg, hd = p.cols, nkv = p.aux;\n"
"    uint kvh = h / (p.rows / nkv);\n"
"    float scale = rsqrt(float(hd));\n"
"    uint T = p.pos + 1;\n"
"    device const float* qh = q + h * hd;\n"
"    // pass 1: thread t computes score[t] (full 128-dim dot, no cross-lane deps)\n"
"    float lmax = -INFINITY;\n"
"    for (uint t = tid; t < T; t += 128u) {\n"
"        device const float* kp = kc + (t * nkv + kvh) * hd;\n"
"        float4 a4 = 0.0f;\n"
"        device const float4* q4 = (device const float4*)qh;\n"
"        device const float4* k4 = (device const float4*)kp;\n"
"        for (uint j = 0; j < hd / 4u; j++) a4 = fma(q4[j], k4[j], a4);\n"
"        float s = (a4.x + a4.y + a4.z + a4.w) * scale;\n"
"        sc[t] = s;\n"
"        lmax = max(lmax, s);\n"
"    }\n"
"    // reduce max over 4 simdgroups\n"
"    lmax = simd_max(lmax);\n"
"    if (lane == 0) red[sgid] = lmax;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float m = max(max(red[0], red[1]), max(red[2], red[3]));\n"
"    // exp + denominator\n"
"    float lden = 0.0f;\n"
"    for (uint t = tid; t < T; t += 128u) { float e = exp(sc[t] - m); sc[t] = e; lden += e; }\n"
"    lden = simd_sum(lden);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (lane == 0) red[sgid] = lden;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float inv = 1.0f / (red[0] + red[1] + red[2] + red[3]);\n"
"    // pass 2: thread owns dim i = tid (hd=128), streams V\n"
"    uint i = tid;\n"
"    float acc = 0.0f;\n"
"    device const float* vbase = vc + kvh * hd + i;\n"
"    for (uint t = 0; t < T; t++) acc += sc[t] * vbase[t * nkv * hd];\n"
"    ao[h * hd + i] = acc * inv;\n"
"}\n"
"\n"
"kernel void krouter(device const float* xn [[buffer(0)]], device const float* w [[buffer(1)]],\n"
"                    device float* logits [[buffer(2)]], constant KP& p [[buffer(29)]],\n"
"                    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    // p.rows = ne, p.cols = d, p.woff = gate_inp float offset\n"
"    device const float* row = w + p.woff + tg * p.cols;\n"
"    float ps = 0.0f;\n"
"    for (uint i = lane; i < p.cols; i += 32u) ps += row[i] * xn[i];\n"
"    ps = simd_sum(ps);\n"
"    if (lane == 0) logits[tg] = ps;\n"
"}\n"
"\n"
"kernel void ktopk(device const float* logits [[buffer(0)]], device uint* top [[buffer(1)]],\n"
"                  device float* wts [[buffer(2)]], constant KP& p [[buffer(29)]],\n"
"                  uint lane [[thread_index_in_simdgroup]]) {\n"
"    // 32-lane parallel top-k: each lane owns ne/32 logits; k rounds of simd argmax.\n"
"    uint ne = p.rows, k = p.aux;\n"
"    uint per = ne / 32u;\n"
"    float lv[16]; // per-lane copy (ne<=512)\n"
"    for (uint i = 0; i < per; i++) lv[i] = logits[lane * per + i];\n"
"    float mx = -INFINITY;\n"
"    for (uint j = 0; j < k; j++) {\n"
"        float bv = -INFINITY; uint bi = 0;\n"
"        for (uint i = 0; i < per; i++) if (lv[i] > bv) { bv = lv[i]; bi = lane * per + i; }\n"
"        float gmax = simd_max(bv);\n"
"        uint gidx = simd_min(bv == gmax ? bi : 0xFFFFFFFFu);\n"
"        if (lane == 0) top[j] = gidx;\n"
"        if (gidx >= lane * per && gidx < (lane + 1u) * per) lv[gidx - lane * per] = -INFINITY;\n"
"        if (gmax > mx) mx = gmax;\n"
"    }\n"
"    if (lane == 0) {\n"
"        float den = 0.0f;\n"
"        for (uint j = 0; j < k; j++) { wts[j] = exp(logits[top[j]] - mx); den += wts[j]; }\n"
"        for (uint j = 0; j < k; j++) wts[j] /= den;\n"
"    }\n"
"}\n"
"\n"
"kernel void kglu(device const uchar* blob [[buffer(0)]], device const char* xq [[buffer(1)]],\n"
"                 device const float* xs [[buffer(2)]], device const int* xsum [[buffer(3)]],\n"
"                 device const uint* top [[buffer(4)]], device float* hbuf [[buffer(5)]],\n"
"                 constant KP& p [[buffer(29)]],\n"
"                 uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    // p.rows = topk*ff, p.aux = ff, p.woff = gate off, p.ooff reused = up off HIGH bits? no:\n"
"    // gate offset = p.woff, up offset = p.woff + p.xoff*(ulong)? -> pass up off via soff/moff pair\n"
"    if (tg >= p.rows) return;\n"
"    uint slot = tg / p.aux, r = tg % p.aux;\n"
"    uint row = top[slot] * p.aux + r;\n"
"    ulong upoff = (ulong(p.moff) << 32) | ulong(p.soff);\n"
"    float g = q4k_lane(blob, p.woff, row, p.bpr, p.nsb, xq, xs, xsum, 0, 0, 0, lane);\n"
"    float u = q4k_lane(blob, upoff, row, p.bpr, p.nsb, xq, xs, xsum, 0, 0, 0, lane);\n"
"    g = simd_sum(g); u = simd_sum(u);\n"
"    if (lane == 0) hbuf[tg] = (g / (1.0f + exp(-g))) * u;\n"
"}\n"
"\n"
"// 4-rows-per-simdgroup gate/up+silu: activation loads amortized over 8 dot streams.\n"
"kernel void kglu4(device const uchar* blob [[buffer(0)]], device const char* xq [[buffer(1)]],\n"
"                  device const float* xs [[buffer(2)]], device const int* xsum [[buffer(3)]],\n"
"                  device const uint* top [[buffer(4)]], device float* hbuf [[buffer(5)]],\n"
"                  constant KP& p [[buffer(29)]],\n"
"                  uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    // tg handles rows [4*tg, 4*tg+4) of the topk*ff logical rows; ff % 4 == 0 so one slot\n"
"    uint lr0 = tg * 4u;\n"
"    if (lr0 >= p.rows) return;\n"
"    uint slot = lr0 / p.aux, r0 = lr0 % p.aux;\n"
"    ulong grow0 = (ulong)(top[slot] * p.aux + r0);\n"
"    ulong upoff = (ulong(p.moff) << 32) | ulong(p.soff);\n"
"    device const uchar* gbase = blob + p.woff + grow0 * p.bpr;\n"
"    device const uchar* ubase = blob + upoff + grow0 * p.bpr;\n"
"    float4 accg = 0.0f, accu = 0.0f;\n"
"    uint units = p.nsb * 8u;\n"
"    for (uint u = lane; u < units; u += 32u) {\n"
"        uint sb = u >> 3, j = u & 7u;\n"
"        uint c = j >> 1, nib = j & 1u;\n"
"        device const char4* x = (device const char4*)(xq + sb * 256u + c * 64u + nib * 32u);\n"
"        float4 xf[8];\n"
"        for (uint i = 0; i < 8u; i++) xf[i] = float4(x[i]);\n"
"        int xsm = xsum[sb * 8u + j];\n"
"        float ysb = xs[sb];\n"
"        for (uint r = 0; r < 4u; r++) {\n"
"            for (uint m = 0; m < 2u; m++) {\n"
"                device const uchar* b = (m ? ubase : gbase) + (ulong)r * p.bpr + sb * 144u;\n"
"                float d = half_at(b), dmin = half_at(b + 2);\n"
"                uint sc, mn; scale_min_k4(j, b + 4, &sc, &mn);\n"
"                device const uchar4* qp = (device const uchar4*)(b + 16 + c * 32u);\n"
"                float4 a4 = 0.0f;\n"
"                for (uint i = 0; i < 8u; i++) {\n"
"                    uchar4 qb = qp[i];\n"
"                    uchar4 qv = nib ? (qb >> 4) : (qb & (uchar)0xF);\n"
"                    a4 = fma(float4(qv), xf[i], a4);\n"
"                }\n"
"                float s0 = a4.x + a4.y + a4.z + a4.w;\n"
"                float contrib = ysb * (d * float(sc) * s0 - dmin * float(int(mn) * xsm));\n"
"                if (m) accu[r] += contrib; else accg[r] += contrib;\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint r = 0; r < 4u; r++) {\n"
"        float g = simd_sum(accg[r]), u2 = simd_sum(accu[r]);\n"
"        if (lane == 0 && lr0 + r < p.rows) hbuf[lr0 + r] = (g / (1.0f + exp(-g))) * u2;\n"
"    }\n"
"}\n"
"\n"
"kernel void kdown(device const uchar* blob [[buffer(0)]], device const char* hq [[buffer(1)]],\n"
"                  device const float* hs [[buffer(2)]], device const uint* top [[buffer(3)]],\n"
"                  device const float* wts [[buffer(4)]], device float* x [[buffer(5)]],\n"
"                  constant KP& p [[buffer(29)]],\n"
"                  uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    // p.rows = d, p.cols = ff, p.aux = topk, p.woff = down off, q6k\n"
"    // lane strides the flattened (expert, unit) space: independent streams -> ILP\n"
"    if (tg >= p.rows) return;\n"
"    uint nsb = p.cols / 256u;\n"
"    uint units = nsb * 16u, total = p.aux * units;\n"
"    float acc = 0.0f;\n"
"    for (uint w = lane; w < total; w += 32u) {\n"
"        uint e = w / units, u = w % units;\n"
"        uint sb = u >> 4, g = u & 15u;\n"
"        device const uchar* b = blob + p.woff + (ulong)(top[e] * p.rows + tg) * p.bpr + sb * 210u;\n"
"        uint h = g >> 3, rem = g & 7u, qg = rem >> 1, is = rem & 1u;\n"
"        uint lbase = is * 16u;\n"
"        device const char4* xv = (device const char4*)(hq + e * p.cols + sb * 256u + h * 128u + qg * 32u + lbase);\n"
"        device const uchar4* ql4 = (device const uchar4*)(b + h * 64u + (qg & 1u) * 32u + lbase);\n"
"        device const uchar4* qh4 = (device const uchar4*)(b + 128 + h * 32u + lbase);\n"
"        int sc = int(*(device const char*)(b + 192 + g));\n"
"        float d2 = half_at(b + 208);\n"
"        bool hiNib = (qg >= 2);\n"
"        uchar hshift = (uchar)(qg * 2u);\n"
"        float4 a4 = 0.0f;\n"
"        for (uint i = 0; i < 4u; i++) {\n"
"            uchar4 lo = hiNib ? (ql4[i] >> 4) : (ql4[i] & (uchar)0xF);\n"
"            uchar4 hi = (qh4[i] >> hshift) & (uchar)3;\n"
"            int4 qv = int4(lo | (hi << 4)) - 32;\n"
"            a4 = fma(float4(qv), float4(xv[i]), a4);\n"
"        }\n"
"        float s0 = a4.x + a4.y + a4.z + a4.w;\n"
"        acc += wts[e] * hs[e * nsb + sb] * d2 * float(sc) * s0;\n"
"    }\n"
"    acc = simd_sum(acc);\n"
"    if (lane == 0) x[tg] += acc;\n"
"}\n"
"\n"
"kernel void kdown4(device const uchar* blob [[buffer(0)]], device const char* hq [[buffer(1)]],\n"
"                   device const float* hs [[buffer(2)]], device const uint* top [[buffer(3)]],\n"
"                   device const float* wts [[buffer(4)]], device float* x [[buffer(5)]],\n"
"                   device const int* hm [[buffer(6)]], constant KP& p [[buffer(29)]],\n"
"                   uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
"    // Q4_K variant of kdown (down tensors are MIXED 12/14 across layers)\n"
"    // lane strides the flattened (expert, unit) space: independent streams -> ILP\n"
"    if (tg >= p.rows) return;\n"
"    uint nsb = p.cols / 256u;\n"
"    uint units = nsb * 8u, total = p.aux * units;\n"
"    float acc = 0.0f;\n"
"    for (uint w = lane; w < total; w += 32u) {\n"
"        uint e = w / units, u = w % units;\n"
"        uint sb = u >> 3, j = u & 7u;\n"
"        uint c = j >> 1, nib = j & 1u;\n"
"        device const uchar* b = blob + p.woff + (ulong)(top[e] * p.rows + tg) * p.bpr + sb * 144u;\n"
"        float d2 = half_at(b), dmin = half_at(b + 2);\n"
"        uint sc, mn; scale_min_k4(j, b + 4, &sc, &mn);\n"
"        device const uchar4* qp = (device const uchar4*)(b + 16 + c * 32u);\n"
"        device const char4* xv = (device const char4*)(hq + e * p.cols + sb * 256u + c * 64u + nib * 32u);\n"
"        float4 a4 = 0.0f;\n"
"        for (uint i = 0; i < 8u; i++) {\n"
"            uchar4 qb = qp[i];\n"
"            uchar4 qv = nib ? (qb >> 4) : (qb & (uchar)0xF);\n"
"            a4 = fma(float4(qv), float4(xv[i]), a4);\n"
"        }\n"
"        float s0 = a4.x + a4.y + a4.z + a4.w;\n"
"        int xsm = hm[e * (p.cols / 32u) + sb * 8u + j];\n"
"        acc += wts[e] * hs[e * nsb + sb] * (d2 * float(sc) * s0 - dmin * float(int(mn) * xsm));\n"
"    }\n"
"    acc = simd_sum(acc);\n"
"    if (lane == 0) x[tg] += acc;\n"
"}\n"
"\n"
"kernel void kamax1(device const float* v [[buffer(0)]], device float* pval [[buffer(1)]],\n"
"                   device uint* pidx [[buffer(2)]], constant KP& p [[buffer(29)]],\n"
"                   uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\n"
"                   uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) {\n"
"    // p.rows = n, chunk = 1024 per tg of 256 threads\n"
"    threadgroup float tv[8]; threadgroup uint ti[8];\n"
"    uint base = tg * 1024u;\n"
"    float bv = -INFINITY; uint bi = 0;\n"
"    for (uint i = tid; i < 1024u; i += 256u) {\n"
"        uint idx = base + i;\n"
"        float val = idx < p.rows ? v[idx] : -INFINITY;\n"
"        if (val > bv) { bv = val; bi = idx; }\n"
"    }\n"
"    float mv = simd_max(bv);\n"
"    uint mi = simd_min(bv == mv ? bi : 0xFFFFFFFFu);\n"
"    if (lane == 0) { tv[sgid] = mv; ti[sgid] = mi; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        float best = tv[0]; uint besti = ti[0];\n"
"        for (uint s = 1; s < 8u; s++) if (tv[s] > best) { best = tv[s]; besti = ti[s]; }\n"
"        pval[tg] = best; pidx[tg] = besti;\n"
"    }\n"
"}\n"
"\n"
"kernel void kamax2(device const float* pval [[buffer(0)]], device const uint* pidx [[buffer(1)]],\n"
"                   device uint* out [[buffer(2)]], constant KP& p [[buffer(29)]],\n"
"                   uint gid [[thread_position_in_grid]]) {\n"
"    if (gid != 0) return;\n"
"    float best = -INFINITY; uint besti = 0;\n"
"    for (uint i = 0; i < p.rows; i++) if (pval[i] > best) { best = pval[i]; besti = pidx[i]; }\n"
"    out[p.aux] = besti;   // per-slot so chained cmdbufs never clobber unread tokens\n"
"}\n";

// ---------------- host-side param struct (must match MSL KP) ----------------
typedef struct { uint64_t woff; uint32_t rows, cols, bpr, nsb, xoff, soff, moff, ooff, add, pos, aux; } KP;

// ---------------- tensor index ----------------
typedef struct { uint32_t type; uint64_t off; uint64_t ne; } TEnt;

// ---------------- CPU reference (scalar port of the verified kernels) ----------------
static void ref_scale_min_k4(int j, const uint8_t *s, uint32_t *sc, uint32_t *mn) {
    if (j < 4) { *sc = s[j] & 63; *mn = s[j+4] & 63; }
    else { *sc = (s[j+4] & 0xF) | ((s[j-4] >> 6) << 4); *mn = (s[j+4] >> 4) | ((s[j] >> 6) << 4); }
}
static float half2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF, out;
    if (e == 0) { if (!m) out = s << 31; else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; out = (s << 31) | (e << 23) | (m << 13); } }
    else if (e == 31) out = (s << 31) | 0x7F800000 | (m << 13);
    else out = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &out, 4); return f;
}
static float ref_q4k_dot(const uint8_t *row, const int8_t *xq, const float *xs, const int32_t *xsum, int cols) {
    float sumf = 0;
    for (int sb = 0; sb < cols / 256; sb++) {
        const uint8_t *b = row + sb * 144;
        float d = half2f(*(uint16_t*)b), dmin = half2f(*(uint16_t*)(b + 2));
        int64_t acc = 0, mint = 0;
        for (int j = 0; j < 8; j++) {
            uint32_t sc, mn; ref_scale_min_k4(j, b + 4, &sc, &mn);
            int c = j >> 1, nib = j & 1;
            const uint8_t *q = b + 16 + c * 32;
            const int8_t *x = xq + sb * 256 + c * 64 + nib * 32;
            int s0 = 0;
            for (int i = 0; i < 32; i++) { int qv = nib ? (q[i] >> 4) : (q[i] & 0xF); s0 += qv * x[i]; }
            acc += (int64_t)sc * s0;
            mint += (int64_t)mn * xsum[sb * 8 + j];
        }
        sumf += xs[sb] * (d * (float)acc - dmin * (float)mint);
    }
    return sumf;
}
static float ref_q6k_dot(const uint8_t *row, const int8_t *xq, const float *xs, int cols) {
    float sumf = 0;
    for (int sb = 0; sb < cols / 256; sb++) {
        const uint8_t *b = row + sb * 210;
        const uint8_t *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
        float d = half2f(*(uint16_t*)(b + 208));
        int64_t acc = 0;
        for (int g = 0; g < 16; g++) {
            int h = g >> 3, rem = g & 7, qg = rem >> 1, is = rem & 1;
            int s0 = 0;
            for (int i = 0; i < 16; i++) {
                int l = is * 16 + i;
                const uint8_t *qlh = ql + h * 64 + (qg & 1) * 32;
                int lo = (qg == 0 || qg == 1) ? (qlh[l & 31] & 0xF) : (qlh[l & 31] >> 4);
                int hi = (qh[h * 32 + l] >> (qg * 2)) & 3;
                s0 += ((lo | (hi << 4)) - 32) * xq[sb * 256 + h * 128 + qg * 32 + l];
            }
            acc += (int64_t)sc[g] * s0;
        }
        sumf += xs[sb] * d * (float)acc;
    }
    return sumf;
}

int main(int argc, char **argv) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 3) { fprintf(stderr, "usage: %s <data-dir> <blob> [prompt|--verify] [ngen]\n", argv[0]); return 1; }
    const char *dir = argv[1], *blobp = argv[2];
    BOOL verify = argc > 3 && !strcmp(argv[3], "--verify");
    const char *prompt = (argc > 3 && !verify) ? argv[3] : "Write a haiku about CPUs.";
    int ngen = argc > 4 ? atoi(argv[4]) : 64;

    // ---- config ----
    {
        char path[1024]; snprintf(path, sizeof path, "%s/config.txt", dir);
        FILE *f = fopen(path, "r"); if (!f) { perror("config"); return 1; }
        char k[64]; double v; char sv[64];
        while (fscanf(f, "%63s %63s", k, sv) == 2) {
            v = atof(sv);
            if (!strcmp(k, "n_embd")) C.d = v; else if (!strcmp(k, "n_layer")) C.nl = v;
            else if (!strcmp(k, "n_head")) C.nh = v; else if (!strcmp(k, "n_head_kv")) C.nkv = v;
            else if (!strcmp(k, "head_dim")) C.hd = v; else if (!strcmp(k, "n_expert")) C.ne = v;
            else if (!strcmp(k, "n_expert_used")) C.topk = v; else if (!strcmp(k, "n_ff_exp")) C.ff = v;
            else if (!strcmp(k, "rope_freq_base")) C.rope_base = v; else if (!strcmp(k, "rms_eps")) C.rms_eps = v;
            else if (!strcmp(k, "n_vocab")) C.nvocab = v;
        }
        fclose(f);
    }
    fprintf(stderr, "cfg: d=%d nl=%d nh=%d nkv=%d hd=%d ne=%d topk=%d ff=%d vocab=%d\n",
            C.d, C.nl, C.nh, C.nkv, C.hd, C.ne, C.topk, C.ff, C.nvocab);

    // ---- manifest ----
    NSMutableDictionary *idx = [NSMutableDictionary new];
    uint64_t data_start = 0;
    {
        char path[1024]; snprintf(path, sizeof path, "%s/manifest.txt", dir);
        FILE *f = fopen(path, "r"); if (!f) { perror("manifest"); return 1; }
        char name[256]; unsigned long long a, b, c2;
        fscanf(f, "%255s %llu", name, &a); data_start = a;   // "data_start N"
        while (fscanf(f, "%255s %llu %llu %llu", name, &a, &b, &c2) == 4) {
            TEnt e = { (uint32_t)a, data_start + b, c2 };
            [idx setObject:[NSData dataWithBytes:&e length:sizeof e] forKey:[NSString stringWithUTF8String:name]];
        }
        fclose(f);
    }
    TEnt (^tent)(NSString *) = ^TEnt(NSString *n) {
        NSData *d = idx[n];
        if (!d) { fprintf(stderr, "missing tensor %s\n", n.UTF8String); exit(1); }
        TEnt e; [d getBytes:&e length:sizeof e]; return e;
    };

    // ---- mmap blob, wrap zero-copy ----
    double t0 = now_s();
    int fd = open(blobp, O_RDONLY);
    if (fd < 0) { perror("blob"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t page = 16384, maplen = ((size_t)st.st_size + page - 1) & ~(page - 1);
    void *map = mmap(NULL, maplen, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLBuffer> blob = [dev newBufferWithBytesNoCopy:map length:maplen options:MTLResourceStorageModeShared deallocator:nil];
    if (!blob) { fprintf(stderr, "newBufferWithBytesNoCopy failed\n"); return 1; }

    NSError *err = nil;
    MTLCompileOptions *opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:MSL] options:opts error:&err];
    if (!lib) { fprintf(stderr, "MSL compile: %s\n", err.description.UTF8String); return 1; }
    #define PSO(n) ({ id<MTLComputePipelineState> _p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@#n] error:&err]; if (!_p) { fprintf(stderr, "pso %s: %s\n", #n, err.description.UTF8String); exit(1); } _p; })
    id<MTLComputePipelineState> Pembed = PSO(kembed), Prms = PSO(krms), Prmsq = PSO(krmsq), Pqb = PSO(kqblock),
        Pmv4x4 = PSO(kmv4x4), Pmv6x4 = PSO(kmv6x4), Pglu4 = PSO(kglu4),
        Pmv4 = PSO(kmv4), Pmv6 = PSO(kmv6), Pqkn = PSO(kqknorm), Pattn = PSO(kattn),
        Prtr = PSO(krouter), Ptopk = PSO(ktopk), Pglu = PSO(kglu), Pdown = PSO(kdown),
        Pdown4 = PSO(kdown4), Pam1 = PSO(kamax1), Pam2 = PSO(kamax2);

    // ---- small f32 tensors packed into one buffer ----
    // layout per layer: attn_norm(d) ffn_norm(d) q_norm(hd) k_norm(hd) gate_inp(ne*d); then out_norm(d)
    uint64_t perLayerF = (uint64_t)C.d * 2 + C.hd * 2 + (uint64_t)C.ne * C.d;
    uint64_t smallN = perLayerF * C.nl + C.d;
    id<MTLBuffer> smalls = [dev newBufferWithLength:smallN * 4 options:MTLResourceStorageModeShared];
    float *sm = (float *)smalls.contents;
    const uint8_t *bb = (const uint8_t *)map;
    for (int il = 0; il < C.nl; il++) {
        uint64_t base = perLayerF * il;
        struct { const char *fmt; uint64_t off; uint64_t n; } parts[4] = {
            {"blk.%d.attn_norm.weight", 0, (uint64_t)C.d}, {"blk.%d.ffn_norm.weight", (uint64_t)C.d, (uint64_t)C.d},
            {"blk.%d.attn_q_norm.weight", (uint64_t)C.d * 2, (uint64_t)C.hd}, {"blk.%d.attn_k_norm.weight", (uint64_t)C.d * 2 + C.hd, (uint64_t)C.hd},
        };
        for (int pi = 0; pi < 4; pi++) {
            char nm[128]; snprintf(nm, sizeof nm, parts[pi].fmt, il);
            TEnt e = tent([NSString stringWithUTF8String:nm]);
            memcpy(sm + base + parts[pi].off, bb + e.off, parts[pi].n * 4);
        }
        char nm[128]; snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", il);
        TEnt e = tent([NSString stringWithUTF8String:nm]);
        memcpy(sm + base + (uint64_t)C.d * 2 + C.hd * 2, bb + e.off, (uint64_t)C.ne * C.d * 4);
    }
    { TEnt e = tent(@"output_norm.weight"); memcpy(sm + perLayerF * C.nl, bb + e.off, (uint64_t)C.d * 4); }

    // per-layer weight offsets
    typedef struct { uint64_t q, k, v, o, gate, up, down; uint32_t vtype, dtype; } LW;
    LW *lw = calloc(C.nl, sizeof(LW));
    for (int il = 0; il < C.nl; il++) {
        #define T(f) ({ char nm[128]; snprintf(nm, sizeof nm, f, il); tent([NSString stringWithUTF8String:nm]); })
        lw[il].q = T("blk.%d.attn_q.weight").off; lw[il].k = T("blk.%d.attn_k.weight").off;
        TEnt ev = T("blk.%d.attn_v.weight"); lw[il].v = ev.off; lw[il].vtype = ev.type;
        lw[il].o = T("blk.%d.attn_output.weight").off;
        lw[il].gate = T("blk.%d.ffn_gate_exps.weight").off; lw[il].up = T("blk.%d.ffn_up_exps.weight").off;
        TEnt ed = T("blk.%d.ffn_down_exps.weight"); lw[il].down = ed.off; lw[il].dtype = ed.type;
        #undef T
    }
    TEnt embedT = tent(@"token_embd.weight");
    TEnt headT = tent(@"output.weight");
    fprintf(stderr, "loaded (mmap zero-copy) in %.2fs\n", now_s() - t0);

    // ---- activation buffers ----
    #define NB(n) [dev newBufferWithLength:(n) options:MTLResourceStorageModeShared]
    int d = C.d, nh = C.nh, nkv = C.nkv, hd = C.hd, ff = C.ff, topk = C.topk;
    id<MTLBuffer> x = NB(d * 4), xn = NB(d * 4), xq = NB(d), xs = NB(d / 256 * 4), xsum = NB(d / 32 * 4);
    id<MTLBuffer> q = NB(nh * hd * 4), k = NB(nkv * hd * 4), v = NB(nkv * hd * 4), ao = NB(nh * hd * 4);
    id<MTLBuffer> aq = NB(nh * hd), asc = NB(nh * hd / 256 * 4), asm_ = NB(nh * hd / 32 * 4);
    id<MTLBuffer> hbuf = NB(topk * ff * 4), hq = NB(topk * ff), hs = NB(topk * ff / 256 * 4), hm = NB(topk * ff / 32 * 4);
    id<MTLBuffer> rlog = NB(C.ne * 4), top = NB(topk * 4), wts = NB(topk * 4);
    id<MTLBuffer> vlog = NB((uint64_t)C.nvocab * 4);
    int namax = (C.nvocab + 1023) / 1024;
    id<MTLBuffer> pval = NB(namax * 4), pidx = NB(namax * 4), outtok = NB(4096 * 4);
    id<MTLBuffer> kcb = NB((uint64_t)C.nl * MAXSEQ * nkv * hd * 4), vcb = NB((uint64_t)C.nl * MAXSEQ * nkv * hd * 4);

    float eps = C.rms_eps, rbase = C.rope_base;
    uint32_t epsb, rbaseb; memcpy(&epsb, &eps, 4); memcpy(&rbaseb, &rbase, 4);

    // ---- encode pieces (embed / one layer / head) ----
    #define DISP(pso, tgs, tpt) [e setComputePipelineState:pso]; [e setBytes:&p length:sizeof p atIndex:29]; \
        [e dispatchThreadgroups:MTLSizeMake(tgs,1,1) threadsPerThreadgroup:MTLSizeMake(tpt,1,1)]
    void (^encEmbed)(id<MTLComputeCommandEncoder>, uint32_t, uint32_t, BOOL) = ^(id<MTLComputeCommandEncoder> e, uint32_t tokid, uint32_t slot, BOOL chained) {
        KP p; memset(&p, 0, sizeof p);
        p.woff = embedT.off; p.bpr = d / 256 * 144; p.cols = d; p.aux = tokid; p.add = chained ? 1 : 0; p.pos = slot;
        [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:x offset:0 atIndex:1]; [e setBuffer:outtok offset:0 atIndex:2];
        DISP(Pembed, (d + 255) / 256, 256);
    };
    void (^encLayer)(id<MTLComputeCommandEncoder>, int, uint32_t) = ^(id<MTLComputeCommandEncoder> e, int il, uint32_t pos) {
        KP p; memset(&p, 0, sizeof p);
        {
            uint64_t smBase = perLayerF * il;
            uint64_t kcOff = (uint64_t)il * MAXSEQ * nkv * hd * 4, vcOff = kcOff;
            // attn rmsnorm + quant (fused)
            memset(&p, 0, sizeof p); p.cols = d; p.woff = smBase; p.aux = epsb;
            [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:xn offset:0 atIndex:1]; [e setBuffer:smalls offset:0 atIndex:2];
            [e setBuffer:xq offset:0 atIndex:3]; [e setBuffer:xs offset:0 atIndex:4]; [e setBuffer:xsum offset:0 atIndex:5];
            DISP(Prmsq, 1, 256);
            // qkv
            [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:xq offset:0 atIndex:1]; [e setBuffer:xs offset:0 atIndex:2]; [e setBuffer:xsum offset:0 atIndex:3];
            memset(&p, 0, sizeof p); p.woff = lw[il].q; p.rows = nh * hd; p.cols = d; p.bpr = d / 256 * 144; p.nsb = d / 256;
            [e setBuffer:q offset:0 atIndex:4]; DISP(Pmv4, nh * hd, 32);
            p.woff = lw[il].k; p.rows = nkv * hd;
            [e setBuffer:k offset:0 atIndex:4]; DISP(Pmv4, nkv * hd, 32);
            p.woff = lw[il].v; p.rows = nkv * hd;
            [e setBuffer:v offset:0 atIndex:4];
            if (lw[il].vtype == 14) { p.bpr = d / 256 * 210; DISP(Pmv6, nkv * hd, 32); }
            else { DISP(Pmv4, nkv * hd, 32); }
            // qk-norm + rope + kv write
            memset(&p, 0, sizeof p); p.rows = nh; p.cols = hd; p.aux = nkv; p.pos = pos;
            p.woff = smBase + (uint64_t)d * 2; p.xoff = (uint32_t)(smBase + (uint64_t)d * 2 + hd);
            p.soff = rbaseb; p.moff = epsb;
            [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:k offset:0 atIndex:1]; [e setBuffer:v offset:0 atIndex:2];
            [e setBuffer:smalls offset:0 atIndex:3]; [e setBuffer:kcb offset:kcOff atIndex:4]; [e setBuffer:vcb offset:vcOff atIndex:5];
            DISP(Pqkn, nh + nkv, 32);
            // attention
            memset(&p, 0, sizeof p); p.rows = nh; p.cols = hd; p.aux = nkv; p.pos = pos;
            [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:kcb offset:kcOff atIndex:1]; [e setBuffer:vcb offset:vcOff atIndex:2]; [e setBuffer:ao offset:0 atIndex:3];
            DISP(Pattn, nh, 128);
            // quant ao
            memset(&p, 0, sizeof p);
            [e setBuffer:ao offset:0 atIndex:0]; [e setBuffer:aq offset:0 atIndex:1]; [e setBuffer:asc offset:0 atIndex:2]; [e setBuffer:asm_ offset:0 atIndex:3];
            DISP(Pqb, nh * hd / 256, 32);
            // o-proj (+ residual into x)
            memset(&p, 0, sizeof p); p.woff = lw[il].o; p.rows = d; p.cols = nh * hd; p.bpr = nh * hd / 256 * 144; p.nsb = nh * hd / 256; p.add = 1;
            [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:aq offset:0 atIndex:1]; [e setBuffer:asc offset:0 atIndex:2]; [e setBuffer:asm_ offset:0 atIndex:3];
            [e setBuffer:x offset:0 atIndex:4]; DISP(Pmv4, d, 32);
            // ffn rmsnorm + quant (fused)
            memset(&p, 0, sizeof p); p.cols = d; p.woff = smBase + d; p.aux = epsb;
            [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:xn offset:0 atIndex:1]; [e setBuffer:smalls offset:0 atIndex:2];
            [e setBuffer:xq offset:0 atIndex:3]; [e setBuffer:xs offset:0 atIndex:4]; [e setBuffer:xsum offset:0 atIndex:5];
            DISP(Prmsq, 1, 256);
            // router + topk
            memset(&p, 0, sizeof p); p.rows = C.ne; p.cols = d; p.woff = smBase + (uint64_t)d * 2 + hd * 2;
            [e setBuffer:xn offset:0 atIndex:0]; [e setBuffer:smalls offset:0 atIndex:1]; [e setBuffer:rlog offset:0 atIndex:2];
            DISP(Prtr, C.ne, 32);
            memset(&p, 0, sizeof p); p.rows = C.ne; p.aux = topk;
            [e setBuffer:rlog offset:0 atIndex:0]; [e setBuffer:top offset:0 atIndex:1]; [e setBuffer:wts offset:0 atIndex:2];
            DISP(Ptopk, 1, 32);
            // gate/up + silu
            memset(&p, 0, sizeof p); p.rows = topk * ff; p.aux = ff; p.bpr = d / 256 * 144; p.nsb = d / 256;
            p.woff = lw[il].gate; p.soff = (uint32_t)(lw[il].up & 0xFFFFFFFFu); p.moff = (uint32_t)(lw[il].up >> 32);
            [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:xq offset:0 atIndex:1]; [e setBuffer:xs offset:0 atIndex:2]; [e setBuffer:xsum offset:0 atIndex:3];
            [e setBuffer:top offset:0 atIndex:4]; [e setBuffer:hbuf offset:0 atIndex:5];
            DISP(Pglu4, (topk * ff + 3) / 4, 32);
            // quant hbuf
            memset(&p, 0, sizeof p);
            [e setBuffer:hbuf offset:0 atIndex:0]; [e setBuffer:hq offset:0 atIndex:1]; [e setBuffer:hs offset:0 atIndex:2]; [e setBuffer:hm offset:0 atIndex:3];
            DISP(Pqb, topk * ff / 256, 32);
            // down + weighted residual accumulate (down tensors are MIXED Q4_K/Q6_K per layer)
            memset(&p, 0, sizeof p); p.rows = d; p.cols = ff; p.aux = topk; p.woff = lw[il].down;
            [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:hq offset:0 atIndex:1]; [e setBuffer:hs offset:0 atIndex:2];
            [e setBuffer:top offset:0 atIndex:3]; [e setBuffer:wts offset:0 atIndex:4]; [e setBuffer:x offset:0 atIndex:5];
            if (lw[il].dtype == 14) { p.bpr = ff / 256 * 210; DISP(Pdown, d, 32); }
            else { p.bpr = ff / 256 * 144; [e setBuffer:hm offset:0 atIndex:6]; DISP(Pdown4, d, 32); }
        }
    };
    void (^encHead)(id<MTLComputeCommandEncoder>, uint32_t) = ^(id<MTLComputeCommandEncoder> e, uint32_t slot) {
        KP p; memset(&p, 0, sizeof p);
        {
            memset(&p, 0, sizeof p); p.cols = d; p.woff = perLayerF * C.nl; p.aux = epsb;
            [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:xn offset:0 atIndex:1]; [e setBuffer:smalls offset:0 atIndex:2];
            [e setBuffer:xq offset:0 atIndex:3]; [e setBuffer:xs offset:0 atIndex:4]; [e setBuffer:xsum offset:0 atIndex:5];
            DISP(Prmsq, 1, 256);
            memset(&p, 0, sizeof p); p.woff = headT.off; p.rows = C.nvocab; p.cols = d; p.nsb = d / 256;
            [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:xq offset:0 atIndex:1]; [e setBuffer:xs offset:0 atIndex:2]; [e setBuffer:xsum offset:0 atIndex:3];
            [e setBuffer:vlog offset:0 atIndex:4];
            if (headT.type == 14) { p.bpr = d / 256 * 210; DISP(Pmv6x4, (C.nvocab + 3) / 4, 32); }
            else { p.bpr = d / 256 * 144; DISP(Pmv4x4, (C.nvocab + 3) / 4, 32); }
            memset(&p, 0, sizeof p); p.rows = C.nvocab;
            [e setBuffer:vlog offset:0 atIndex:0]; [e setBuffer:pval offset:0 atIndex:1]; [e setBuffer:pidx offset:0 atIndex:2];
            DISP(Pam1, namax, 256);
            memset(&p, 0, sizeof p); p.rows = namax; p.aux = slot;
            [e setBuffer:pval offset:0 atIndex:0]; [e setBuffer:pidx offset:0 atIndex:1]; [e setBuffer:outtok offset:0 atIndex:2];
            DISP(Pam2, 1, 1);
        }
    };

    uint32_t (^runToken)(uint32_t, uint32_t, BOOL) = ^uint32_t(uint32_t tokid, uint32_t pos, BOOL withHead) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encEmbed(e, tokid, 0, NO);
        for (int il = 0; il < C.nl; il++) encLayer(e, il, pos);
        if (withHead) encHead(e, 0);
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { fprintf(stderr, "GPU error: %s\n", cb.error.description.UTF8String); exit(1); }
        return *(uint32_t *)outtok.contents;
    };

    // ---- KTIME: per-kernel-class GPU time, 48 layers each, real weights ----
    if (getenv("CPBRR_KTIME")) {
        double (^timeCB)(void (^)(id<MTLComputeCommandEncoder>)) = ^double(void (^body)(id<MTLComputeCommandEncoder>)) {
            double best = 1e9;
            for (int it = 0; it < 4; it++) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                body(e); [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                double g = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                if (it > 0 && g < best) best = g;
            }
            return best;
        };
        // fake state: quantize zeros (fine for traffic timing), fixed top/wts
        uint32_t *tp3 = (uint32_t *)top.contents; float *wp3 = (float *)wts.contents;
        for (int i = 0; i < topk; i++) { tp3[i] = i * 13; wp3[i] = 0.125f; }
        __block KP p;
        double tqkv = timeCB(^(id<MTLComputeCommandEncoder> e) {
            for (int il = 0; il < C.nl; il++) {
                memset(&p, 0, sizeof p);
                [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:xq offset:0 atIndex:1]; [e setBuffer:xs offset:0 atIndex:2]; [e setBuffer:xsum offset:0 atIndex:3];
                p.woff = lw[il].q; p.rows = nh * hd; p.cols = d; p.bpr = d / 256 * 144; p.nsb = d / 256;
                [e setBuffer:q offset:0 atIndex:4]; DISP(Pmv4, nh * hd, 32);
                p.woff = lw[il].k; p.rows = nkv * hd; [e setBuffer:k offset:0 atIndex:4]; DISP(Pmv4, nkv * hd, 32);
                p.woff = lw[il].v; [e setBuffer:v offset:0 atIndex:4];
                if (lw[il].vtype == 14) { p.bpr = d / 256 * 210; DISP(Pmv6, nkv * hd, 32); } else { DISP(Pmv4, nkv * hd, 32); }
                p.woff = lw[il].o; p.rows = d; p.cols = nh * hd; p.bpr = nh * hd / 256 * 144; p.nsb = nh * hd / 256; p.add = 1;
                [e setBuffer:aq offset:0 atIndex:1]; [e setBuffer:asc offset:0 atIndex:2]; [e setBuffer:asm_ offset:0 atIndex:3];
                [e setBuffer:x offset:0 atIndex:4]; DISP(Pmv4, d, 32);
            }
        });
        double tglu = timeCB(^(id<MTLComputeCommandEncoder> e) {
            for (int il = 0; il < C.nl; il++) {
                memset(&p, 0, sizeof p); p.rows = topk * ff; p.aux = ff; p.bpr = d / 256 * 144; p.nsb = d / 256;
                p.woff = lw[il].gate; p.soff = (uint32_t)(lw[il].up & 0xFFFFFFFFu); p.moff = (uint32_t)(lw[il].up >> 32);
                [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:xq offset:0 atIndex:1]; [e setBuffer:xs offset:0 atIndex:2]; [e setBuffer:xsum offset:0 atIndex:3];
                [e setBuffer:top offset:0 atIndex:4]; [e setBuffer:hbuf offset:0 atIndex:5];
                DISP(Pglu4, (topk * ff + 3) / 4, 32);
            }
        });
        double tdown = timeCB(^(id<MTLComputeCommandEncoder> e) {
            for (int il = 0; il < C.nl; il++) {
                memset(&p, 0, sizeof p); p.rows = d; p.cols = ff; p.aux = topk; p.woff = lw[il].down;
                [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:hq offset:0 atIndex:1]; [e setBuffer:hs offset:0 atIndex:2];
                [e setBuffer:top offset:0 atIndex:3]; [e setBuffer:wts offset:0 atIndex:4]; [e setBuffer:x offset:0 atIndex:5];
                if (lw[il].dtype == 14) { p.bpr = ff / 256 * 210; DISP(Pdown, d, 32); }
                else { p.bpr = ff / 256 * 144; [e setBuffer:hm offset:0 atIndex:6]; DISP(Pdown4, d, 32); }
            }
        });
        double thead = timeCB(^(id<MTLComputeCommandEncoder> e) {
            memset(&p, 0, sizeof p); p.woff = headT.off; p.rows = C.nvocab; p.cols = d; p.nsb = d / 256;
            [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:xq offset:0 atIndex:1]; [e setBuffer:xs offset:0 atIndex:2]; [e setBuffer:xsum offset:0 atIndex:3];
            [e setBuffer:vlog offset:0 atIndex:4];
            if (headT.type == 14) { p.bpr = d / 256 * 210; DISP(Pmv6x4, (C.nvocab + 3) / 4, 32); }
            else { p.bpr = d / 256 * 144; DISP(Pmv4x4, (C.nvocab + 3) / 4, 32); }
        });
        double tattn = timeCB(^(id<MTLComputeCommandEncoder> e) {
            for (int il = 0; il < C.nl; il++) {
                memset(&p, 0, sizeof p); p.rows = nh; p.cols = hd; p.aux = nkv; p.pos = 100;
                [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:kcb offset:(uint64_t)il * MAXSEQ * nkv * hd * 4 atIndex:1];
                [e setBuffer:vcb offset:(uint64_t)il * MAXSEQ * nkv * hd * 4 atIndex:2]; [e setBuffer:ao offset:0 atIndex:3];
                DISP(Pattn, nh, 128);
            }
        });
        double tsmall = timeCB(^(id<MTLComputeCommandEncoder> e) {
            for (int il = 0; il < C.nl; il++) {
                memset(&p, 0, sizeof p); p.cols = d; p.woff = perLayerF * il; p.aux = epsb;
                [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:xn offset:0 atIndex:1]; [e setBuffer:smalls offset:0 atIndex:2];
                DISP(Prms, 1, 256); DISP(Prms, 1, 256);
                memset(&p, 0, sizeof p);
                [e setBuffer:xn offset:0 atIndex:0]; [e setBuffer:xq offset:0 atIndex:1]; [e setBuffer:xs offset:0 atIndex:2]; [e setBuffer:xsum offset:0 atIndex:3];
                DISP(Pqb, d / 256, 32); DISP(Pqb, d / 256, 32); DISP(Pqb, topk * ff / 256, 32);
                memset(&p, 0, sizeof p); p.rows = C.ne; p.cols = d; p.woff = perLayerF * il + (uint64_t)d * 2 + hd * 2;
                [e setBuffer:xn offset:0 atIndex:0]; [e setBuffer:smalls offset:0 atIndex:1]; [e setBuffer:rlog offset:0 atIndex:2];
                DISP(Prtr, C.ne, 32);
                memset(&p, 0, sizeof p); p.rows = C.ne; p.aux = topk;
                [e setBuffer:rlog offset:0 atIndex:0]; [e setBuffer:top offset:0 atIndex:1]; [e setBuffer:wts offset:0 atIndex:2];
                DISP(Ptopk, 1, 32);
                memset(&p, 0, sizeof p); p.rows = nh; p.cols = hd; p.aux = nkv; p.pos = 100;
                p.woff = perLayerF * il + (uint64_t)d * 2; p.xoff = (uint32_t)(perLayerF * il + (uint64_t)d * 2 + hd);
                p.soff = rbaseb; p.moff = epsb;
                [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:k offset:0 atIndex:1]; [e setBuffer:v offset:0 atIndex:2];
                [e setBuffer:smalls offset:0 atIndex:3]; [e setBuffer:kcb offset:0 atIndex:4]; [e setBuffer:vcb offset:0 atIndex:5];
                DISP(Pqkn, nh + nkv, 32);
            }
        });
        printf("KTIME (48 layers, GPU ms): qkv+o %.2f | glu %.2f | down %.2f | attn %.2f | small %.2f | head %.2f | SUM %.2f\n",
               tqkv, tglu, tdown, tattn, tsmall, thead, tqkv + tglu + tdown + tattn + tsmall + thead);
        printf("traffic est: qkv+o 508MB glu 680MB down 420MB head 255MB -> GB/s: qkv+o %.0f glu %.0f down %.0f head %.0f\n",
               0.508 / (tqkv / 1000), 0.680 / (tglu / 1000), 0.420 / (tdown / 1000), 0.255 / (thead / 1000));
        return 0;
    }

    // ---- verify mode: GPU matvec vs CPU scalar reference on real weights ----
    if (verify) {
        // random xn -> GPU qblock -> readback -> both dots on blk.0 rows
        float *xnp = (float *)xn.contents;
        srand(42);
        for (int i = 0; i < d; i++) xnp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        KP p; memset(&p, 0, sizeof p);
        [e setComputePipelineState:Pqb];
        [e setBuffer:xn offset:0 atIndex:0]; [e setBuffer:xq offset:0 atIndex:1]; [e setBuffer:xs offset:0 atIndex:2]; [e setBuffer:xsum offset:0 atIndex:3];
        [e setBytes:&p length:sizeof p atIndex:29];
        [e dispatchThreadgroups:MTLSizeMake(d / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        // q4k matvec on attn_q first 64 rows
        memset(&p, 0, sizeof p); p.woff = lw[0].q; p.rows = 64; p.cols = d; p.bpr = d / 256 * 144; p.nsb = d / 256;
        [e setComputePipelineState:Pmv4];
        [e setBuffer:blob offset:0 atIndex:0]; [e setBuffer:xq offset:0 atIndex:1]; [e setBuffer:xs offset:0 atIndex:2]; [e setBuffer:xsum offset:0 atIndex:3];
        [e setBuffer:q offset:0 atIndex:4]; [e setBytes:&p length:sizeof p atIndex:29];
        [e dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        // q6k matvec on attn_v first 64 rows
        memset(&p, 0, sizeof p); p.woff = lw[0].v; p.rows = 64; p.cols = d; p.bpr = d / 256 * 210; p.nsb = d / 256;
        [e setComputePipelineState:Pmv6];
        [e setBuffer:v offset:0 atIndex:4]; [e setBytes:&p length:sizeof p atIndex:29];
        [e dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];

        const int8_t *xqp = (const int8_t *)xq.contents;
        const float *xsp = (const float *)xs.contents;
        const int32_t *xmp = (const int32_t *)xsum.contents;
        float e4 = 0, e6 = 0;
        for (int r = 0; r < 64; r++) {
            float ref = ref_q4k_dot(bb + lw[0].q + (uint64_t)r * (d / 256 * 144), xqp, xsp, xmp, d);
            float got = ((float *)q.contents)[r];
            e4 = fmaxf(e4, fabsf(ref - got) / (fabsf(ref) + 1e-6f));
            ref = ref_q6k_dot(bb + lw[0].v + (uint64_t)r * (d / 256 * 210), xqp, xsp, d);
            got = ((float *)v.contents)[r];
            e6 = fmaxf(e6, fabsf(ref - got) / (fabsf(ref) + 1e-6f));
        }
        printf("verify q4k matvec GPU vs CPU-ref: max rel err %.2e  %s\n", e4, e4 < 1e-4 ? "PASS" : "FAIL");
        printf("verify q6k matvec GPU vs CPU-ref: max rel err %.2e  %s\n", e6, e6 < 1e-4 ? "PASS" : "FAIL");
        return (e4 < 1e-4 && e6 < 1e-4) ? 0 : 1;
    }

    // ---- tokenizer ----
    char path[1024]; snprintf(path, sizeof path, "%s/tokens.bin", dir);
    NSData *tb = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    const uint8_t *tp = tb.bytes;
    uint32_t ntok = *(const uint32_t *)tp;
    NSMutableArray *toks = [NSMutableArray arrayWithCapacity:ntok];
    NSMutableDictionary *lookup = [NSMutableDictionary new], *specials = [NSMutableDictionary new];
    {
        size_t o = 4;
        for (uint32_t i = 0; i < ntok; i++) {
            uint16_t l = *(const uint16_t *)(tp + o); o += 2;
            NSData *t = [NSData dataWithBytes:tp + o length:l]; o += l;
            [toks addObject:t];
            if (l >= 2 && tp[o - l] == '<' && tp[o - l + 1] == '|') { specials[t] = @(i); }
            else if (l > 0 && !lookup[t]) { lookup[t] = @(i); }
        }
    }
    NSMutableArray *ids = [NSMutableArray new];
    void (^enc)(const char *) = ^(const char *s) {
        size_t n = strlen(s), i = 0;
        while (i < n) {
            size_t l = (n - i) < 48 ? (n - i) : 48;
            for (;; l--) {
                NSData *key = [NSData dataWithBytes:s + i length:l];
                NSNumber *id_ = lookup[key];
                if (id_) { [ids addObject:id_]; i += l; break; }
                if (l == 1) { fprintf(stderr, "tokenizer: no match\n"); exit(1); }
            }
        }
    };
    uint32_t (^sp)(const char *) = ^uint32_t(const char *s) {
        NSNumber *id_ = specials[[NSData dataWithBytes:s length:strlen(s)]];
        if (!id_) { fprintf(stderr, "missing special %s\n", s); exit(1); }
        return id_.unsignedIntValue;
    };
    uint32_t im_start = sp("<|im_start|>"), im_end = sp("<|im_end|>");

    if (argc > 3 && !strcmp(argv[3], "--debug")) {
        float (^bsum)(id<MTLBuffer>, int) = ^float(id<MTLBuffer> b, int n) {
            const float *f = (const float *)b.contents; float s = 0; for (int i = 0; i < n; i++) s += f[i]; return s;
        };
        void (^step)(void (^)(id<MTLComputeCommandEncoder>)) = ^(void (^body)(id<MTLComputeCommandEncoder>)) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            body(e); [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) { fprintf(stderr, "GPU err: %s\n", cb.error.description.UTF8String); exit(1); }
        };
        uint32_t tok = im_start;
        step(^(id<MTLComputeCommandEncoder> e) { encEmbed(e, tok, 0, NO); });
        const float *xp = (const float *)x.contents;
        printf("DBG embed sum %.6f x[0..4] %f %f %f %f\n", bsum(x, d), xp[0], xp[1], xp[2], xp[3]);
        step(^(id<MTLComputeCommandEncoder> e) { encLayer(e, 0, 0); });
        printf("DBG L0 qkv q %.6f k %.6f v %.6f\n", bsum(q, nh * hd), bsum(k, nkv * hd), bsum(v, nkv * hd));
        printf("DBG L0 ao %.6f x-after-ffn %.6f\n", bsum(ao, nh * hd), bsum(x, d));
        const uint32_t *tp2 = (const uint32_t *)top.contents; const float *wp = (const float *)wts.contents;
        printf("DBG L0 top [%u,%u,%u,%u,%u,%u,%u,%u] wts [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
               tp2[0],tp2[1],tp2[2],tp2[3],tp2[4],tp2[5],tp2[6],tp2[7], wp[0],wp[1],wp[2],wp[3],wp[4],wp[5],wp[6],wp[7]);
        step(^(id<MTLComputeCommandEncoder> e) { encLayer(e, 1, 0); });
        printf("DBG L1 x-after-ffn %.6f\n", bsum(x, d));
        for (int il = 2; il < C.nl; il++) {
            int cil = il;
            step(^(id<MTLComputeCommandEncoder> e) { encLayer(e, cil, 0); });
            float s = bsum(x, d);
            printf("DBG L%d x %.6f ao %.6f\n", il, s, bsum(ao, nh * hd));
            if (isnan(s)) {
                // dump the layer's surviving intermediates to find the NaN birth stage
                const float *rl = (const float *)rlog.contents;
                float rs = 0; int rnan = 0;
                for (int i = 0; i < C.ne; i++) { rs += rl[i]; rnan += isnan(rl[i]); }
                const uint32_t *tpp = (const uint32_t *)top.contents; const float *wp = (const float *)wts.contents;
                const float *hb = (const float *)hbuf.contents;
                int hnan = 0; float hsum = 0;
                for (int i = 0; i < topk * ff; i++) { hnan += isnan(hb[i]); hsum += hb[i]; }
                const float *xnp2 = (const float *)xn.contents; int xnnan = 0;
                for (int i = 0; i < d; i++) xnnan += isnan(xnp2[i]);
                printf("DBG L%d POSTMORTEM xn sum %.6f nan %d | rlog sum %.4f nan %d | top [%u,%u,%u,%u,%u,%u,%u,%u]\n",
                       il, bsum(xn, d), xnnan, rs, rnan, tpp[0],tpp[1],tpp[2],tpp[3],tpp[4],tpp[5],tpp[6],tpp[7]);
                printf("DBG L%d POSTMORTEM wts [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] | hbuf sum %.4f nan %d/%d\n",
                       il, wp[0],wp[1],wp[2],wp[3],wp[4],wp[5],wp[6],wp[7], hsum, hnan, topk * ff);
                // where in x, and are quant buffers sane?
                const float *xp2 = (const float *)x.contents; int xnan = 0, firstnan = -1;
                for (int i = 0; i < d; i++) { if (isnan(xp2[i])) { xnan++; if (firstnan < 0) firstnan = i; } }
                const float *hsp = (const float *)hs.contents; int hsnan = 0;
                for (int i = 0; i < topk * ff / 256; i++) hsnan += (isnan(hsp[i]) || isinf(hsp[i]));
                printf("DBG L%d POSTMORTEM x nan %d/%d first %d | hs bad %d/%d hs[0..3] %g %g %g\n",
                       il, xnan, d, firstnan, hsnan, topk * ff / 256, hsp[0], hsp[1], hsp[2]);
                // CPU recompute of kdown for the first NaN row from GPU's OWN inputs
                if (firstnan >= 0) {
                    const int8_t *hqp = (const int8_t *)hq.contents;
                    float acc = 0;
                    for (int e2 = 0; e2 < topk; e2++) {
                        float v2 = ref_q6k_dot(bb + lw[il].down + (uint64_t)(tpp[e2] * d + firstnan) * (ff / 256 * 210),
                                               hqp + e2 * ff, hsp + e2 * (ff / 256), ff);
                        printf("DBG   e%d row %u cpu-dot %g wt %g\n", e2, tpp[e2] * d + firstnan, v2, wp[e2]);
                        acc += wp[e2] * v2;
                    }
                    printf("DBG   cpu-recompute down[%d] = %g (GPU x[%d] = %g)\n", firstnan, acc, firstnan, xp2[firstnan]);
                    // requantize hbuf on CPU (fresh) and redo the dots -> isolates GPU-quant vs weights
                    int8_t *hq2 = malloc(topk * ff); float *hs2 = malloc(topk * ff / 256 * 4); int32_t *hm2 = malloc(topk * ff / 32 * 4);
                    for (int sb = 0; sb < topk * ff / 256; sb++) {
                        float amax = 1e-12f;
                        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(hb[sb * 256 + i]));
                        float sc = amax / 127.0f; hs2[sb] = sc;
                        for (int j = 0; j < 8; j++) { int32_t a2 = 0;
                            for (int i = 0; i < 32; i++) { int qi = (int)lrintf(fminf(fmaxf(hb[sb*256+j*32+i]/sc, -127.f), 127.f)); hq2[sb*256+j*32+i] = qi; a2 += qi; }
                            hm2[sb * 8 + j] = a2; }
                    }
                    float acc2 = 0;
                    for (int e2 = 0; e2 < topk; e2++) {
                        float v2 = ref_q6k_dot(bb + lw[il].down + (uint64_t)(tpp[e2] * d + firstnan) * (ff / 256 * 210),
                                               hq2 + e2 * ff, hs2 + e2 * (ff / 256), ff);
                        printf("DBG   e%d fresh-quant cpu-dot %g\n", e2, v2);
                        acc2 += wp[e2] * v2;
                    }
                    printf("DBG   fresh-quant recompute down[%d] = %g\n", firstnan, acc2);
                }
                break;
            }
        }
        step(^(id<MTLComputeCommandEncoder> e) { encHead(e, 0); });
        uint32_t tid = *(uint32_t *)outtok.contents;
        const float *vl = (const float *)vlog.contents;
        float vmax = -1e30f, vmin = 1e30f; uint32_t vargmax = 0;
        for (int i = 0; i < C.nvocab; i++) { if (vl[i] > vmax) { vmax = vl[i]; vargmax = i; } if (vl[i] < vmin) vmin = vl[i]; }
        printf("DBG head argmax id %u score %.4f | cpu-scan argmax %u max %.4f min %.4f xn %.6f\n",
               tid, vl[tid], vargmax, vmax, vmin, bsum(xn, d));
        return 0;
    }
    [ids addObject:@(im_start)]; enc("system\nYou are a helpful assistant."); [ids addObject:@(im_end)];
    [ids addObject:@(im_start)]; { char ub[8192]; snprintf(ub, sizeof ub, "\nuser\n%s", prompt); enc(ub); } [ids addObject:@(im_end)];
    [ids addObject:@(im_start)]; enc("\nassistant\n");
    fprintf(stderr, "prompt: %lu tokens\n", (unsigned long)ids.count);

    // ---- prefill (head only on last) ----
    double t1 = now_s();
    uint32_t next = 0;
    for (NSUInteger i = 0; i < ids.count; i++)
        next = runToken([ids[i] unsignedIntValue], (uint32_t)i, i == ids.count - 1);
    fprintf(stderr, "prefill %lu tok in %.2fs (%.1f tok/s)\n", (unsigned long)ids.count,
            now_s() - t1, ids.count / (now_s() - t1));

    // ---- decode: GPU-chained + pipelined ----
    // Each cmdbuf's embed reads the previous argmax from outtok[] ON THE GPU, so all
    // ngen command buffers are encoded+committed ahead; the CPU only trails behind
    // reading finished slots. No CPU->GPU round-trip on the critical path.
    // (On early EOS the already-queued tail buffers run harmlessly on stale tokens.)
    // NOTE: prefill's head wrote outtok[0] = `next` above.
    double t2 = now_s();
    uint32_t pos = (uint32_t)ids.count;
    NSMutableArray *cbs = [NSMutableArray arrayWithCapacity:ngen];
    for (int g = 0; g < ngen; g++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encEmbed(e, 0, (uint32_t)(g + 1), YES);      // reads outtok[g]
        for (int il = 0; il < C.nl; il++) encLayer(e, il, pos + g);
        encHead(e, (uint32_t)(g + 1));               // writes outtok[g+1]
        [e endEncoding];
        [cb commit];
        [cbs addObject:cb];
    }
    const uint32_t *slots = (const uint32_t *)outtok.contents;
    int n = 0;
    for (int g = 0; g < ngen; g++) {
        if (next == im_end || next >= ntok) break;
        NSData *tt = toks[next];
        fwrite(tt.bytes, 1, tt.length, stdout);
        id<MTLCommandBuffer> cb = cbs[g];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { fprintf(stderr, "GPU error: %s\n", cb.error.description.UTF8String); exit(1); }
        next = slots[g + 1];
        n++;
    }
    double dt = now_s() - t2;
    printf("\n---\ndecode: %d tokens in %.2fs = %.1f tok/s\n", n, dt, n / dt);
    return 0;
} }
