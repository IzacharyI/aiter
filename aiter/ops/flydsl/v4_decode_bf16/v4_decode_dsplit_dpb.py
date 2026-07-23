"""M28: BK=64 tile (amortize per-tile barrier/overhead — ATT showed s_barrier 33%
+ s_waitcnt 26% = 59% sync-bound). Halving the tile count halves the barriers.
Base = dpd (dv-split 2-wave + vmcnt ping-pong prefetch + QK interleave + DPP
softmax + QK dedup). BK 32->64: 4 QK subtiles, 4-way softmax, 2 PV K-chunks
(ds_tr row-offset 32*kc). 2 lds_k buffers @ BK=64 = 133KB + lds_p 16KB = 146KB
< 160KB (fits, keeps ping-pong). acc unchanged (16 v4f32, PV output dv-indep of BK).
"""
import functools
import torch
import triton

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Vector as Vec

from aiter.ops.flydsl.kernels.tensor_shim import GTensor, STensor, _run_compiled
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu as mlir_gpu
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf as _scf

from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.flydsl.v4_decode_bf16.reduce_kernel import _paged_decode_reduce_kernel

D = 512
BK = 64
NKC = D // 32          # 16 QK contract-d chunks
NSUB = BK // 16        # 4 QK subtiles (16 kv each)
NKPV = BK // 32        # 2 PV K-chunks (32 kv each)
NW = 8
NOT_HALF = (D // 16) // 2   # 16 dv-tiles/warp (256 dv)
COOP_ITERS = BK * D // (NW * 64 * 8)   # 8
LOG2E = 1.4426950408889634
NEG = -3.4028234663852886e38


@functools.lru_cache(maxsize=None)
def build_dpb(qk_scale: float, KV_SPLITS: int, kpad: int = 8, tail_mask: int = 0):
    _REV = 1

    @flyc.kernel(name=f"v4dec_dpb_ks{KV_SPLITS}", known_block_size=[NW * 64, 1, 1])
    def kern(q: fx.Tensor, kv: fx.Tensor, kv_indices: fx.Tensor,
             kv_indptr: fx.Tensor, m_partial: fx.Tensor, l_partial: fx.Tensor,
             acc_partial: fx.Tensor, H: fx.Int32):
        def IX(x):
            return fx.Index(x)
        fmf = arith.FastMathFlags.fast

        def wait_vmcnt(nn):
            _llvm.InlineAsmOp(None, [], f"s_waitcnt vmcnt({nn})", "", has_side_effects=True)

        def fmul(a, b): return arith.mulf(_raw(a), _raw(b), fastmath=fmf)
        def fadd(a, b): return arith.addf(_raw(a), _raw(b), fastmath=fmf)
        def fsub(a, b): return arith.subf(_raw(a), _raw(b), fastmath=fmf)

        def shfl_x(v, off):
            vi = arith.bitcast(T.i32, _raw(v))
            if const_expr(off == 1):
                ri = rocdl.update_dpp(T.i32, _raw(vi), _raw(vi), 0xB1, 0xF, 0xF, True)
            elif const_expr(off == 2):
                ri = rocdl.update_dpp(T.i32, _raw(vi), _raw(vi), 0x4E, 0xF, 0xF, True)
            else:
                enc = (off << 10) | 0x1F
                ri = rocdl.ds_swizzle(T.i32, _raw(vi), _raw(arith.constant(enc, type=T.i32)))
            return arith.bitcast(T.f32, _raw(ri))

        def gmax(v):
            for off in (8, 4, 2, 1):
                v = arith.maximumf(_raw(v), _raw(shfl_x(v, off)))
            return v

        def gsum(v):
            for off in (8, 4, 2, 1):
                v = fadd(v, shfl_x(v, off))
            return v

        t = fx.block_idx.x
        by = fx.block_idx.y
        pid_k = fx.block_idx.z
        tid = fx.thread_idx.x
        wid = tid // fx.Int32(64)
        ht = wid % fx.Int32(4)       # warp-binding B: spread dh0(QK) across all 4 SIMDs
        dh = wid // fx.Int32(4)      # (M31: +5-8%; dedup QK phase uses 4 SIMDs not 2)
        lane = tid % fx.Int32(64)
        m = lane % fx.Int32(16)
        kg = lane // fx.Int32(16)
        n = lane % fx.Int32(16)

        KD = D + kpad
        BKD = BK * KD
        Hc = H
        KS = fx.Int32(KV_SPLITS)
        scale_c = arith.constant(qk_scale, type=T.f32)
        neg_c = arith.constant(NEG, type=T.f32)
        f32_0 = arith.constant(0.0, type=T.f32)
        v4f32 = T.vec(4, T.f32)

        q_ = GTensor(q, dtype=T.bf16, shape=(-1,))
        kv_ = GTensor(kv, dtype=T.bf16, shape=(-1,))
        idx_ = GTensor(kv_indices, dtype=T.i32, shape=(-1,))
        indptr_ = GTensor(kv_indptr, dtype=T.i32, shape=(-1,))
        mp_ = GTensor(m_partial, dtype=T.f32, shape=(-1,))
        lp_ = GTensor(l_partial, dtype=T.f32, shape=(-1,))
        ap_ = GTensor(acc_partial, dtype=T.f32, shape=(-1,))

        arch = get_rocm_arch()
        alloc = SmemAllocator(None, arch=arch, global_sym_name=f"v4dec_dpb_smem_v{_REV}")
        o_k = alloc._align(alloc.ptr, 16); alloc.ptr = o_k + 2 * BKD * 2
        o_p = alloc._align(alloc.ptr, 16); alloc.ptr = o_p + 4 * 16 * BK * 4
        o_a = alloc._align(alloc.ptr, 16); alloc.ptr = o_a + 4 * 16 * 4
        base = alloc.get_base()
        lds_k = STensor(SmemPtr(base, o_k, T.bf16, shape=(2 * BKD,)), dtype=T.bf16, shape=(2 * BKD,))
        lds_p = STensor(SmemPtr(base, o_p, T.f32, shape=(4 * 16 * BK,)), dtype=T.f32, shape=(4 * 16 * BK,))
        lds_a = STensor(SmemPtr(base, o_a, T.f32, shape=(4 * 16,)), dtype=T.f32, shape=(4 * 16,))

        def ds_tr(cur_buf, k_row, v_col):
            elem = cur_buf * fx.Int32(BKD) + k_row * fx.Int32(KD) + v_col
            byte = fx.Int32(o_k) + elem * fx.Int32(2)
            bi = arith.index_cast(T.i64, arith.index_cast(T.index, byte))
            ptr = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), bi).result
            return rocdl.ds_read_tr16_b64(T.vec(4, T.bf16), ptr).result

        pb = ht * fx.Int32(16 * BK)
        ab = ht * fx.Int32(16)
        dh0 = arith.cmpi(arith.CmpIPredicate.eq, _raw(dh), _raw(fx.Int32(0)))

        kv_start = indptr_[t]
        kv_end = indptr_[t + fx.Int32(1)]
        kv_len = kv_end - kv_start
        num_tiles = (kv_len + fx.Int32(BK - 1)) // fx.Int32(BK)
        tps = (kv_len + KS * fx.Int32(BK) - fx.Int32(1)) // (KS * fx.Int32(BK))
        tile_start = pid_k * tps
        tile_end = arith.minsi(_raw((pid_k + fx.Int32(1)) * tps), _raw(num_tiles))
        kvb_start = IX(tile_start * fx.Int32(BK))
        kvb_end = IX(tile_end * fx.Int32(BK))

        h_a = by * fx.Int32(64) + ht * fx.Int32(16) + m
        q_base = (t * Hc + h_a) * fx.Int32(D)
        a_frags = [q_.vec_load((q_base + fx.Int32(c * 32) + kg * fx.Int32(8),), 8)
                   for c in range_constexpr(NKC)]

        tr_kg = (lane % fx.Int32(16)) // fx.Int32(4)
        tr_cs = lane % fx.Int32(4)
        h_k_row = kg * fx.Int32(8) + tr_kg

        def coop_load(kvb_i):
            out = []
            for i in range_constexpr(COOP_ITERS):
                kvrow = wid + fx.Int32(i * (NW * 64 // (D // 8)))   # wave-uniform -> scalar idx load
                dv8 = lane * fx.Int32(8)
                _ip = arith.minsi(_raw(kv_start + kvb_i + kvrow), _raw(kv_end - fx.Int32(1)))
                slot = idx_[_ip]
                out.append(kv_.vec_load((slot * fx.Int32(D) + dv8,), 8))
            return out

        def coop_store(pf, cur_buf):
            for i in range_constexpr(COOP_ITERS):
                kvrow = wid + fx.Int32(i * (NW * 64 // (D // 8)))   # wave-uniform -> scalar idx load
                dv8 = lane * fx.Int32(8)
                lds_k.vec_store((IX(cur_buf * fx.Int32(BKD) + kvrow * fx.Int32(KD) + dv8),), pf[i], 8)

        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        pf0 = coop_load(tile_start * fx.Int32(BK))
        init_args = [_raw(neg_c) for _ in range_constexpr(4)] + \
                    [_raw(f32_0) for _ in range_constexpr(4)] + \
                    [_raw(c_zero_v4) for _ in range_constexpr(NOT_HALF)] + \
                    [_raw(v) for v in pf0]

        loop_results = init_args
        for kvb, iters in range(kvb_start, kvb_end, BK, init=init_args):
            m_i = [iters[i] for i in range_constexpr(4)]
            l_i = [iters[4 + i] for i in range_constexpr(4)]
            acc_o = [iters[8 + ot] for ot in range_constexpr(NOT_HALF)]
            pf = [iters[8 + NOT_HALF + i] for i in range_constexpr(COOP_ITERS)]
            kvb_i = fx.Int32(kvb)
            cur_buf = (kvb_i // fx.Int32(BK)) % fx.Int32(2)

            wait_vmcnt(0)
            rocdl.sched_barrier(0)
            coop_store(pf, cur_buf)
            gpu.barrier()
            pf_next = coop_load(kvb_i + fx.Int32(BK))
            rocdl.sched_barrier(0)

            # (2+3) QK (4 subtiles, interleaved) + softmax (4-way) : dh0 only
            def _then():
                def _qkrd(kvrow, c):
                    return lds_k.vec_load((IX(cur_buf * fx.Int32(BKD) + kvrow * fx.Int32(KD) + fx.Int32(c * 32) + kg * fx.Int32(8)),), 8)
                b = [_qkrd(fx.Int32(s * 16) + n, 0) for s in range_constexpr(NSUB)]
                acc = [vector.from_elements(v4f32, [f32_0, f32_0, f32_0, f32_0]) for _ in range_constexpr(NSUB)]
                for c in range_constexpr(NKC):
                    if const_expr(c + 1 < NKC):
                        bn = [_qkrd(fx.Int32(s * 16) + n, c + 1) for s in range_constexpr(NSUB)]
                    rocdl.sched_barrier(0)
                    for s in range_constexpr(NSUB):
                        acc[s] = rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a_frags[c], b[s], acc[s], 0, 0, 0])
                    if const_expr(c + 1 < NKC):
                        b = bn
                s_sub = acc  # NSUB subtiles

                if const_expr(tail_mask):
                    vm = [arith.cmpi(arith.CmpIPredicate.slt, _raw(kvb_i + fx.Int32(s * 16) + n), _raw(kv_len)) for s in range_constexpr(NSUB)]
                nm = [None] * 4
                nl = [None] * 4
                for i in range_constexpr(4):
                    sj = [fmul(vector.extract(s_sub[s], static_position=[i], dynamic_position=[]), scale_c) for s in range_constexpr(NSUB)]
                    if const_expr(tail_mask):
                        sj = [arith.select(vm[s], _raw(sj[s]), _raw(neg_c)) for s in range_constexpr(NSUB)]
                    lmax = sj[0]
                    for s in range_constexpr(NSUB - 1):
                        lmax = arith.maximumf(_raw(lmax), _raw(sj[s + 1]))
                    rowmax = gmax(lmax)
                    m_new = arith.maximumf(_raw(m_i[i]), _raw(rowmax))
                    al = rocdl.exp2(T.f32, fsub(m_i[i], m_new))
                    pj = [rocdl.exp2(T.f32, fsub(sj[s], m_new)) for s in range_constexpr(NSUB)]
                    psum = pj[0]
                    for s in range_constexpr(NSUB - 1):
                        psum = fadd(psum, pj[s + 1])
                    ts = gsum(psum)
                    nl[i] = fadd(fmul(l_i[i], al), ts)
                    nm[i] = m_new
                    hh = kg * fx.Int32(4) + fx.Int32(i)
                    for s in range_constexpr(NSUB):
                        lds_p[IX(pb + hh * fx.Int32(BK) + fx.Int32(s * 16) + n)] = pj[s]
                    lds_a[IX(ab + hh)] = al
                return [nm[0], nm[1], nm[2], nm[3], nl[0], nl[1], nl[2], nl[3]]

            def _else():
                return [m_i[0], m_i[1], m_i[2], m_i[3], l_i[0], l_i[1], l_i[2], l_i[3]]

            _if = _scf.IfOp(_raw(dh0), [T.f32] * 8, has_else=True)
            with ir.InsertionPoint(_if.then_block):
                _scf.YieldOp([_raw(v) for v in _then()])
            with ir.InsertionPoint(_if.else_block):
                _scf.YieldOp([_raw(v) for v in _else()])
            res = list(_if.results)
            m_i = [res[i] for i in range_constexpr(4)]
            l_i = [res[4 + i] for i in range_constexpr(4)]
            gpu.barrier()

            # (4) PV : both warps, 2 K-chunks (contract 64 kv), dv-half via ds_tr
            a_frag = []
            for kc in range_constexpr(NKPV):
                a_raw = lds_p.vec_load((IX(pb + m * fx.Int32(BK) + fx.Int32(kc * 32) + kg * fx.Int32(8)),), 8)
                a_frag.append(Vec(a_raw, (8,), fx.Float32).to(fx.BFloat16).ir_value())
            al4 = [lds_a[IX(ab + kg * fx.Int32(4) + fx.Int32(i))] for i in range_constexpr(4)]
            alpha4 = vector.from_elements(v4f32, [_raw(al4[0]), _raw(al4[1]), _raw(al4[2]), _raw(al4[3])])

            def _pvrd(gdt, kc):
                hv = gdt * fx.Int32(16) + tr_cs * fx.Int32(4)
                r = h_k_row + fx.Int32(kc * 32)
                return (ds_tr(cur_buf, r, hv), ds_tr(cur_buf, r + fx.Int32(4), hv))
            gdt0 = dh * fx.Int32(NOT_HALF)
            # PV pipelined: prefetch next (ot,kc) b_frag before current MFMA (ping-pong bf/bf2)
            pv_pairs = [(ot, kc) for ot in range(NOT_HALF) for kc in range(NKPV)]
            NPV = len(pv_pairs)
            def _pv_read(pi):
                ot_, kc_ = pv_pairs[pi]
                lo, hi = _pvrd(gdt0 + fx.Int32(ot_), kc_)
                return vector.shuffle(lo, hi, [0, 1, 2, 3, 4, 5, 6, 7])
            rocdl.sched_barrier(0)
            bf = _pv_read(0)
            resc_cur = None
            for pi in range_constexpr(NPV):
                ot, kc = pv_pairs[pi]
                if const_expr(kc == 0):
                    resc_cur = fmul(acc_o[ot], alpha4)
                if const_expr(pi + 1 < NPV):
                    bf2 = _pv_read(pi + 1)
                cin = resc_cur if const_expr(kc == 0) else acc_o[ot]
                acc_o[ot] = rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a_frag[kc], bf, cin, 0, 0, 0])
                if const_expr(pi + 1 < NPV):
                    bf = bf2

            loop_results = yield [m_i[0], m_i[1], m_i[2], m_i[3],
                                  l_i[0], l_i[1], l_i[2], l_i[3]] + acc_o + \
                                 [_raw(v) for v in pf_next]

        m_i = [loop_results[i] for i in range_constexpr(4)]
        l_i = [loop_results[4 + i] for i in range_constexpr(4)]
        acc_o = [loop_results[8 + ot] for ot in range_constexpr(NOT_HALF)]

        _ifw = _scf.IfOp(_raw(dh0), [], has_else=False)
        with ir.InsertionPoint(_ifw.then_block):
            for i in range_constexpr(4):
                hh = by * fx.Int32(64) + ht * fx.Int32(16) + kg * fx.Int32(4) + fx.Int32(i)
                mp_[(t * KS + pid_k) * Hc + hh] = m_i[i]
                lp_[(t * KS + pid_k) * Hc + hh] = l_i[i]
            _scf.YieldOp([])

        ap_base = (t * KS + pid_k) * Hc * fx.Int32(D)
        for ot in range_constexpr(NOT_HALF):
            gdt = dh * fx.Int32(NOT_HALF) + fx.Int32(ot)
            for i in range_constexpr(4):
                hh = by * fx.Int32(64) + ht * fx.Int32(16) + kg * fx.Int32(4) + fx.Int32(i)
                dv = gdt * fx.Int32(16) + n
                ci = vector.extract(acc_o[ot], static_position=[i], dynamic_position=[])
                ap_[ap_base + hh * fx.Int32(D) + dv] = ci

        alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalize()

    @flyc.jit
    def launch(q, kv, kv_indices, kv_indptr, mp, lp, ap, H: fx.Int32,
               gx: fx.Int32, gy: fx.Int32, gz: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        kern(q, kv, kv_indices, kv_indptr, mp, lp, ap, H).launch(
            grid=(gx, gy, gz), block=(NW * 64, 1, 1), stream=stream)
    return launch


def _splits(T, H, kv_len=None):
    """Pick KV_SPLITS so the grid fills the GPU in ONE wave. dpb runs
    1 workgroup/CU (AGPR-bound), so the optimum is
        blocks = T * ceil(H/64) * KS  ~=  num_cu
    NOT the sglang heuristic's 2*num_cu target (that assumes 2 wg/CU, which
    dpb does not achieve). Round DOWN to a power of two (rounding up
    over-splits); cap by available tiles so short-kv splits stay fed
    (>=1 tile/split). Measured optimum across T=1..8, kv=50-60K (M34/M34b)."""
    ncu = get_cu_num()
    base = max(1, T * ((H + 63) // 64))        # grid.x * grid.y (pre-split)
    ks = ncu / base
    if kv_len is not None:
        ks = min(ks, (kv_len + BK - 1) // BK)  # keep each split >= 1 tile
    ks = max(1, int(ks))
    p = 1
    while p * 2 <= ks:                          # prev_pow2
        p *= 2
    return p


def flydsl_dpb_full(d, kv_splits=None):
    Tn, H = d["T"], d["h"]
    KS = kv_splits or _splits(Tn, H, d["kv_len"])
    tm = int(d["kv_len"] % 64 != 0)
    exe = build_dpb(float(d["sm"]) * LOG2E, KS, 8, tm)
    q, ukv, idx, indptr = d["q"], d["unified_kv"], d["kv_indices"], d["kv_indptr"]
    mp = torch.empty((Tn, KS, H), dtype=torch.float32, device="cuda")
    lp = torch.empty_like(mp)
    ap = torch.empty((Tn, KS, H, D), dtype=torch.float32, device="cuda")
    _run_compiled(exe, q, ukv, idx, indptr, mp, lp, ap, H,
                  Tn, H // 64, KS, torch.cuda.current_stream())
    out = torch.empty(Tn, H, D, device="cuda", dtype=torch.bfloat16)
    block_d = triton.next_power_of_2(D)
    base_grid = Tn * H
    target = 2 * get_cu_num()
    if base_grid >= target:
        d_chunk = block_d
    else:
        dcn = max(1, target // base_grid); dcn = min(dcn, block_d // 32)
        d_chunk = max(32, triton.next_power_of_2(block_d // dcn))
    grid_reduce = (Tn, H, (D + d_chunk - 1) // d_chunk)
    _paged_decode_reduce_kernel[grid_reduce](
        mp, lp, ap, d["attn_sink"], indptr, out,
        mp.stride(0), mp.stride(1), mp.stride(2),
        lp.stride(0), lp.stride(1), lp.stride(2),
        ap.stride(0), ap.stride(1), ap.stride(2), ap.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        LOG2E, H, D, KS, BLOCK_D=block_d, D_CHUNK=d_chunk, BLOCK_K=BK, num_warps=4)
    return out
