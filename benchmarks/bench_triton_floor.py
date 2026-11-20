"""What plain Triton reaches on this GPU for w2v-BERT 2.0's attention shape (B=16, H=16, D=64, bf16, no bias): a
minimal FlashAttention forward with pointer loads and with TMA tensor descriptors, against PyTorch's flash SDPA and
this package's relative-key kernel. Separates the cost of the relative bias from what Triton's code generation
allows."""

import argparse
import json

import torch
import triton
import triton.language as tl

from w2vbert_fast.triton_attention import fused_relative_key_attention


def _configs():
    return [triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_stages=s, num_warps=w)
            for m, n, s, w in ((128, 64, 3, 4), (128, 64, 4, 4), (128, 64, 3, 8), (128, 128, 3, 4), (128, 128, 3, 8),
                               (128, 128, 4, 8), (128, 128, 2, 8), (64, 64, 3, 4))]


@triton.autotune(configs=_configs(), key=["T", "HEAD_DIM"])
@triton.jit
def _fa_ptr(Q, K, V, O, T, scale, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = pid_bh * T * HEAD_DIM
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=(offs_m < T)[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = scale * 1.4426950408889634
    for start_n in range(0, (T // BLOCK_N) * BLOCK_N, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(K + base + cols[:, None] * HEAD_DIM + offs_d[None, :])
        v = tl.load(V + base + cols[:, None] * HEAD_DIM + offs_d[None, :])
        s = tl.dot(q, tl.trans(k))
        m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m_i = m_new
    tl.store(O + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], (acc / l_i[:, None]).to(O.dtype.element_ty),
             mask=(offs_m < T)[:, None])


@triton.autotune(configs=_configs(), key=["T", "HEAD_DIM"])
@triton.jit
def _fa_tma(Q, K, V, O, BH, T, scale, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    desc_q = tl.make_tensor_descriptor(Q, shape=[BH, T, HEAD_DIM], strides=[T * HEAD_DIM, HEAD_DIM, 1],
                                       block_shape=[1, BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(K, shape=[BH, T, HEAD_DIM], strides=[T * HEAD_DIM, HEAD_DIM, 1],
                                       block_shape=[1, BLOCK_N, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(V, shape=[BH, T, HEAD_DIM], strides=[T * HEAD_DIM, HEAD_DIM, 1],
                                       block_shape=[1, BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(O, shape=[BH, T, HEAD_DIM], strides=[T * HEAD_DIM, HEAD_DIM, 1],
                                       block_shape=[1, BLOCK_M, HEAD_DIM])
    q = desc_q.load([pid_bh, pid_m * BLOCK_M, 0]).reshape(BLOCK_M, HEAD_DIM)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = scale * 1.4426950408889634
    for start_n in range(0, (T // BLOCK_N) * BLOCK_N, BLOCK_N):
        k = desc_k.load([pid_bh, start_n, 0]).reshape(BLOCK_N, HEAD_DIM)
        v = desc_v.load([pid_bh, start_n, 0]).reshape(BLOCK_N, HEAD_DIM)
        s = tl.dot(q, tl.trans(k))
        m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m_i = m_new
    out = (acc / l_i[:, None]).to(O.dtype.element_ty)
    desc_o.store([pid_bh, pid_m * BLOCK_M, 0], out.reshape(1, BLOCK_M, HEAD_DIM))


def timed(fn, repeat=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def sweep(frames):
    """Each configuration of the relative-key forward kernel on its own (the packed layout the model uses)."""
    from w2vbert_fast import triton_attention as ta

    batch, heads, dim = 16, 16, 64
    qkv = torch.randn(batch, frames, 3, heads, dim, device="cuda", dtype=torch.bfloat16)
    e = torch.randn(73, dim, device="cuda", dtype=torch.bfloat16)
    flops = 4 * batch * heads * frames * frames * dim
    configs = list(ta._forward.configs)
    for config in configs:
        ta._forward.configs = [config]
        ta._forward.cache.clear()
        try:
            with torch.no_grad():
                ms = timed(lambda: ta.fused_relative_key_attention_packed(qkv, e, 64, 8))
            row = {"frames": frames, "forward_config": str(config), "ms": round(ms, 3),
                   "TFLOPS": round(flops / ms / 1e9)}
        except Exception as error:
            row = {"frames": frames, "forward_config": str(config), "error": f"{type(error).__name__}"}
        print(json.dumps(row), flush=True)
    ta._forward.configs = configs
    ta._forward.cache.clear()


def probe(frames, workdir):
    """Where the relative-key forward spends its time: copies of the kernel module with parts removed (wrong
    outputs; timing only), each configuration timed on its own."""
    import importlib.util
    import pathlib

    import w2vbert_fast.triton_attention as ta

    source = pathlib.Path(ta.__file__).read_text()
    band = """    for start_n in range(a, b2, BAND_N):"""
    tail = """    for start_n in range(full, kv_len, BAND_N):"""
    variants = {
        "full_kernel": source,
        "no_band": source.replace(band, "    for start_n in range(a, a, BAND_N):"),
        "no_tail": source.replace(tail, "    for start_n in range(full, full, BAND_N):"),
        "far_only": source.replace(band, "    for start_n in range(a, a, BAND_N):").replace(
            tail, "    for start_n in range(full, full, BAND_N):"),
    }
    batch, heads, dim = 16, 16, 64
    qkv = torch.randn(batch, frames, 3, heads, dim, device="cuda", dtype=torch.bfloat16)
    e = torch.randn(73, dim, device="cuda", dtype=torch.bfloat16)
    for index, (name, text) in enumerate(variants.items()):
        assert name == "full_kernel" or text != source, name
        path = pathlib.Path(workdir) / f"probe_{index}.py"
        path.write_text(text.replace("w2vbert_fast::", f"w2vbert_probe{index}::").replace(
            "from .attention", "from w2vbert_fast.attention"))
        spec = importlib.util.spec_from_file_location(f"probe_{index}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for config in list(module._forward.configs):
            module._forward.configs = [config]
            module._forward.cache.clear()
            row = {"variant": name, "config": str(config).split(", num_ctas")[0]}
            try:
                with torch.no_grad():
                    row["ms"] = round(timed(lambda: module.fused_relative_key_attention_packed(qkv, e, 64, 8)), 3)
            except Exception as error:  # e.g. a tile that needs more shared memory than the GPU has
                row["error"] = f"{type(error).__name__}: {str(error)[:120]}"
            print(json.dumps(row), flush=True)


def report_registers():
    """Registers, spills and shared memory of every compiled variant of the three forward kernels."""
    from w2vbert_fast import triton_attention as ta

    for kernel_fn, name in ((ta._forward.fn, "_forward"), (_fa_ptr.fn, "_fa_ptr"), (_fa_tma.fn, "_fa_tma")):
        for device_cache in kernel_fn.device_caches.values():
            for key, compiled in device_cache[0].items():
                meta = compiled.metadata
                print(json.dumps({"kernel": name, "regs": compiled.n_regs, "spills": compiled.n_spills,
                                  "warps": meta.num_warps, "stages": meta.num_stages, "shared": meta.shared,
                                  "key": str(key)[-160:]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, nargs="+", default=[1536, 1500])
    parser.add_argument("--sweep", action="store_true", help="time each relative-key forward configuration")
    parser.add_argument("--probe", default=None, help="directory for kernel copies with parts removed (timing)")
    args = parser.parse_args()
    if args.probe:
        for frames in args.frames:
            probe(frames, args.probe)
        return
    if args.sweep:
        for frames in args.frames:
            sweep(frames)
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    batch, heads, dim = 16, 16, 64
    for frames in args.frames:
        q, k, v = (torch.randn(batch, heads, frames, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        e = torch.randn(73, dim, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(q)
        flops = 4 * batch * heads * frames * frames * dim
        scale = dim ** -0.5

        def grid(meta):
            return (triton.cdiv(frames, meta["BLOCK_M"]), batch * heads)

        runs = {
            "triton_ptr_no_bias": lambda: _fa_ptr[grid](q, k, v, out, frames, scale, HEAD_DIM=dim),
            "triton_tma_no_bias": lambda: _fa_tma[grid](q, k, v, out, batch * heads, frames, scale, HEAD_DIM=dim),
            "sdpa_flash_no_bias": lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            "relative_key_triton": lambda: fused_relative_key_attention(q, k, v, e, 64, 8),
        }
        for name, fn in runs.items():
            try:
                with torch.no_grad():
                    ms = timed(fn)
                row = {"frames": frames, "kernel": name, "ms": round(ms, 3), "TFLOPS": round(flops / ms / 1e9)}
                kernel = {"triton_ptr_no_bias": _fa_ptr, "triton_tma_no_bias": _fa_tma}.get(name)
                if kernel is not None and kernel.best_config is not None:
                    row["config"] = str(kernel.best_config)
            except Exception as error:  # report and go on
                row = {"frames": frames, "kernel": name, "error": f"{type(error).__name__}: {str(error)[:200]}"}
            print(json.dumps(row), flush=True)
    if args.sweep:
        report_registers()


if __name__ == "__main__":
    main()
