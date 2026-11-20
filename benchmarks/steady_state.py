"""Steady-state timeline for Nsight Systems: the compiled model, warmed up, then 10 back-to-back forwards (or
training steps) inside cudaProfilerStart/Stop.

    nsys profile --capture-range=cudaProfilerApi -o out python benchmarks/steady_state.py --mode infer
    python benchmarks/steady_state.py --gaps out.sqlite   # device idle time between kernels, per iteration
"""

import argparse
import sqlite3

import torch


def run(mode, frames):
    from transformers import Wav2Vec2BertModel

    from w2vbert_fast import optimize

    model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    model.config.layerdrop = 0.0
    model.config.mask_time_prob = 0.0
    optimize(model, compile=True)
    model = model.to("cuda", torch.bfloat16)
    features = torch.randn(16, frames, 160, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(16, frames, 1024, device="cuda", dtype=torch.bfloat16)

    def step():
        if mode == "infer":
            with torch.no_grad():
                model(features)
        else:
            model.zero_grad(set_to_none=True)
            model(features).last_hidden_state.backward(grad)

    model.train(mode == "train")
    for _ in range(4):
        step()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.cudart().cudaProfilerStart()
    start.record()
    for _ in range(10):
        step()
    end.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"{mode} T={frames}: {start.elapsed_time(end) / 10:.2f} ms per step (events)")


def gaps(path):
    db = sqlite3.connect(path)
    rows = db.execute("SELECT start, end, (SELECT value FROM StringIds WHERE id = demangledName) FROM "
                      "CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    for table in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        try:
            rows += [(s, e, table) for s, e in db.execute(f"SELECT start, end FROM {table}").fetchall()]
        except sqlite3.OperationalError:
            pass
    rows.sort()
    busy = sum(e - s for s, e, _ in rows)
    span = rows[-1][1] - rows[0][0]
    idle = {}
    for (s0, e0, n0), (s1, e1, n1) in zip(rows, rows[1:]):
        if s1 > e0:
            key = f"{str(n0)[:60]} -> {str(n1)[:60]}"
            idle[key] = idle.get(key, 0) + s1 - e0
    print(f"span {span / 1e6:.2f} ms, busy {busy / 1e6:.2f} ms, idle {(span - busy) / 1e6:.2f} ms over 10 steps; "
          f"{len(rows)} kernels/memsets")
    for key, value in sorted(idle.items(), key=lambda kv: -kv[1])[:12]:
        print(f"{value / 1e6:9.3f} ms  {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="infer", choices=["infer", "train"])
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--gaps", default=None)
    args = parser.parse_args()
    if args.gaps:
        gaps(args.gaps)
    else:
        run(args.mode, args.frames)
