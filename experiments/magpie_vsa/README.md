# Magpie × FastVideo VSA blind rediscovery

This experiment asks a narrow question:

> Can AMD-AGI/Magpie, paired with an agent, independently surface the real VSA
> coarse/sparse-combine bottleneck that was later fixed in FastVideo PR #1813?

The branch is deliberately based on FastVideo `main` at
`0bd19a976b2e88ccd0d10b687b238bc1fa28c52a`, before the known optimization.
The optimized branch/PR must remain unread until `hypothesis.md` is written.

## Why this target

It has a known ground truth on Radeon AI PRO R9700 / gfx1201:

- H3-like combine shape: bf16, B=1, H=56, S=15488, D=128, block 64
- the pre-change path expands a block-level coarse tensor to full sequence
- gated combine then performs additional full-size elementwise work
- an already-measured optimized implementation exists, but is intentionally
  hidden during discovery

That makes this a useful test of the evaluator rather than an open-ended tuning
exercise.

## Files

- `vsa_combine_baseline.py` — isolated pre-optimization combine plus
  correctness and direct latency/VRAM measurement
- `magpie_baseline.yaml` — Magpie PyTorch analyze config using
  `rocprof-compute`
- `run_baseline.sh` — preflight, direct baseline, Magpie smoke test, Magpie
  profiled analyze
- `AGENT_PROMPT.md` — blind agent procedure and acceptance criteria

## Install Magpie

From a separate checkout:

```bash
git clone https://github.com/AMD-AGI/Magpie.git ~/src/Magpie
python -m pip install -e ~/src/Magpie
```

For agent use, also install Magpie's skill or run its MCP server per the Magpie
documentation.

## Run

From the FastVideo repo root:

```bash
bash experiments/magpie_vsa/run_baseline.sh
```

Then give the agent `experiments/magpie_vsa/AGENT_PROMPT.md`.

## What success looks like

A strong rediscovery should conclude—without seeing the known patch—that the
combine is dominated by avoidable full-sequence materialization/data movement,
propose keeping the coarse value at block resolution and broadcasting it during
the combine, and treat in-place fusion as conditional on autograd safety.

A weaker but still useful result is that Magpie identifies memory/copy-heavy
dispatches and the agent localizes them to the combine.

A gfx1201 profiler failure is also useful: Magpie currently does not list R9700
as a verified platform, so the experiment doubles as a compatibility test.

## Evaluation discipline

Do not score an optimization from a microbenchmark alone. The combine benchmark
is the first gate. Any accepted source patch still needs a real MiniMax H3
end-to-end run under the same workload before claiming H3 speedup.
