# Magpie × FastVideo VSA blind rediscovery

This experiment asks a narrow question:

> Can AMD-AGI/Magpie, paired with an agent, independently identify and improve a
> real FastVideo VSA bottleneck on Radeon AI PRO R9700 / gfx1201?

The branch is deliberately based on FastVideo `main` at
`0bd19a976b2e88ccd0d10b687b238bc1fa28c52a`, before the known optimization.
The known optimized branch/PR must remain unread until `hypothesis.md` is
written.

## Target workload

The isolated target is the gated VSA coarse/sparse combine at an H3-like shape:

- bf16
- B=1
- H=56
- S=15488
- D=128
- block elements=64

A previously measured optimized implementation exists and provides ground truth
for scoring the rediscovery later, but its implementation and numbers are
intentionally hidden during the blind phase.

## Files

- `vsa_combine_baseline.py` — isolated baseline combine plus correctness and
  direct latency/VRAM measurement
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

## Evaluation discipline

The agent must separate observations produced by Magpie from hypotheses inferred
from source. It must write `hypothesis.md` before inspecting the known optimized
branch or PR.

A candidate only passes the isolated gate if correctness holds under the same
shape/dtype while both median combine latency and peak allocation improve.
Do not claim an end-to-end MiniMax H3 speedup until a real H3 run verifies it.

Because Magpie does not currently list gfx1201/R9700 as a verified platform, a
cleanly documented profiler incompatibility is also a useful experiment result.
