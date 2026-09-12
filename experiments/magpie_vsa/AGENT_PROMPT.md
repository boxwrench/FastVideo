# Blind Magpie VSA Rediscovery Task

You are evaluating whether AMD-AGI/Magpie can help discover a real FastVideo
optimization on an AMD Radeon AI PRO R9700 (gfx1201).

## Integrity rule

Until you have written `experiments/magpie_vsa/hypothesis.md`, DO NOT inspect:

- GitHub PR `hao-ai-lab/FastVideo#1813`
- branch `perf/vsa-block-resolution-combine`
- commits descended from that branch
- any file or git history that reveals the known solution

Work only from `main` plus this experiment directory.

## Goal

Determine why the **gated VSA coarse+sparse combine** consumes avoidable time and
VRAM at an H3-like shape and propose the smallest safe source change.

The baseline source under test is:

- `experiments/magpie_vsa/vsa_combine_baseline.py`
- production context:
  `fastvideo-kernel/python/fastvideo_kernel/ops.py`

Use Magpie as the evaluator. Prefer its MCP tools if available; otherwise use the
CLI/skill.

## Procedure

1. Record environment:
   - git commit
   - GPU / gfx arch
   - ROCm
   - PyTorch
   - Magpie commit/version
   - rocprof-compute version

2. Run:
   `bash experiments/magpie_vsa/run_baseline.sh`

3. Inspect Magpie's `analyze_report.json`, kernel summary, raw profiler output,
   and workload directory. If gfx1201 profiling is unsupported, say exactly
   where it fails; do not silently substitute another GPU.

4. Inspect the two source files above and explain the allocation/data-movement
   mechanism responsible for the baseline behavior.

5. Write `experiments/magpie_vsa/hypothesis.md` containing:
   - bottleneck mechanism
   - expected number/size of avoidable full-sequence temporaries
   - proposed code transformation
   - correctness risks (including autograd / mutation)
   - expected impact on latency and peak allocation
   - what would falsify the hypothesis

6. Only after the hypothesis file exists, implement your candidate on a NEW
   branch. Add correctness tests before accepting performance results.

7. Benchmark the candidate with the exact same H3 shape. Final acceptance
   requires:
   - correctness passes
   - lower peak allocation
   - lower median combine latency
   - no changed workload shape/dtype
   - no claim of end-to-end H3 speedup until a real H3 run verifies it

## Important

This is an evaluator test as much as a kernel test. Separate what **Magpie
measured** from what **you inferred from source**. If Magpie cannot profile
gfx1201 cleanly, that is a finding, not a reason to hide the failure.
