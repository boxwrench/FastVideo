#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "== Magpie / VSA blind baseline preflight =="
echo "repo: $ROOT"
echo "commit: $(git rev-parse HEAD)"
echo

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda/rocm available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("hip:", getattr(torch.version, "hip", None))
PY

if command -v rocprof-compute >/dev/null 2>&1; then
  echo "rocprof-compute: $(command -v rocprof-compute)"
  rocprof-compute --version || true
else
  echo "WARNING: rocprof-compute not found; Magpie performance profiling will fail."
fi

if command -v magpie >/dev/null 2>&1; then
  MAGPIE=(magpie)
elif python -c 'import Magpie' >/dev/null 2>&1; then
  MAGPIE=(python -m Magpie)
else
  cat <<'EOF'
Magpie is not installed.

Recommended:
  git clone https://github.com/AMD-AGI/Magpie.git ~/src/Magpie
  python -m pip install -e ~/src/Magpie
EOF
  exit 2
fi

echo
echo "== Magpie GPU info =="
"${MAGPIE[@]}" --gpu-info || true

echo
echo "== Small correctness/perf smoke test =="
python experiments/magpie_vsa/vsa_combine_baseline.py \
  --shape small --gated --warmup 2 --iterations 3

echo
echo "== H3-shape direct baseline =="
python experiments/magpie_vsa/vsa_combine_baseline.py \
  --shape h3 --gated --warmup 3 --iterations 5 | tee /tmp/magpie-vsa-direct-baseline.json

echo
echo "== Magpie correctness/execution smoke test (no profiler) =="
"${MAGPIE[@]}" analyze \
  --kernel-config experiments/magpie_vsa/magpie_baseline.yaml \
  --no-perf \
  --output-dir results/magpie_vsa_smoke

echo
echo "== Magpie profiled analyze =="
set +e
"${MAGPIE[@]}" analyze \
  --kernel-config experiments/magpie_vsa/magpie_baseline.yaml \
  --output-dir results/magpie_vsa_profile
STATUS=$?
set -e

echo
if [[ $STATUS -ne 0 ]]; then
  echo "Magpie profiling exited $STATUS."
  echo "This is still a useful compatibility result on gfx1201; preserve the workspace/logs."
else
  echo "Magpie profile completed."
fi

echo
echo "Results:"
echo "  results/magpie_vsa_smoke"
echo "  results/magpie_vsa_profile"
