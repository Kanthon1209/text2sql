#!/usr/bin/env bash
# 停掉 0、1 卡上的生成服务，准备测试集完全体，再用两张卡做无训练推理。
set -euo pipefail

OUT=/data/k/runs/test-full-v1
PY=/data/k/experiments/run_test_full.py
mkdir -p "$OUT"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export NO_PROXY='*'
export PYTHONUNBUFFERED=1

python - << 'PY'
import os, signal, time
from pathlib import Path

def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""

def children(pid: int) -> list[int]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            text = (proc / "status").read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("PPid:"):
                if int(line.split()[1]) == pid:
                    found.append(int(proc.name))
                break
    return found

roots = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    cmd = cmdline(int(proc.name))
    if "swift deploy" in cmd or "swift/cli/deploy.py" in cmd or cmd.startswith("VLLM::"):
        roots.append(int(proc.name))
victims = []
seen = set()

def walk(pid: int) -> None:
    if pid in seen:
        return
    seen.add(pid)
    for child in children(pid):
        walk(child)
    victims.append(pid)

for pid in roots:
    walk(pid)
for pid in victims:
    print(f"stop pid={pid} {cmdline(pid)[:180]}", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
time.sleep(8)
for pid in victims:
    try:
        os.kill(pid, 0)
    except OSError:
        continue
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"kill -9 {pid}", flush=True)
    except OSError:
        pass
PY

for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  if [[ "$used" -lt 2000 ]]; then
    break
  fi
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
echo "gpu memory used MiB sum=$used"
if [[ "$used" -ge 2000 ]]; then
  nvidia-smi
  echo "0、1 卡显存没有释放" >&2
  exit 1
fi

router_lines=0
if [[ -f "$OUT/router.jsonl" ]]; then
  router_lines=$(wc -l < "$OUT/router.jsonl")
fi
if [[ "$router_lines" -ge 400 && -f "$OUT/similar_top10.json" && -f "$OUT/knowledge_rules.json" ]]; then
  echo "route 和 retrieve 已完成，跳过"
else
  CUDA_VISIBLE_DEVICES=0 python -u "$PY" route --output "$OUT" >"$OUT/router.log" 2>&1 &
  route_pid=$!
  CUDA_VISIBLE_DEVICES=1 python -u "$PY" retrieve --output "$OUT" >"$OUT/retrieve.log" 2>&1 &
  retrieve_pid=$!
  fail=0
  wait "$route_pid" || fail=1
  wait "$retrieve_pid" || fail=1
  if [[ "$fail" -ne 0 ]]; then
    echo "准备阶段失败，见 $OUT/router.log 和 $OUT/retrieve.log" >&2
    exit 1
  fi
fi

python -u "$PY" assemble --output "$OUT" | tee "$OUT/assemble.log"

start_serve() {
  local gpu="$1"
  local port="$2"
  local log="$OUT/serve-gpu${gpu}.log"
  local pidf="$OUT/serve-gpu${gpu}.pid"
  echo "[serve] gpu=$gpu port=$port"
  CUDA_VISIBLE_DEVICES="$gpu" nohup swift deploy \
    --model /data/models/Qwen3.5-4B \
    --host 127.0.0.1 \
    --port "$port" \
    --served_model_name Qwen3.5-4B \
    --infer_backend vllm \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --enable_thinking false \
    --torch_dtype bfloat16 \
    --max_length 8192 \
    --max_new_tokens 1024 \
    --temperature 0 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_max_model_len 8192 \
    >"$log" 2>&1 &
  echo $! >"$pidf"
}

stop_tree() {
  local pid="$1"
  local child
  for child in $(ps -o pid= --ppid "$pid" 2>/dev/null); do
    stop_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

stop_serves() {
  local gpu pid
  for gpu in 0 1; do
    if [[ -f "$OUT/serve-gpu${gpu}.pid" ]]; then
      pid=$(cat "$OUT/serve-gpu${gpu}.pid")
      stop_tree "$pid"
    fi
  done
}

cleanup() {
  stop_serves
}
trap cleanup EXIT

start_serve 0 8002
start_serve 1 8003
for port in 8002 8003; do
  ready=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[serve] :${port} 就绪"
      ready=1
      break
    fi
    sleep 3
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[serve] :${port} 未就绪" >&2
    exit 1
  fi
done

python -u "$PY" infer --output "$OUT" --shard 0 --shards 2 --base-url http://127.0.0.1:8002/v1 >"$OUT/infer-gpu0.log" 2>&1 &
infer0=$!
python -u "$PY" infer --output "$OUT" --shard 1 --shards 2 --base-url http://127.0.0.1:8003/v1 >"$OUT/infer-gpu1.log" 2>&1 &
infer1=$!
fail=0
wait "$infer0" || fail=1
wait "$infer1" || fail=1
if [[ "$fail" -ne 0 ]]; then
  echo "推理失败，见 $OUT/infer-gpu0.log 和 $OUT/infer-gpu1.log" >&2
  exit 1
fi

python -u "$PY" merge --output "$OUT" | tee "$OUT/merge.log"
echo "完成 $OUT"
