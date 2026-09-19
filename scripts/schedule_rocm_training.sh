#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export TZ=Europe/Madrid
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_dir"

state_dir="$project_dir/.cache/rocm-training-schedule"
mkdir -p "$state_dir/logs"
exec 9>"$state_dir/lock"
flock 9

container=qwen35-rocm-full-patched
checkpoint_dir="$project_dir/artifacts/rocm/hipblaslt_patched/checkpoints/primary"
result_file="$project_dir/artifacts/rocm/hipblaslt_patched/metrics/training_result.json"
clock="$(date +%H%M)"
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$state_dir/events.log"; }
container_state() { docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true; }

case "${1:-}" in
  stop)
    if (( 10#$clock >= 1030 && 10#$clock < 2330 )); then
      log 'Stop timer fired during training hours; ignoring stale event.'
      exit 0
    fi
    if [[ "$(container_state)" == running ]]; then
      log 'Stopping training container at start of quiet hours.'
      docker stop --time 15 "$container" >/dev/null
      log 'Training container stopped; the next start will verify checkpoint provenance.'
    fi
    ;;
  start)
    if (( 10#$clock < 1030 || 10#$clock >= 2330 )); then
      log 'Start timer fired during quiet hours; ignoring stale event.'
      exit 0
    fi
    if [[ -f "$result_file" ]]; then
      log 'Training result already exists; no restart needed.'
      exit 0
    fi
    current_state="$(container_state)"
    if [[ "$current_state" == running || "$current_state" == paused ]]; then
      exit 0
    fi
    checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null)"
    if [[ -z "$checkpoint" ]]; then
      log 'No ROCm checkpoint found; refusing to restart from step zero.'
      exit 1
    fi
    free_mb="$(amd-smi metric --gpu 0 | awk '/FREE_VRAM:/ {print $2; exit}')"
    if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < 23552 )); then
      log "GPU has ${free_mb:-unknown} MB free; require at least 23552 MB. Retry at next timer tick."
      exit 0
    fi
    if [[ -n "$current_state" ]]; then
      container_id="$(docker inspect -f '{{.Id}}' "$container")"
      docker logs "$container" >"$state_dir/logs/${container_id}.log" 2>&1 || true
      docker rm "$container" >/dev/null
      log "Archived previous container log as ${container_id}.log."
    fi
    docker compose -f compose.train.rocm.yaml -f compose.train.rocm.patched.yaml run -d --no-deps \
      --name "$container" train python -m docker_k8s_finetune.cli train \
      --config config/training.rocm.patched.yaml --resume-from-latest >/dev/null
    log 'Started ROCm training with explicit latest compatible checkpoint resume.'
    ;;
  status)
    printf 'Local time: %s\nContainer: %s\n' "$(date --iso-8601=seconds)" "$(container_state)"
    if [[ -d "$checkpoint_dir" ]]; then
      find "$checkpoint_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -n 3
    fi
    ;;
  *)
    printf 'Usage: %s {start|stop|status}\n' "$0" >&2
    exit 2
    ;;
esac
