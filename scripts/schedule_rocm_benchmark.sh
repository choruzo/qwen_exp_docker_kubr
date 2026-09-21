#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export TZ=Europe/Madrid
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_dir"
state_dir="$project_dir/.cache/rocm-benchmark-schedule"
mkdir -p "$state_dir/logs"
exec 9>"$state_dir/lock"
flock 9
clock="$(date +%H%M)"
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$state_dir/events.log"; }
name_for() { printf 'qwen35-rocm-benchmark-%s' "${1//_/-}"; }
container_state() { docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null || true; }
result_for() { printf '%s/benchmarks/rocm/hipblaslt_patched/%s_results.json' "$project_dir" "$1"; }
complete_result() { jq -e '.completion.full_test == true and .completion.generation == true' "$1" >/dev/null 2>&1; }
archive_logs() {
  local id
  id="$(docker inspect -f '{{.Id}}' "$1" 2>/dev/null || true)"
  if [[ -n "$id" ]]; then docker logs "$1" >"$state_dir/logs/$id.log" 2>&1 || true; fi
}
next_variant() {
  local variant result
  for variant in baseline finetuned_safetensors; do
    result="$(result_for "$variant")"
    if [[ -f "$result" ]]; then
      if complete_result "$result"; then continue; fi
      log "Incomplete result for $variant; manual diagnosis required: $result"
      return 1
    fi
    printf '%s\n' "$variant"
    return 0
  done
}
case "${1:-}" in
  start)
    if (( 10#$clock < 800 || 10#$clock >= 2330 )); then
      log 'Start event outside the 08:00–23:30 window; ignored.'
      exit 0
    fi
    for variant in baseline finetuned_safetensors; do
      name="$(name_for "$variant")"
      if [[ "$(container_state "$name")" == running ]]; then
        printf '%s\n' "$variant" >"$state_dir/active_variant"
        exit 0
      fi
    done
    variant="$(next_variant)" || exit 1
    if [[ -z "$variant" ]]; then
      log 'Both generation variants are complete; nothing to start.'
      exit 0
    fi
    name="$(name_for "$variant")"
    previous="$(sed -n '1p' "$state_dir/active_variant" 2>/dev/null || true)"
    resume="$(sed -n '1p' "$state_dir/resume_variant" 2>/dev/null || true)"
    if [[ "$previous" == "$variant" && "$resume" != "$variant" ]]; then
      log "Unexpected exit of $name; refusing an automatic retry. Inspect its logs."
      exit 1
    fi
    free_mb="$(amd-smi metric --gpu 0 2>/dev/null | awk '/FREE_VRAM:/ {print $2; exit}' || true)"
    if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < 23552 )); then
      log "GPU free VRAM ${free_mb:-unknown} MB; need 23552 MB before starting $variant."
      exit 0
    fi
    if [[ -n "$(container_state "$name")" ]]; then
      archive_logs "$name"
      docker rm "$name" >/dev/null
    fi
    docker compose -f compose.train.rocm.yaml -f compose.train.rocm.patched.yaml run -d --no-deps \
      --name "$name" train python -m docker_k8s_finetune.cli benchmark \
      --variant "$variant" --config config/benchmark.rocm.patched.yaml \
      --skip-judge --skip-syntax >/dev/null
    printf '%s\n' "$variant" >"$state_dir/active_variant"
    if [[ "$resume" == "$variant" ]]; then rm "$state_dir/resume_variant"; fi
    log "Started $variant from its deterministic generation cache."
    ;;
  stop)
    if (( 10#$clock < 2330 && 10#$clock >= 800 )); then
      log 'Stop event during evaluation hours; ignored.'
      exit 0
    fi
    for variant in baseline finetuned_safetensors; do
      name="$(name_for "$variant")"
      if [[ "$(container_state "$name")" == running ]]; then
        printf '%s\n' "$variant" >"$state_dir/resume_variant"
        archive_logs "$name"
        log "Stopping $variant at the end of the evaluation window."
        docker stop --time 30 "$name" >/dev/null
        log "Stopped $variant; next start will reuse its completed cache records."
      fi
    done
    ;;
  status)
    printf 'Local time: %s\n' "$(date --iso-8601=seconds)"
    for variant in baseline finetuned_safetensors; do
      name="$(name_for "$variant")"
      result="$(result_for "$variant")"
      state=absent
      if [[ -f "$result" ]]; then
        state=incomplete
        if complete_result "$result"; then state=complete; fi
      fi
      printf '%s: container=%s result=%s\n' "$variant" "$(container_state "$name")" "$state"
    done
    ;;
  *)
    printf 'Usage: %s {start|stop|status}\n' "$0" >&2
    exit 2
    ;;
esac
