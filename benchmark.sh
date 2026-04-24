#!/usr/bin/env bash
# benchmark.sh — full evaluation sweep across graph configs and all providers.
# Provider order: OpenAI → Gemini → Groq → Ollama (cloud-first, local last).
set -euo pipefail

# Load .env so API keys are visible in this shell (python-dotenv handles it for
# orchestrator.py, but bash needs an explicit source).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/.env" ]] && set -a && source "$SCRIPT_DIR/.env" && set +a

# Automatically activate the virtual environment if it exists.
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Graph configurations (2 combinations) ────────────────────────────────────
# Each entry: "directed_flag|weighted_flag|dir_label|wt_label"
GRAPH_CONFIGS=(
    " | |undirected|unweighted"
    " |--gen-weighted|undirected|weighted"
    "--gen-directed| |directed|unweighted"
    "--gen-directed|--gen-weighted|directed|weighted"
)

SEED=1
SIZE="small"
SHOTS=(0 2)

# ── Models — cloud providers first, Ollama last ───────────────────────────────
declare -A PROVIDER_OF
declare -A ENV_KEY_OF

# OpenAI
OPENAI_MODELS=("gpt-4o-mini")
for m in "${OPENAI_MODELS[@]}"; do PROVIDER_OF[$m]="openai";  ENV_KEY_OF[$m]="OPENAI_API_KEY";  done

# Gemini  (auto-detected from model name — no LLM_PROVIDER needed)
GEMINI_MODELS=("gemini-2.0-flash")
for m in "${GEMINI_MODELS[@]}"; do PROVIDER_OF[$m]="gemini";  ENV_KEY_OF[$m]="GOOGLE_API_KEY"; done

# Groq  (requires LLM_PROVIDER=groq because names overlap with Ollama)
GROQ_MODELS=("llama-3.3-70b-versatile")
for m in "${GROQ_MODELS[@]}"; do PROVIDER_OF[$m]="groq";      ENV_KEY_OF[$m]="GROQ_API_KEY2";    done

# Ollama  (local — checked last)
OLLAMA_MODELS=("llama3.1:8b" "graphwalker:latest")
for m in "${OLLAMA_MODELS[@]}"; do PROVIDER_OF[$m]="ollama";  ENV_KEY_OF[$m]="";                 done

ALL_MODELS=(
    "${OPENAI_MODELS[@]}"
    "${OLLAMA_MODELS[@]}"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
graph_name() {
    # graph_name <dir_label> <wt_label>
    echo "erdos_renyi_${1}_${2}_${SIZE}_seed${SEED}"
}

safe_model() { echo "${1//:/_}"; }

csv_path() {
    # csv_path <graph_name> <model> <shots>
    local safe; safe=$(safe_model "$2")
    safe="${safe}_${3}shot"
    echo "results/${1}/${safe}/${safe}.csv"
}

has_key() {
    # has_key <ENV_VAR_NAME>  — returns true if the var is set and non-empty
    [[ -z "$1" ]] && return 0          # Ollama: no key required
    local val="${!1:-}"
    [[ -n "$val" ]]
}

ran=()
skipped=()
failed=()

# ── Main sweep ────────────────────────────────────────────────────────────────
for shots in "${SHOTS[@]}"; do
    for cfg in "${GRAPH_CONFIGS[@]}"; do
        IFS="|" read -r dir_flag wt_flag dir_label wt_label <<< "$cfg"
        dir_flag="${dir_flag// /}"   # trim
        wt_flag="${wt_flag// /}"
        gname=$(graph_name "$dir_label" "$wt_label")

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Graph: ${gname}  |  Shots: ${shots}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        for model in "${ALL_MODELS[@]}"; do
            provider="${PROVIDER_OF[$model]}"
            env_key="${ENV_KEY_OF[$model]}"
            csv=$(csv_path "$gname" "$model" "$shots")
            label="${model}  [${provider}]  shots=${shots}  graph=${gname}"

        # Skip: output already exists.
        if [[ -f "$csv" ]]; then
            echo "[SKIP] $label"
            echo "       → $csv"
            skipped+=("$label")
            continue
        fi

        # Skip: required API key is missing.
        if ! has_key "$env_key"; then
            echo "[SKIP] $label — ${env_key} not set in environment"
            skipped+=("$label")
            continue
        fi

        echo ""
        echo "[RUN]  $label"

        # Set LLM_PROVIDER for Groq (names overlap with Ollama).
        if [[ "$provider" == "groq" ]]; then
            export LLM_PROVIDER=groq
        else
            unset LLM_PROVIDER 2>/dev/null || true
        fi

        # Build the orchestrator command.
        cmd=(python orchestrator.py
            --generate
            --gen-size  "$SIZE"
            --gen-seed  "$SEED"
            --model     "$model"
            --descriptors adjacency_list edge_list gml l2sp_paths random_walk
            --few-shot-k "$shots"
        )
        [[ -n "$dir_flag" ]] && cmd+=("$dir_flag")
        [[ -n "$wt_flag"  ]] && cmd+=("$wt_flag")

        if "${cmd[@]}"; then
            ran+=("$label")
            echo "[OK]   $label"
        else
            failed+=("$label")
            echo "[FAIL] $label"
        fi
    done
  done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                     BENCHMARK SUMMARY                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  %-10s %d run(s)\n" "Ran:"     "${#ran[@]}"
printf "║  %-10s %d run(s)\n" "Skipped:" "${#skipped[@]}"
printf "║  %-10s %d run(s)\n" "Failed:"  "${#failed[@]}"
echo "╚══════════════════════════════════════════════════════════╝"

if [[ ${#ran[@]} -gt 0 ]]; then
    echo ""
    echo "Completed:"
    printf "  ✓ %s\n" "${ran[@]}"
fi

if [[ ${#failed[@]} -gt 0 ]]; then
    echo ""
    echo "Failed:"
    printf "  ✗ %s\n" "${failed[@]}"
    exit 1
fi
