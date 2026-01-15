#!/usr/bin/env bash
#
# ToolOrchestra "small scale" end-to-end demo training script
# ----------------------------------------------------------
# What this script does (in ONE place):
#  1) Create a smaller training JSONL from the repo's `data/data.jsonl` (QA-only subset)
#  2) Define small training hyperparameters (few steps, few turns, tiny batches)
#  3) Start essential services:
#     - a lightweight retrieval server stub (so you don't need INDEX_DIR/Tavily for a demo)
#     - a single vLLM "expert model" server (all tools map to this one model)
#  4) Run the existing RL trainer (GRPO) and save outputs to a new folder
#  5) Print a simple summary + generate a simple plot from logs (reward vs step)
#
# This script intentionally *repurposes* existing repo flows:
# - Training launcher pattern: inspired by `training/train_orchestrator.sh`
# - "Serve sidecars" pattern: inspired by `training/se_t4_1.sh`
# - Trainer entrypoint: `python -m recipe.algo.main_grpo_quick3` (same as `training/train_orchestrator.sh`)
#
# Notes:
# - This is meant to be an **educational** small run to confirm the pipeline works, not to match paper scores.
# - It assumes you're running on a machine with CUDA GPUs if you enable vLLM.
#
# Usage (example):
#   bash training/run_small_orchestrator_demo.sh
#
# Optional environment variables:
#   ORCH_BASE_MODEL         HF model name to fine-tune as orchestrator (default: Qwen/Qwen3-8B)
#   ORCH_TP_SIZE            tensor parallel size for orchestrator vLLM rollout (default: 2)
#   ORCH_NUM_GPUS           number of GPUs the trainer should use on this node (default: 2)
#   CUDA_VISIBLE_DEVICES    which GPU(s) to use (default unchanged)
#   NUM_EXAMPLES            number of QA examples to sample (default: 64)
#   MAX_TURNS               max turns per rollout episode (default: 5)
#   RETRIEVER_PORT          port for retrieval server (default: 18001)
#   START_LOCAL_EXPERT_VLLM start a local vLLM expert server anyway (default: 0). Not needed if all experts are OpenRouter.
#   EXPERT_MODEL            HF model name served by local vLLM when START_LOCAL_EXPERT_VLLM=1 (default: microsoft/Phi-4-mini-instruct)
#   EXPERT_VLLM_PORT        port for local vLLM expert server (default: 18002)
#   OUT_ROOT                root output folder (default: training/outputs/small_demo)
#   SKIP_VLLM               set to 1 to skip starting vLLM (use only if you already have an endpoint running)
#
# OpenRouter expert control:
#   OPENROUTER_API_KEY                 required for OpenRouter experts
#   OPENROUTER_EXPERTS_BY_KEY_JSON     optional: map tool keys -> OpenRouter model IDs (without the "openrouter/" prefix).
#                                     Example:
#                                       export OPENROUTER_EXPERTS_BY_KEY_JSON='{
#                                         "search-1":"openai/gpt-4o-mini",
#                                         "search-2":"openai/gpt-4o-mini",
#                                         "search-3":"openai/gpt-4o-mini",
#                                         "reasoner-1":"anthropic/claude-3.5-sonnet",
#                                         "reasoner-2":"openai/gpt-4o",
#                                         "reasoner-3":"qwen/qwen-2.5-coder-32b-instruct",
#                                         "answer-math-1":"qwen/qwen-2.5-math-72b-instruct",
#                                         "answer-math-2":"qwen/qwen-2.5-math-7b-instruct",
#                                         "answer-1":"openai/gpt-4o",
#                                         "answer-2":"openai/gpt-4o-mini",
#                                         "answer-3":"meta-llama/llama-3.3-70b-instruct",
#                                         "answer-4":"microsoft/phi-4-mini-instruct"
#                                       }'
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/training"

# -------------------------------------------------------------------
# AMD/ROCm GPU visibility (Ray requirement)
# -------------------------------------------------------------------
# Ray's AMD GPU detection errors out if ROCR_VISIBLE_DEVICES is set:
#   "Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES"
# To make the script robust on ROCm clusters:
# - If ROCR_VISIBLE_DEVICES is set and HIP_VISIBLE_DEVICES is not, we copy it.
# - We then unset ROCR_VISIBLE_DEVICES to avoid Ray raising.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
  fi
  unset ROCR_VISIBLE_DEVICES
fi

ORCH_BASE_MODEL="${ORCH_BASE_MODEL:-Qwen/Qwen3-8B}"
ORCH_TP_SIZE="${ORCH_TP_SIZE:-2}"
ORCH_NUM_GPUS="${ORCH_NUM_GPUS:-2}"
NUM_EXAMPLES="${NUM_EXAMPLES:-64}"
MAX_TURNS="${MAX_TURNS:-5}"
RETRIEVER_PORT="${RETRIEVER_PORT:-18001}"
START_LOCAL_EXPERT_VLLM="${START_LOCAL_EXPERT_VLLM:-0}"
EXPERT_MODEL="${EXPERT_MODEL:-microsoft/Phi-4-mini-instruct}"
EXPERT_VLLM_PORT="${EXPERT_VLLM_PORT:-18002}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/training/outputs/small_demo}"
SKIP_VLLM="${SKIP_VLLM:-0}"

# Export variables used by the embedded Python snippets below.
export REPO_ROOT EXPERT_MODEL ORCH_BASE_MODEL ORCH_TP_SIZE ORCH_NUM_GPUS NUM_EXAMPLES MAX_TURNS RETRIEVER_PORT EXPERT_VLLM_PORT OUT_ROOT SKIP_VLLM
export START_LOCAL_EXPERT_VLLM

if [[ "$ORCH_BASE_MODEL" == openrouter/* ]]; then
  echo "[demo] ERROR: ORCH_BASE_MODEL is set to an OpenRouter model id."
  echo "[demo] RL training requires a local HF checkpoint path/name because we need gradients."
  echo "[demo] You *can* use OpenRouter for experts, but the orchestrator being trained must be local."
  exit 2
fi

RUN_ID="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "[demo] repo: $REPO_ROOT"
echo "[demo] out:  $OUT_DIR"
echo "[demo] orchestrator base model: $ORCH_BASE_MODEL"
echo "[demo] orchestrator TP size (vLLM): $ORCH_TP_SIZE"
echo "[demo] trainer GPUs per node: $ORCH_NUM_GPUS"
echo "[demo] num examples: $NUM_EXAMPLES"
echo "[demo] max turns: $MAX_TURNS"

cleanup() {
  set +e
  if [[ -n "${RETRIEVER_PID:-}" ]]; then
    echo "[demo] stopping retriever (pid=$RETRIEVER_PID)"
    kill "$RETRIEVER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${VLLM_PID:-}" ]]; then
    echo "[demo] stopping vLLM expert server (pid=$VLLM_PID)"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

########################################
# (1) Create a smaller QA-only JSONL
########################################
# Source data: repo `data/data.jsonl` (same file referenced in `training/train_orchestrator.sh`)
#
# We keep only `"category": "qa"` examples so the rollout loop looks like HLE/FRAMES QA
# and avoids the tau2 "func_call" side pipeline.
#
# We now keep MULTIPLE expert models (like the project describes), but route them to OpenRouter.
# This mirrors the idea of using "generalist + specialized" experts (e.g., GPT-like, Claude-like,
# coding, math) without hosting each of them locally. See the project description for examples
# of the expert model/tool diversity: `https://research.nvidia.com/labs/lpr/ToolOrchestra/`.
#
SMALL_TRAIN_JSONL="$OUT_DIR/small_train_qa.jsonl"
export SMALL_TRAIN_JSONL
python3 - <<'PY'
import json, os, random
import re

repo_root = os.environ["REPO_ROOT"]
in_path = os.path.join(repo_root, "data", "data.jsonl")
out_path = os.environ["SMALL_TRAIN_JSONL"]
num_examples = int(os.environ["NUM_EXAMPLES"])
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    raise SystemExit("OPENROUTER_API_KEY is not set (required when using multiple OpenRouter experts).")

def hf_to_openrouter_id(hf_name: str) -> str:
    """
    Best-effort conversion from HF-style 'Org/Model-Name' to OpenRouter-style 'org/model-name'.
    This is not guaranteed for every model; you can override via OPENROUTER_MODEL_MAP_JSON.
    """
    parts = hf_name.split("/", 1)
    if len(parts) != 2:
        return hf_name.lower()
    org, name = parts[0].lower(), parts[1].lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return f"{org}/{name}"

# Optional: explicit mapping (JSON dict) to avoid naming mismatches.
# Example:
#   export OPENROUTER_MODEL_MAP_JSON='{"gpt-5":"openai/gpt-4o","Qwen/Qwen2.5-Math-72B-Instruct":"qwen/qwen-2-5-math-72b-instruct"}'
explicit_map = {}
if os.getenv("OPENROUTER_MODEL_MAP_JSON"):
    explicit_map = json.loads(os.environ["OPENROUTER_MODEL_MAP_JSON"])
    if not isinstance(explicit_map, dict):
        raise SystemExit("OPENROUTER_MODEL_MAP_JSON must be a JSON object (dict)")

random.seed(0)

qa = []
with open(in_path, "r") as f:
    for line in f:
        ex = json.loads(line)
        if ex.get("category") == "qa":
            qa.append(ex)

if len(qa) < num_examples:
    raise SystemExit(f"Not enough QA examples in {in_path}. Found {len(qa)}, need {num_examples}.")

sample = qa[:num_examples]  # deterministic small slice for beginners

for ex in sample:
    # =========================
    # BEGIN OPENROUTER PATCH (demo script)
    # =========================
    # Convert all mapped expert model names to "openrouter/<provider>/<model>" ids.
    # The training runtime routes these through the OpenRouter backend we added to LLM_CALL.py.
    #
    # IMPORTANT: We also avoid local-vLLM-only branches by making the mapping values start with "openrouter/".
    if "model_mapping" in ex and isinstance(ex["model_mapping"], dict):
        experts_by_key = {}
        if os.getenv("OPENROUTER_EXPERTS_BY_KEY_JSON"):
            experts_by_key = json.loads(os.environ["OPENROUTER_EXPERTS_BY_KEY_JSON"])
            if not isinstance(experts_by_key, dict):
                raise SystemExit("OPENROUTER_EXPERTS_BY_KEY_JSON must be a JSON object (dict)")
        for k, v in list(ex["model_mapping"].items()):
            if not isinstance(v, str):
                continue
            # Highest priority: explicit per-tool-key mapping (search-1, answer-math-1, etc.)
            if k in experts_by_key and isinstance(experts_by_key[k], str) and experts_by_key[k]:
                ex["model_mapping"][k] = "openrouter/" + experts_by_key[k]
                continue
            # Keep already-converted values
            if v.startswith("openrouter/"):
                continue
            # Replace NVIDIA-internal "nvdev/..." names with something usable via OpenRouter if provided
            if v in explicit_map:
                ex["model_mapping"][k] = "openrouter/" + explicit_map[v]
            elif v in ("gpt-5", "gpt-5-mini"):
                # OpenRouter may not expose "gpt-5" ids; default to close public equivalents.
                fallback = "openai/gpt-4o" if v == "gpt-5" else "openai/gpt-4o-mini"
                ex["model_mapping"][k] = "openrouter/" + fallback
            elif "/" in v:
                ex["model_mapping"][k] = "openrouter/" + hf_to_openrouter_id(v)
            else:
                # last resort: treat as already an OpenRouter provider/model id
                ex["model_mapping"][k] = "openrouter/" + v

    # Ensure tool_pricing includes entries for the OpenRouter-mapped model ids (placeholders).
    tp = ex.get("tool_pricing") or {}
    if "model_mapping" in ex and isinstance(ex["model_mapping"], dict):
        for v in ex["model_mapping"].values():
            if isinstance(v, str) and v.startswith("openrouter/"):
                tp.setdefault(v, {"input_tokens_per_million": 0.0, "output_tokens_per_million": 0.0})
    ex["tool_pricing"] = tp
    # =========================
    # END OPENROUTER PATCH (demo script)
    # =========================

with open(out_path, "w") as f:
    for ex in sample:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"[demo] wrote {len(sample)} examples -> {out_path}")

# Print the unique OpenRouter expert ids we will call (so it's explicit).
experts = set()
for ex in sample:
    mm = ex.get("model_mapping") or {}
    if isinstance(mm, dict):
        for v in mm.values():
            if isinstance(v, str) and v.startswith("openrouter/"):
                experts.add(v)
experts = sorted(experts)
print(f"[demo] unique OpenRouter experts (count={len(experts)}):")
for e in experts:
    print("  -", e)
with open(os.path.join(os.path.dirname(out_path), "experts_openrouter.txt"), "w") as f:
    f.write("\n".join(experts) + "\n")
PY

########################################
# (2) Write a tiny tool config (optional)
########################################
# We reuse the existing tool schema used in evaluation/training:
# - `training/tools.json` matches the tool signatures (search/answer/enhance_reasoning)
# We'll just copy it into OUT_DIR to keep the run self-contained and reproducible.
cp "$REPO_ROOT/training/tools.json" "$OUT_DIR/tools.json"

########################################
# (3) Start essential components
########################################

## (3a) Lightweight retrieval server stub
#
# Why stub? `training/retrieval_general_thought.py` requires:
# - INDEX_DIR containing a FAISS index + corpus
# - TAVILY_KEY
# For a beginner "pipeline sanity" run, that’s too much friction.
#
# This stub mimics the response shape expected by the rollout code:
# - POST /retrieve  {"queries": ["..."], "topk": 5, "return_scores": true, "eid": "..."}
# - Returns: [[ {"document": {"content": "..."}, "score": 1.0}, ... ]]
#
RETRIEVER_STUB="$OUT_DIR/retriever_stub.py"
cat > "$RETRIEVER_STUB" <<'PY'
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 5
    return_scores: bool = True
    eid: Optional[str] = None

app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    q = request.queries[0] if request.queries else ""
    topk = request.topk or 5
    # Simple deterministic "documents" so you can see them show up in prompts
    docs = []
    for i in range(topk):
        docs.append({
            "document": {"content": f"[stub-doc-{i}] This is a stub retrieved passage for query: {q}"},
            "score": 1.0 - i * 0.01
        })
    return [docs]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
PY

echo "[demo] starting retriever stub on :$RETRIEVER_PORT"
python3 "$RETRIEVER_STUB" --port "$RETRIEVER_PORT" >"$OUT_DIR/retriever.log" 2>&1 &
RETRIEVER_PID="$!"

## (3b) Optional local vLLM expert server (one model only)
#
# This is inspired by `training/se_t4_1.sh` which does:
#   vllm serve microsoft/Phi-4-mini-instruct --port 1408
#
if [[ "$START_LOCAL_EXPERT_VLLM" == "1" && "$SKIP_VLLM" != "1" ]]; then
  echo "[demo] starting local vLLM expert server on :$EXPERT_VLLM_PORT (model=$EXPERT_MODEL)"
  # NOTE: You must have vLLM installed in your env. If you use conda like the README suggests,
  # run this script from that env (e.g. the repo's `vllm1` env).
  vllm serve "$EXPERT_MODEL" --host 0.0.0.0 --port "$EXPERT_VLLM_PORT" >"$OUT_DIR/vllm_expert.log" 2>&1 &
  VLLM_PID="$!"

  echo "[demo] waiting for vLLM to become ready..."
  python3 - <<'PY'
import os, time, urllib.request, json
port = int(os.environ["EXPERT_VLLM_PORT"])
url = f"http://127.0.0.1:{port}/v1/models"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            json.loads(r.read().decode("utf-8"))
        print("[demo] vLLM ready")
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("[demo] vLLM did not become ready in time. Check vllm_expert.log")
PY
else
  echo "[demo] not starting local expert vLLM (START_LOCAL_EXPERT_VLLM=$START_LOCAL_EXPERT_VLLM, SKIP_VLLM=$SKIP_VLLM)."
fi

########################################
# (3c) Write a minimal vLLM model config JSON for tools
########################################
# The rollout code expects `vllm_model_configs` to contain:
# - "retrieval": [{ip_addr, port}]
# - sometimes "wiki_retrieval": [{ip_addr, port}] (used when index starts with "wiki")
# - plus "vllm_model_config_path" (self-reference)
VLLM_CONFIG="$OUT_DIR/serve_small.json"
export VLLM_CONFIG
python3 - <<'PY'
import json, os

out_path = os.environ["VLLM_CONFIG"]
retriever_port = int(os.environ["RETRIEVER_PORT"])

cfg = {
    "retrieval": [{"ip_addr": "127.0.0.1", "port": str(retriever_port)}],
    "wiki_retrieval": [{"ip_addr": "127.0.0.1", "port": str(retriever_port)}],
    "vllm_model_config_path": out_path,
}

with open(out_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"[demo] wrote vllm model config -> {out_path}")
PY

########################################
# (4) Run the trainer (GRPO) with tiny parameters
########################################
#
# This reuses the exact same entrypoint as `training/train_orchestrator.sh`:
#   python -m recipe.algo.main_grpo_quick3
#
# Differences from the full run:
# - much smaller batches
# - fewer turns
# - fewer epochs
# - console-only logging
#
# Important: disable the LLM-as-judge step for QA mismatch,
# otherwise the rollout tries to call GPT-5 (see patch in generation_quick3.py).
export TOOLORCHESTRA_DISABLE_LLM_JUDGE=1
export REPO_ROOT

TRAIN_LOG="$OUT_DIR/train.log"
export TRAIN_LOG OUT_DIR

echo "[demo] starting training... (logging to $TRAIN_LOG)"
python3 -u -m recipe.algo.main_grpo_quick3 \
  +data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  data.train_files="['$SMALL_TRAIN_JSONL']" \
  data.val_files="['$SMALL_TRAIN_JSONL']" \
  data.train_tool_config_path="$OUT_DIR/tools.json" \
  data.test_tool_config_path="$OUT_DIR/tools.json" \
  +data.vllm_model_configs="$VLLM_CONFIG" \
  +data.my_output_dir="$OUT_DIR/run_artifacts" \
  +data.cur_transfer_dir="$OUT_DIR/transfer" \
  +data.model_type="Qwen/Qwen3-8B" \
  +data.topk_doc=3 \
  +data.exp_tag="small_demo" \
  +data.use_llm_reward=false \
  +data.efficiency_reward=false \
  +data.use_qa_reward=true \
  data.prompt_template=qwen-base \
  data.train_batch_size=2 \
  data.gen_batch_size=8 \
  data.val_batch_size=8 \
  data.max_prompt_length=4096 \
  data.max_response_length=768 \
  +max_turns="$MAX_TURNS" \
  actor_rollout_ref.model.path="$ORCH_BASE_MODEL" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.n=1 \
  +actor_rollout_ref.rollout.n_agent=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ORCH_TP_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  algorithm.use_kl_in_reward=false \
  trainer.logger="['console']" \
  trainer.project_name="small_demo" \
  trainer.experiment_name="$RUN_ID" \
  trainer.val_before_train=false \
  trainer.n_gpus_per_node="$ORCH_NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=0 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="$OUT_DIR/ckpt" \
  +retriever.url="http://127.0.0.1:$RETRIEVER_PORT/retrieve" \
  +retriever.topk=3 \
  reward_manager.type=match \
  reward_manager.max_concurrency=8 \
  2>&1 | tee "$TRAIN_LOG"

########################################
# (5) Simple numbers + plot
########################################
#
# We try to extract JSON-ish metric payloads from the training log and plot reward over steps.
# If your logger format is different on your setup, you can still inspect `$TRAIN_LOG`.
python3 - <<'PY'
import os, re, json
from pathlib import Path

log_path = Path(os.environ["TRAIN_LOG"])
out_dir = Path(os.environ["OUT_DIR"])

steps = []
rewards = []

def try_parse_json_from_line(line: str):
    # Heuristic: find the first {...} region and try json.loads
    i = line.find("{")
    j = line.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    blob = line[i:j+1]
    try:
        return json.loads(blob)
    except Exception:
        return None

for line in log_path.read_text(errors="ignore").splitlines():
    obj = try_parse_json_from_line(line)
    if isinstance(obj, dict):
        # common keys in verl metrics
        # rewards often appear as critic/rewards/mean or reward/mean depending on config
        step = obj.get("step") or obj.get("global_step") or obj.get("trainer/global_step")
        r = obj.get("critic/rewards/mean") or obj.get("critic/reward/mean") or obj.get("reward/mean") or obj.get("train/seq_reward_mean")
        if step is not None and r is not None:
            steps.append(float(step))
            rewards.append(float(r))

# Fallback: regex parse for something like "critic/rewards/mean": 0.123
if not rewards:
    rgx = re.compile(r"(critic/rewards/mean|reward/mean)[^0-9-]*([-]?\d+(?:\.\d+)?)")
    step_rgx = re.compile(r"step[^0-9]*([0-9]+)")
    cur_step = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = step_rgx.search(line)
        if m:
            cur_step = float(m.group(1))
        m2 = rgx.search(line)
        if m2 and cur_step is not None:
            steps.append(cur_step)
            rewards.append(float(m2.group(2)))

summary_path = out_dir / "summary.txt"
summary = []
summary.append(f"Log: {log_path}")
summary.append(f"Parsed points: {len(rewards)}")
if rewards:
    summary.append(f"Reward min/mean/max: {min(rewards):.4f} / {sum(rewards)/len(rewards):.4f} / {max(rewards):.4f}")
else:
    summary.append("No reward series could be parsed from logs. Inspect train.log manually.")

summary_path.write_text("\n".join(summary) + "\n")
print("\n".join(summary))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if rewards:
        plt.figure(figsize=(7,4))
        plt.plot(steps, rewards, marker="o")
        plt.title("Small-demo training: reward vs step")
        plt.xlabel("step")
        plt.ylabel("reward (parsed)")
        plt.grid(True, alpha=0.3)
        fig_path = out_dir / "reward_curve.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        print(f"Wrote plot: {fig_path}")
except Exception as e:
    print(f"Plot skipped (matplotlib missing or error): {e}")
PY

echo "[demo] done. outputs in: $OUT_DIR"

