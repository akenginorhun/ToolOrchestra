# ./scripts/eval_model.sh --model /lustre/fsw/portfolios/nvr/users/mingjiel/models/DeepSeek-R1-Distill-Qwen-1.5B --data-dir /lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler --output-dir /lustre/fsw/portfolios/nvr/users/mingjiel/results/tests/DeepSeek-R1-Distill-Qwen-1.5B
# ./scripts/eval_model.sh --model /lustre/fsw/portfolios/nvr/users/mingjiel/results/deepscaler/deepscaler-1.5b-16k-reset/ckpt/global_step_224/actor/huggingface --data-dir /lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler --output-dir /lustre/fsw/portfolios/nvr/users/mingjiel/results/tests/deepscaler-1.5b-16k-reset/ckpt/global_step_224
# export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("aime") # "amc" "match" "minverva" "olimpiad_bench")
OUTPUT_DIR="$HOME"  # Add default output directory
DATA_PATH="$HOME/deepscaler/data"
TEMPERATURE=0.6
N_SAMPLES=16
RESPONSE_LENGTH=32768
NUM_NODES=1
SERVER_IP="localhost"
BATCH_SIZE=2048
# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --response-length)
            RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --data-dir)
            DATA_PATH="$2"
            shift 2
            ;;
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --server-ip)
            SERVER_IP="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.nvidia.eval.general_eval \
        trainer.nnodes=${NUM_NODES} \
        trainer.n_gpus_per_node=8 \
        data.train_files=$DATA_PATH/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}_${TEMPERATURE}_${RESPONSE_LENGTH}_${N_SAMPLES}.parquet \
        data.train_batch_size=${BATCH_SIZE} \
        data.max_prompt_length=2048 \
        data.truncation='left' \
        data.max_response_length=${RESPONSE_LENGTH} \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE} \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.n=${N_SAMPLES} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        reward_manager.type=prime \
        reward_manager.max_concurrency=256 \
        reward_manager.use_remote_reward=True \
        reward_manager.server_ip=[${SERVER_IP}]
done
